import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Hypothetical Python wrappers for the open-source linguistic engines
from external_nlp import VidyutParser, HeritageSegmenter
from custom_kernels import sparse_paninian_attention # The custom C++/Triton kernel


# ==============================================================================
# DHAATU DATABASE — The Foundation of Factorized Embeddings
# ==============================================================================

class DhaatuDatabase:
    """
    Sanskrit Verb Root (Dhātu) Database for Panini-LM.
    
    Loads the structured JSON database of ~2259 dhātus from Siddhānta-Kaumudī,
    enabling:
    - Direct root_id lookup for factorized embeddings
    - Grammatical feature extraction (gaṇa, pada, transitivity, seṭ/aniṭ)
    - Semantic search via English/Hindi meanings
    
    This replaces the need for a 50,000+ token vocabulary with ~2300 semantic
    primitives that can compose ANY valid Sanskrit verbal form.
    """
    
    def __init__(self, json_path: Optional[Path] = None):
        if json_path is None:
            json_path = Path(__file__).parent / "data" / "dhaatu.json"
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.metadata = data["_metadata"]
        self.dhatus = data["dhatus"]
        
        # Build lookup indices for O(1) access
        self._root_to_id: Dict[str, int] = {}
        self._id_to_dhaatu: Dict[int, dict] = {}
        
        for dhaatu in self.dhatus:
            dhaatu_id = dhaatu["id"]
            root = dhaatu["root"]
            
            self._root_to_id[root] = dhaatu_id
            self._id_to_dhaatu[dhaatu_id] = dhaatu
            
            # Also index by root_with_markers for flexibility
            if "root_with_markers" in dhaatu:
                self._root_to_id[dhaatu["root_with_markers"]] = dhaatu_id
    
    @property
    def num_roots(self) -> int:
        """Total number of dhātus in the database (~2259)."""
        return self.metadata["total_roots"]
    
    @property 
    def gana_distribution(self) -> Dict[str, int]:
        """Distribution of roots across the 10 gaṇas (verb classes)."""
        return self.metadata["gana_counts"]
    
    def get_root_id(self, root: str) -> int:
        """
        Get the embedding ID for a Sanskrit root.
        
        Args:
            root: Sanskrit root in Devanagari (e.g., "भू", "गम्", "कृ")
        
        Returns:
            Integer ID for embedding lookup (0 to num_roots-1)
        
        Raises:
            KeyError: If root not found in database
        """
        if root in self._root_to_id:
            return self._root_to_id[root]
        raise KeyError(f"Unknown dhātu: {root}")
    
    def get_dhaatu(self, dhaatu_id: int) -> dict:
        """
        Get full dhātu information by ID.
        
        Returns dict with: root, gana, pada, transitivity, set_type, meanings
        """
        return self._id_to_dhaatu[dhaatu_id]
    
    def get_by_root(self, root: str) -> Optional[dict]:
        """Get full dhātu information by root string."""
        if root in self._root_to_id:
            return self._id_to_dhaatu[self._root_to_id[root]]
        return None
    
    def get_gana(self, root: str) -> Optional[str]:
        """Get the gaṇa (verb class) for a root."""
        dhaatu = self.get_by_root(root)
        return dhaatu.get("gana") if dhaatu else None
    
    def get_pada(self, root: str) -> Optional[str]:
        """Get the pada (voice type: parasmaipadi/atmanepadi/ubhayapadi)."""
        dhaatu = self.get_by_root(root)
        return dhaatu.get("pada") if dhaatu else None
    
    def get_transitivity(self, root: str) -> Optional[str]:
        """Get transitivity (akarmaka/sakarmaka/dvikarmaka)."""
        dhaatu = self.get_by_root(root)
        return dhaatu.get("transitivity") if dhaatu else None
    
    def search_by_meaning(self, query: str, language: str = "en") -> List[dict]:
        """
        Search dhātus by English or Hindi meaning.
        
        Args:
            query: Search term (case-insensitive)
            language: "en" for English, "hi" for Hindi
        
        Returns:
            List of matching dhātu entries
        """
        query_lower = query.lower()
        results = []
        
        key = "meanings_en" if language == "en" else "meanings_hi"
        
        for dhaatu in self.dhatus:
            meanings = dhaatu.get(key, [])
            for meaning in meanings:
                if query_lower in meaning.lower():
                    results.append(dhaatu)
                    break
        
        return results
    
    def get_roots_by_gana(self, gana: str) -> List[dict]:
        """Get all roots belonging to a specific gaṇa."""
        return [d for d in self.dhatus if d.get("gana") == gana]
    
    def get_roots_by_pada(self, pada: str) -> List[dict]:
        """Get all roots with a specific pada (voice type)."""
        return [d for d in self.dhatus if d.get("pada") == pada]


# Global database instance (lazy-loaded)
_dhaatu_db: Optional[DhaatuDatabase] = None

def get_dhaatu_database() -> DhaatuDatabase:
    """Get the global DhaatuDatabase instance (lazy initialization)."""
    global _dhaatu_db
    if _dhaatu_db is None:
        _dhaatu_db = DhaatuDatabase()
    return _dhaatu_db


class PaninianEmbedding(nn.Module):
    """
    Factorized Embedding Layer — THE KEY INNOVATION OF PANINI-LM.
    
    Traditional Transformers: nn.Embedding(50000, d_model) → 25.6M parameters
    Panini-LM: Additive composition of small embedding matrices → 2.1M parameters
    
    The neural network sees each word as the SUM of its mathematical parts:
        E(gacchati) = E(√gam) + E(tiṅanta) + E(laṭ) + E(prathama) + E(eka)
    
    Now uses the actual DhaatuDatabase for:
    - Accurate root count (~2259 dhātus)
    - Gaṇa (verb class) embeddings for conjugation patterns
    - Pada (voice type) embeddings for semantic nuances
    
    Benefits:
    1. ZERO OOV for inflections — all valid forms are compositionally derivable
    2. 12× parameter reduction in embeddings
    3. Morphological knowledge is encoded structurally, not learned implicitly
    """
    
    def __init__(self, d_model: int = 512, dhaatu_db: Optional[DhaatuDatabase] = None):
        super().__init__()
        
        # Load or use provided DhaatuDatabase
        self.dhaatu_db = dhaatu_db or get_dhaatu_database()
        
        # === Core Semantic Embeddings (Roots & Stems) ===
        # ~2259 dhātus from database + ~1500 prātipadikas + special tokens
        num_dhatus = self.dhaatu_db.num_roots
        num_pratipadikas = 1500  # Nominal stems (to be loaded from separate database)
        num_special = 200       # upasargas, pratyayas, special tokens
        total_roots = num_dhatus + num_pratipadikas + num_special
        
        self.root_embed = nn.Embedding(total_roots, d_model)
        
        # === Grammatical Meta-Data Embeddings (Tiny Matrices) ===
        self.type_embed = nn.Embedding(7, d_model)      # subanta, tiṅanta, avyaya, kṛdanta, taddhita, samāsa, none
        self.vibhakti_embed = nn.Embedding(9, d_model)  # 1-7 + vocative + none
        self.vacana_embed = nn.Embedding(4, d_model)    # singular, dual, plural, none
        self.purusa_embed = nn.Embedding(4, d_model)    # 3rd, 2nd, 1st, none (Sanskrit convention)
        
        # === NEW: Dhātu-specific embeddings from database ===
        self.gana_embed = nn.Embedding(11, d_model)     # 10 gaṇas + none
        self.pada_embed = nn.Embedding(4, d_model)      # parasmaipadi, atmanepadi, ubhayapadi, none
        self.transitivity_embed = nn.Embedding(4, d_model)  # akarmaka, sakarmaka, dvikarmaka, none
        
    def forward(
        self,
        root_ids: torch.LongTensor,
        type_ids: torch.LongTensor,
        vibhakti_ids: torch.LongTensor,
        vacana_ids: torch.LongTensor,
        purusa_ids: torch.LongTensor,
        gana_ids: Optional[torch.LongTensor] = None,
        pada_ids: Optional[torch.LongTensor] = None,
        transitivity_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """
        Construct embeddings by summing morphological components.
        
        Args:
            root_ids: (batch, seq) — Root/stem IDs (the semantic core)
            type_ids: (batch, seq) — Token type (subanta/tiṅanta/etc.)
            vibhakti_ids: (batch, seq) — Case (1-7, vocative, or none)
            vacana_ids: (batch, seq) — Number (singular/dual/plural/none)
            purusa_ids: (batch, seq) — Person (3rd/2nd/1st/none)
            gana_ids: (batch, seq) — Gaṇa/verb class (1-10, or none) [optional]
            pada_ids: (batch, seq) — Voice type (P/A/U/none) [optional]
            transitivity_ids: (batch, seq) — Transitivity [optional]
        
        Returns:
            (batch, seq, d_model) — Factorized embeddings
        """
        # The neural network sees the word as a sum of its mathematical parts.
        embedding = (
            self.root_embed(root_ids) +
            self.type_embed(type_ids) +
            self.vibhakti_embed(vibhakti_ids) +
            self.vacana_embed(vacana_ids) +
            self.purusa_embed(purusa_ids)
        )
        
        # Add dhātu-specific features if provided (for verbal forms)
        if gana_ids is not None:
            embedding = embedding + self.gana_embed(gana_ids)
        if pada_ids is not None:
            embedding = embedding + self.pada_embed(pada_ids)
        if transitivity_ids is not None:
            embedding = embedding + self.transitivity_embed(transitivity_ids)
        
        return embedding


class PaninianNeuroSymbolicLLM(nn.Module):
    """
    The Panini-LM Architecture: A Neuro-Symbolic Language Model for Sanskrit.
    
    Key Innovation: Factorized Embeddings using DhaatuDatabase
        - Uses actual ~2259 dhātu roots from Siddhānta-Kaumudī
        - Zero OOV errors for any valid Sanskrit inflection
        - 12× parameter reduction in embedding layer
        - Morphological structure preserved, not learned implicitly
    """
    
    def __init__(self, d_model: int = 512, num_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        
        # Load the DhaatuDatabase
        self.dhaatu_db = get_dhaatu_database()
        
        # Vocabulary = dhātus + prātipadikas + special tokens
        self.vocab_size = self.dhaatu_db.num_roots + 1700  # ~2259 + 1700 = ~4000
        
        # ==========================================
        # TRACK A: The Symbolic Engine (Syntax)
        # ==========================================
        self.segmenter = HeritageSegmenter() # Handles Sandhi & Samasa (Phase 1)
        self.morph_analyzer = VidyutParser() # Extracts Purusa, Vacana, etc.
        
        # ==========================================
        # TRACK B: The Neural Engine (Semantics)
        # ==========================================
        # Phase 2B: FACTORIZED Position-Agnostic Embeddings 
        # Note: We explicitly DO NOT instantiate RoPE or absolute positional encodings here.
        # Note: We use FACTORIZED embeddings with DhaatuDatabase
        self.embedding = PaninianEmbedding(d_model, self.dhaatu_db)
        
        # Linear projections for Attention
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        # Phase 4: Semantic Maturation (Dense network)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(), # Swish activation
            nn.Linear(d_model * 4, d_model)
        )
        
        # Phase 5: Vocabulary Projection (to ~4000 roots, NOT 50,000 surface forms)
        self.lm_head = nn.Linear(d_model, self.vocab_size)

    def _build_adjacency_matrix(self, metadata: list) -> torch.Tensor:
        """
        Phase 2A: The core logic that evaluates Ashtadhyayi rules to map Karaka links.
        Returns an N x N tensor populated with 0.0 (valid) and -inf (invalid).
        """
        seq_len = len(metadata)
        M = torch.full((seq_len, seq_len), float('-inf')) # Default to impossible
        
        for i, token_a in enumerate(metadata):
            for j, token_b in enumerate(metadata):
                # Evaluate Paninian mathematical functions
                if self.morph_analyzer.is_grammatically_valid_link(token_a, token_b):
                    M[i, j] = 0.0 # Unmask this pathway
        return M
    
    def _factorize_tokens(self, split_tokens: list, metadata: list) -> dict:
        """
        Convert morphological analysis to factorized tensor representation.
        
        This is the key step that enables:
        - Zero OOV: Any valid inflection can be embedded
        - 12× parameter reduction in embeddings
        
        Uses DhaatuDatabase for:
        - Root ID lookup
        - Gaṇa (verb class) extraction
        - Pada (voice type) extraction
        - Transitivity extraction
        
        Returns parallel ID tensors for each morphological dimension.
        """
        root_ids, type_ids, vibhakti_ids, vacana_ids, purusa_ids = [], [], [], [], []
        gana_ids, pada_ids, transitivity_ids = [], [], []
        
        type_map = {"subanta": 0, "tinanta": 1, "avyaya": 2, "krdanta": 3, 
                    "taddhita": 4, "samasa": 5, "none": 6}
        
        gana_map = {"bhvadi": 1, "adadi": 2, "juhotyadi": 3, "divadi": 4,
                    "svadi": 5, "tudadi": 6, "rudhadi": 7, "tanadi": 8,
                    "kryadi": 9, "curadi": 10, "none": 0}
        
        pada_map = {"parasmaipadi": 1, "atmanepadi": 2, "ubhayapadi": 3, "none": 0}
        
        transitivity_map = {"akarmaka": 1, "sakarmaka": 2, "dvikarmaka": 3, "none": 0}
        
        for meta in metadata:
            root = meta.get("root", "")
            
            # Try to get root_id from DhaatuDatabase first
            try:
                root_id = self.dhaatu_db.get_root_id(root)
                
                # Get additional dhātu features
                dhaatu_info = self.dhaatu_db.get_dhaatu(root_id)
                gana = dhaatu_info.get("gana", "none")
                pada = dhaatu_info.get("pada", "none")
                trans = dhaatu_info.get("transitivity", "none")
                
            except KeyError:
                # Fall back to morph_analyzer for non-dhātu roots (prātipadikas)
                root_id = self.morph_analyzer.get_root_id(root)
                gana, pada, trans = "none", "none", "none"
            
            root_ids.append(root_id)
            type_ids.append(type_map.get(meta.get("type", "none"), 6))
            vibhakti_ids.append(meta.get("vibhakti", 0))  # 0 = none
            vacana_ids.append(meta.get("vacana", 0))      # 0 = none
            purusa_ids.append(meta.get("purusa", 0))      # 0 = none
            
            # Dhātu-specific features
            gana_ids.append(gana_map.get(gana, 0))
            pada_ids.append(pada_map.get(pada, 0))
            transitivity_ids.append(transitivity_map.get(trans, 0))
        
        return {
            "root_ids": torch.tensor(root_ids, dtype=torch.long),
            "type_ids": torch.tensor(type_ids, dtype=torch.long),
            "vibhakti_ids": torch.tensor(vibhakti_ids, dtype=torch.long),
            "vacana_ids": torch.tensor(vacana_ids, dtype=torch.long),
            "purusa_ids": torch.tensor(purusa_ids, dtype=torch.long),
            "gana_ids": torch.tensor(gana_ids, dtype=torch.long),
            "pada_ids": torch.tensor(pada_ids, dtype=torch.long),
            "transitivity_ids": torch.tensor(transitivity_ids, dtype=torch.long),
        }

    def forward(self, raw_sanskrit_text: str):
        """
        The Forward Pass: Training and Context Processing
        
        Key difference from standard Transformers:
        - Input is factorized into 5 parallel ID tensors
        - Embedding is SUM of morphological components
        - Vocab is ~4000 primitives, not 50,000 surface forms
        """
        # --- PHASE 1: Morphological Ingestion ---
        # "Rāmo'pi" -> ["Rāmaḥ", "api"]
        split_tokens = self.segmenter.resolve_sandhi(raw_sanskrit_text)
        
        # Extract metadata: [{"root": "rāma", "vibhakti": 1, "vacana": 1}, ...]
        metadata = self.morph_analyzer.extract_tags(split_tokens)
        
        # --- FACTORIZE: Convert to parallel ID tensors ---
        # This is THE KEY INNOVATION — not a single token_id, but 5 IDs per position
        factorized = self._factorize_tokens(split_tokens, metadata)
        
        # --- PHASE 2A: Generate the Matrix M ---
        # This operates purely on the metadata, completely independent of the dense embeddings.
        matrix_M = self._build_adjacency_matrix(metadata)
        matrix_M = matrix_M.to('cuda') # Move mask to GPU
        
        # --- PHASE 2B: Neural Meaning Track (FACTORIZED) ---
        # The embedding is a SUM of morphological components, not a lookup
        # Now includes gaṇa, pada, transitivity from DhaatuDatabase
        X = self.embedding(
            factorized["root_ids"].to('cuda'),
            factorized["type_ids"].to('cuda'),
            factorized["vibhakti_ids"].to('cuda'),
            factorized["vacana_ids"].to('cuda'),
            factorized["purusa_ids"].to('cuda'),
            factorized["gana_ids"].to('cuda'),
            factorized["pada_ids"].to('cuda'),
            factorized["transitivity_ids"].to('cuda'),
        )
        Q = self.q_proj(X)
        K = self.k_proj(X)
        V = self.v_proj(X)
        
        # --- PHASE 3: Sparse Paninian Attention ---
        # The custom kernel bypasses O(N^2) math by using matrix_M as a hardware routing map.
        # It only computes (Q @ K.T) if matrix_M[i, j] is not -inf.
        attn_output = sparse_paninian_attention(Q, K, V, matrix_M)
        
        # --- PHASE 4: Semantic Maturation ---
        hidden_states = self.ffn(attn_output)
        
        # Calculate raw probabilities over the vocabulary (~4000 roots)
        raw_logits = self.lm_head(hidden_states)
        
        return raw_logits, metadata, factorized

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int):
        """
        The Inference Pass: Grammar-Constrained Decoding
        
        Key difference from standard Transformers:
        - Vocabulary is ~4000 roots, not 50,000 surface forms
        - Grammar mask ensures only valid Sanskrit forms are generated
        - Zero OOV: any inflection can be produced from ~4000 primitives
        """
        current_text = prompt
        
        for _ in range(max_new_tokens):
            # Run the forward pass to get the logits for the current sequence
            raw_logits, metadata, factorized = self.forward(current_text)
            
            # Isolate the predictions for the very next word (over ~4000 roots)
            next_token_logits = raw_logits[-1, :]
            
            # --- PHASE 5: Grammar-Constrained Decoding ---
            # Look at the morphological state of the last generated word
            last_word_state = metadata[-1]
            
            # The Symbolic Engine calculates exactly which roots are legally allowed next.
            # Returns a 1D tensor of size [vocab_size=4000] with 0.0 for legal, -inf for illegal.
            grammar_mask = self.morph_analyzer.get_valid_next_roots_mask(last_word_state)
            
            # Add the mask. Impossible roots instantly become -inf.
            constrained_logits = next_token_logits + grammar_mask
            
            # Calculate final probabilities ONLY on grammatically correct choices
            probs = F.softmax(constrained_logits, dim=-1)
            
            # Select the most semantically appropriate root from the legal subset (~4000)
            next_root_id = torch.argmax(probs, dim=-1)
            
            # Decode root_id back to surface form using morphological rules
            # The Symbolic Engine reconstructs the inflected form from the root
            next_word = self.morph_analyzer.decode_root_to_surface(next_root_id, last_word_state)
            
            current_text += " " + next_word
            
        return current_text


# ==============================================================================
# PARAMETER COUNT COMPARISON
# ==============================================================================
#
# Standard Transformer (vocab 50,000):
#   Token Embedding:     50,000 × 512 = 25,600,000 parameters
#
# Panini-LM with DhaatuDatabase (factorized, vocab ~4000):
#   Root Embedding:       3,959 × 512 =  2,027,008 parameters
#     - 2,259 dhātus (from dhaatu.json)
#     - 1,500 prātipadikas (nominal stems)
#     - 200 special tokens (upasargas, pratyayas, etc.)
#   Type Embedding:           7 × 512 =      3,584 parameters
#   Vibhakti Embedding:       9 × 512 =      4,608 parameters
#   Vacana Embedding:         4 × 512 =      2,048 parameters
#   Purusa Embedding:         4 × 512 =      2,048 parameters
#   Gaṇa Embedding:          11 × 512 =      5,632 parameters  [NEW from DhaatuDatabase]
#   Pada Embedding:           4 × 512 =      2,048 parameters  [NEW from DhaatuDatabase]
#   Transitivity Embedding:   4 × 512 =      2,048 parameters  [NEW from DhaatuDatabase]
#   -------------------------------------------
#   TOTAL EMBEDDING:                     2,049,024 parameters
#
# REDUCTION: 12.5× fewer embedding parameters
#
# Additional benefits from DhaatuDatabase:
#   - Accurate root lookup with O(1) access
#   - Semantic search by English/Hindi meanings
#   - Gaṇa-aware conjugation (10 verb classes)
#   - Pada-aware generation (parasmaipadi/atmanepadi/ubhayapadi)
#   - Transitivity constraints (akarmaka/sakarmaka/dvikarmaka)
#
# ZERO OOV: A form like "gaccheyuḥ" (they might go) can be embedded even if never seen,
# because the model composes: E(√gam) + E(tiṅanta) + E(optative) + E(bhvādi) + E(P) + E(3rd) + E(plural)
# ==============================================================================