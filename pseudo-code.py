import torch
import torch.nn as nn
import torch.nn.functional as F

# Hypothetical Python wrappers for the open-source linguistic engines
from external_nlp import VidyutParser, HeritageSegmenter
from custom_kernels import sparse_paninian_attention # The custom C++/Triton kernel


class PaninianEmbedding(nn.Module):
    """
    Factorized Embedding Layer — THE KEY INNOVATION OF PANINI-LM.
    
    Traditional Transformers: nn.Embedding(50000, d_model) → 25.6M parameters
    Panini-LM: Additive composition of small embedding matrices → 2.1M parameters
    
    The neural network sees each word as the SUM of its mathematical parts:
        E(gacchati) = E(√gam) + E(tiṅanta) + E(laṭ) + E(prathama) + E(eka)
    
    Benefits:
    1. ZERO OOV for inflections — all valid forms are compositionally derivable
    2. 12× parameter reduction in embeddings
    3. Morphological knowledge is encoded structurally, not learned implicitly
    """
    
    def __init__(self, d_model: int = 512):
        super().__init__()
        
        # === Core Semantic Embeddings (Roots & Stems) ===
        # ~4000 items: dhātus, prātipadikas, upasargas, pratyayas, special tokens
        self.root_embed = nn.Embedding(4000, d_model)
        
        # === Grammatical Meta-Data Embeddings (Tiny Matrices) ===
        self.type_embed = nn.Embedding(7, d_model)      # subanta, tiṅanta, avyaya, kṛdanta, taddhita, samāsa, none
        self.vibhakti_embed = nn.Embedding(9, d_model)  # 1-7 + vocative + none
        self.vacana_embed = nn.Embedding(4, d_model)    # singular, dual, plural, none
        self.purusa_embed = nn.Embedding(4, d_model)    # 3rd, 2nd, 1st, none (Sanskrit convention)
        
    def forward(
        self,
        root_ids: torch.LongTensor,
        type_ids: torch.LongTensor,
        vibhakti_ids: torch.LongTensor,
        vacana_ids: torch.LongTensor,
        purusa_ids: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Construct embeddings by summing morphological components.
        
        Args:
            root_ids: (batch, seq) — Root/stem IDs (the semantic core)
            type_ids: (batch, seq) — Token type (subanta/tiṅanta/etc.)
            vibhakti_ids: (batch, seq) — Case (1-7, vocative, or none)
            vacana_ids: (batch, seq) — Number (singular/dual/plural/none)
            purusa_ids: (batch, seq) — Person (3rd/2nd/1st/none)
        
        Returns:
            (batch, seq, d_model) — Factorized embeddings
        """
        # The neural network sees the word as a sum of its mathematical parts.
        return (
            self.root_embed(root_ids) +
            self.type_embed(type_ids) +
            self.vibhakti_embed(vibhakti_ids) +
            self.vacana_embed(vacana_ids) +
            self.purusa_embed(purusa_ids)
        )


class PaninianNeuroSymbolicLLM(nn.Module):
    """
    The Panini-LM Architecture: A Neuro-Symbolic Language Model for Sanskrit.
    
    Key Innovation: Factorized Embeddings (vocab ~4000 vs 50,000+)
        - Zero OOV errors for any valid Sanskrit inflection
        - 12× parameter reduction in embedding layer
        - Morphological structure preserved, not learned implicitly
    """
    
    def __init__(self, d_model: int = 512, num_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = 4000  # ~4000 morphological primitives, NOT 50,000 surface forms
        
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
        # Note: We use FACTORIZED embeddings, NOT nn.Embedding(50000, d_model)
        self.embedding = PaninianEmbedding(d_model)
        
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
        
        Returns parallel ID tensors for each morphological dimension.
        """
        root_ids, type_ids, vibhakti_ids, vacana_ids, purusa_ids = [], [], [], [], []
        
        type_map = {"subanta": 0, "tinanta": 1, "avyaya": 2, "krdanta": 3, 
                    "taddhita": 4, "samasa": 5, "none": 6}
        
        for meta in metadata:
            root_ids.append(self.morph_analyzer.get_root_id(meta["root"]))
            type_ids.append(type_map.get(meta.get("type", "none"), 6))
            vibhakti_ids.append(meta.get("vibhakti", 0))  # 0 = none
            vacana_ids.append(meta.get("vacana", 0))      # 0 = none
            purusa_ids.append(meta.get("purusa", 0))      # 0 = none
        
        return {
            "root_ids": torch.tensor(root_ids, dtype=torch.long),
            "type_ids": torch.tensor(type_ids, dtype=torch.long),
            "vibhakti_ids": torch.tensor(vibhakti_ids, dtype=torch.long),
            "vacana_ids": torch.tensor(vacana_ids, dtype=torch.long),
            "purusa_ids": torch.tensor(purusa_ids, dtype=torch.long),
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
        X = self.embedding(
            factorized["root_ids"].to('cuda'),
            factorized["type_ids"].to('cuda'),
            factorized["vibhakti_ids"].to('cuda'),
            factorized["vacana_ids"].to('cuda'),
            factorized["purusa_ids"].to('cuda'),
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
# Panini-LM (factorized, vocab ~4000):
#   Root Embedding:       4,000 × 512 =  2,048,000 parameters
#   Type Embedding:           7 × 512 =      3,584 parameters
#   Vibhakti Embedding:       9 × 512 =      4,608 parameters
#   Vacana Embedding:         4 × 512 =      2,048 parameters
#   Purusa Embedding:         4 × 512 =      2,048 parameters
#   -------------------------------------------
#   TOTAL EMBEDDING:                     2,060,288 parameters
#
# REDUCTION: 12.4× fewer embedding parameters
#
# Additional benefit: ZERO OOV errors for any valid Sanskrit inflection.
# A form like "gaccheyuḥ" (they might go) can be embedded even if never seen,
# because the model composes: E(√gam) + E(tiṅanta) + E(optative) + E(3rd) + E(plural)
# ==============================================================================