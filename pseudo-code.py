import torch
import torch.nn as nn
import torch.nn.functional as F

# Hypothetical Python wrappers for the open-source linguistic engines
from external_nlp import VidyutParser, HeritageSegmenter
from custom_kernels import sparse_paninian_attention # The custom C++/Triton kernel

class PaninianNeuroSymbolicLLM(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads):
        super().__init__()
        
        # ==========================================
        # TRACK A: The Symbolic Engine (Syntax)
        # ==========================================
        self.segmenter = HeritageSegmenter() # Handles Sandhi & Samasa (Phase 1)
        self.morph_analyzer = VidyutParser() # Extracts Purusa, Vacana, etc.
        
        # ==========================================
        # TRACK B: The Neural Engine (Semantics)
        # ==========================================
        # Phase 2B: Position-Agnostic Embeddings 
        # Note: We explicitly DO NOT instantiate RoPE or absolute positional encodings here.
        self.embedding = nn.Embedding(vocab_size, d_model)
        
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
        
        # Phase 5: Vocabulary Projection
        self.lm_head = nn.Linear(d_model, vocab_size)

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

    def forward(self, raw_sanskrit_text: str):
        """
        The Forward Pass: Training and Context Processing
        """
        # --- PHASE 1: Morphological Ingestion ---
        # "Rāmo'pi" -> ["Rāmaḥ", "api"]
        split_tokens = self.segmenter.resolve_sandhi(raw_sanskrit_text)
        
        # Extract metadata: [{"root": "rāma", "vibhakti": 1, "vacana": 1}, ...]
        metadata = self.morph_analyzer.extract_tags(split_tokens)
        token_ids = self.morph_analyzer.convert_to_ids(split_tokens)
        
        # --- PHASE 2A: Generate the Matrix M ---
        # This operates purely on the metadata, completely independent of the dense embeddings.
        matrix_M = self._build_adjacency_matrix(metadata)
        matrix_M = matrix_M.to('cuda') # Move mask to GPU
        
        # --- PHASE 2B: Neural Meaning Track ---
        X = self.embedding(token_ids)
        Q = self.q_proj(X)
        K = self.k_proj(X)
        V = self.v_proj(X)
        
        # --- PHASE 3: Sparse Paninian Attention ---
        # The custom kernel bypasses O(N^2) math by using matrix_M as a hardware routing map.
        # It only computes (Q @ K.T) if matrix_M[i, j] is not -inf.
        attn_output = sparse_paninian_attention(Q, K, V, matrix_M)
        
        # --- PHASE 4: Semantic Maturation ---
        hidden_states = self.ffn(attn_output)
        
        # Calculate raw probabilities over the vocabulary
        raw_logits = self.lm_head(hidden_states)
        
        return raw_logits, metadata

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int):
        """
        The Inference Pass: Grammar-Constrained Decoding
        """
        current_text = prompt
        
        for _ in range(max_new_tokens):
            # Run the forward pass to get the logits for the current sequence
            raw_logits, metadata = self.forward(current_text)
            
            # Isolate the predictions for the very next word
            next_token_logits = raw_logits[-1, :]
            
            # --- PHASE 5: Grammar-Constrained Decoding ---
            # Look at the morphological state of the last generated word
            last_word_state = metadata[-1]
            
            # The Symbolic Engine calculates exactly which affixes/roots are legally allowed next.
            # Returns a 1D tensor of size [vocab_size] with 0.0 for legal, -inf for illegal.
            
            grammar_mask = self.morph_analyzer.get_valid_next_tokens_mask(last_word_state)
            
            # Add the mask. Impossible tokens instantly become -inf.
            constrained_logits = next_token_logits + grammar_mask
            
            # Calculate final probabilities ONLY on grammatically correct choices
            probs = F.softmax(constrained_logits, dim=-1)
            
            # Select the most semantically appropriate token from the legal subset
            next_token_id = torch.argmax(probs, dim=-1)
            next_word = self.morph_analyzer.decode_id(next_token_id)
            
            current_text += " " + next_word
            
        return current_text