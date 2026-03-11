"""
Adjacency matrix builder for Phase 2A.

Builds the sparse grammatical adjacency matrix M from morphological tokens.
M[i,j] = 0.0 means token i can attend to token j.
M[i,j] = -inf means token i cannot attend to token j.
"""

from typing import List, Optional
import logging

import torch

from panini_lm.core.types import (
    MorphToken,
    AdjacencyMatrix,
    AdjacencyMeta,
    GrammaticalLink,
)
from panini_lm.core.config import SymbolicConfig
from panini_lm.phase2a_symbolic.rules import GrammarRule, get_default_rules


logger = logging.getLogger(__name__)


def compute_adjacency_meta(matrix: torch.Tensor) -> AdjacencyMeta:
    """
    Compute metadata about the adjacency matrix.
    
    Args:
        matrix: Adjacency matrix (N x N), values 0.0 or -inf
        
    Returns:
        AdjacencyMeta with statistics
    """
    N = matrix.shape[0]
    total_edges = N * N
    
    # Count valid edges (where value is 0.0, not -inf)
    valid_mask = matrix == 0.0
    num_valid = valid_mask.sum().item()
    
    # Sparsity ratio
    sparsity = num_valid / total_edges if total_edges > 0 else 0.0
    
    # Average connections per token
    avg_k = num_valid / N if N > 0 else 0.0
    
    return {
        "seq_len": N,
        "num_valid_edges": int(num_valid),
        "sparsity_ratio": sparsity,
        "avg_connections_per_token": avg_k,
    }


def build_adjacency_matrix(
    tokens: List[MorphToken],
    rules: Optional[List[GrammarRule]] = None,
    config: Optional[SymbolicConfig] = None,
    device: torch.device = torch.device("cpu"),
) -> AdjacencyMatrix:
    """
    Build the grammatical adjacency matrix M from morphological tokens.
    
    The matrix is built by applying grammar rules to each token pair.
    If any rule allows the connection, M[i,j] = 0.0.
    Otherwise, M[i,j] = -inf.
    
    Args:
        tokens: List of MorphToken from Phase 1
        rules: Grammar rules to apply (default: get_default_rules())
        config: SymbolicConfig for customization
        device: Device to create tensor on
        
    Returns:
        AdjacencyMatrix with:
        - matrix: (N, N) tensor
        - meta: Statistics
        - links: List of valid grammatical links
        
    Example:
        >>> tokens = [
        ...     {"surface": "rāmaḥ", "type": "subanta", "attributes": {"vibhakti": 1}},
        ...     {"surface": "gacchati", "type": "tinanta", "attributes": {}},
        ... ]
        >>> adj = build_adjacency_matrix(tokens)
        >>> adj.matrix[0, 1]  # Subject can attend to verb
        tensor(0.)
    """
    if rules is None:
        rules = get_default_rules()
    
    if config is None:
        config = SymbolicConfig()
    
    N = len(tokens)
    
    # Handle empty input
    if N == 0:
        empty_matrix = torch.zeros(0, 0, device=device)
        return AdjacencyMatrix(
            matrix=empty_matrix,
            meta=compute_adjacency_meta(empty_matrix),
            links=[],
        )
    
    # Initialize with all blocked (-inf)
    matrix = torch.full((N, N), float('-inf'), dtype=torch.float32, device=device)
    links: List[GrammaticalLink] = []
    
    # Apply rules to each token pair
    for i, token_i in enumerate(tokens):
        for j, token_j in enumerate(tokens):
            # Check each rule in priority order
            for rule in rules:
                if rule.check(token_i, token_j, i, j):
                    matrix[i, j] = 0.0
                    links.append({
                        "source_idx": i,
                        "target_idx": j,
                        "link_type": rule.get_link_type(token_i, token_j),
                        "rule_applied": rule.name,
                    })
                    break  # First matching rule wins
    
    meta = compute_adjacency_meta(matrix)
    
    logger.debug(
        f"Built adjacency matrix: {N}x{N}, "
        f"sparsity={meta['sparsity_ratio']:.2%}, "
        f"avg_k={meta['avg_connections_per_token']:.1f}"
    )
    
    return AdjacencyMatrix(matrix=matrix, meta=meta, links=links)


def visualize_matrix(adj: AdjacencyMatrix, tokens: List[MorphToken]) -> str:
    """
    Create ASCII visualization of the adjacency matrix.
    
    Useful for debugging and understanding matrix structure.
    
    Args:
        adj: AdjacencyMatrix to visualize
        tokens: Original tokens (for labels)
        
    Returns:
        ASCII string representation
    """
    N = adj.matrix.shape[0]
    if N == 0:
        return "(empty matrix)"
    
    # Get short labels for tokens
    labels = [t["surface"][:6] for t in tokens]
    
    lines = []
    
    # Header row
    header = "        " + " ".join(f"{l:>6}" for l in labels)
    lines.append(header)
    lines.append("-" * len(header))
    
    # Data rows
    for i, label in enumerate(labels):
        row_values = []
        for j in range(N):
            val = adj.matrix[i, j].item()
            if val == float('-inf'):
                row_values.append("  ----")
            else:
                row_values.append("     ✓")
        
        line = f"{label:>6}: " + " ".join(row_values)
        lines.append(line)
    
    return "\n".join(lines)
