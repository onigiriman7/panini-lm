"""
Grammar State Tracker.

Tracks the current morphological state during generation,
including agreement requirements and structural expectations.

The state is updated after each generated token and used to
determine which tokens are valid next.
"""

from __future__ import annotations

from typing import Optional, List, Set
from dataclasses import dataclass, field
from copy import deepcopy

from panini_lm.core.types import MorphToken, MorphAttributes


@dataclass
class AgreementRequirement:
    """
    A pending agreement requirement.
    
    For example, if a singular subject was generated, there should be
    a requirement for a singular verb somewhere in the sentence.
    """
    
    category: str
    """Category of token needed (e.g., "verb", "noun")."""
    
    attributes: MorphAttributes
    """Required attribute values."""
    
    source_idx: int
    """Index of token that created this requirement."""
    
    satisfied: bool = False
    """Whether this requirement has been satisfied."""


@dataclass
class GrammarState:
    """
    Tracks grammar constraints during generation.
    
    State includes:
    - Generated tokens so far
    - Pending agreement requirements
    - Current position type expectations
    - Structural requirements (e.g., need verb)
    
    The state is immutable-ish: use `update()` to create new state.
    """
    
    tokens: List[MorphToken] = field(default_factory=list)
    """Tokens generated so far."""
    
    pending_agreements: List[AgreementRequirement] = field(default_factory=list)
    """Agreement requirements not yet satisfied."""
    
    has_subject: bool = False
    """Whether a subject (nominative) has been seen."""
    
    has_verb: bool = False
    """Whether a finite verb has been seen."""
    
    expected_vacana: Optional[int] = None
    """Expected number (singular=1, dual=2, plural=3) from subject."""
    
    open_compounds: int = 0
    """Number of open compound markers."""
    
    @classmethod
    def initial(cls) -> "GrammarState":
        """Create initial empty state."""
        return cls()
    
    def copy(self) -> "GrammarState":
        """Create a deep copy of the state."""
        return deepcopy(self)
    
    def update(self, token: MorphToken) -> "GrammarState":
        """
        Update state with a new generated token.
        
        Args:
            token: The newly generated token
        
        Returns:
            New GrammarState with updated constraints
        """
        new_state = self.copy()
        new_state.tokens = self.tokens + [token]
        
        token_type = token.get("type", "unknown")
        attrs = token.get("attributes", {})
        
        # Handle subject (nominative noun)
        if token_type == "subanta":
            vibhakti = attrs.get("vibhakti", 0)
            if vibhakti == 1:  # Nominative
                new_state.has_subject = True
                new_state.expected_vacana = attrs.get("vacana")
                
                # Add verb agreement requirement
                new_state.pending_agreements.append(
                    AgreementRequirement(
                        category="verb",
                        attributes={"vacana": attrs.get("vacana")},
                        source_idx=len(new_state.tokens) - 1,
                    )
                )
        
        # Handle verb
        if token_type == "tinanta":
            new_state.has_verb = True
            verb_vacana = attrs.get("vacana")
            
            # Check and satisfy pending agreements
            for req in new_state.pending_agreements:
                if (req.category == "verb" 
                    and not req.satisfied
                    and req.attributes.get("vacana") == verb_vacana):
                    req.satisfied = True
        
        return new_state
    
    def get_unsatisfied_requirements(self) -> List[AgreementRequirement]:
        """Get list of unsatisfied agreement requirements."""
        return [r for r in self.pending_agreements if not r.satisfied]
    
    def is_complete_sentence(self) -> bool:
        """
        Check if current state represents a complete sentence.
        
        A complete sentence needs:
        - At least one verb (for finite clauses)
        - All agreement requirements satisfied
        """
        if not self.has_verb:
            return False
        
        unsatisfied = self.get_unsatisfied_requirements()
        if unsatisfied:
            return False
        
        return True
    
    def requires_verb(self) -> bool:
        """Check if a verb is required but not yet seen."""
        return self.has_subject and not self.has_verb
    
    def get_required_vacana(self) -> Optional[int]:
        """Get the required number for verb agreement."""
        if self.expected_vacana is not None and not self.has_verb:
            return self.expected_vacana
        return None
    
    def to_dict(self) -> dict:
        """Convert state to dictionary for debugging."""
        return {
            "num_tokens": len(self.tokens),
            "has_subject": self.has_subject,
            "has_verb": self.has_verb,
            "expected_vacana": self.expected_vacana,
            "pending_agreements": len(self.pending_agreements),
            "unsatisfied": len(self.get_unsatisfied_requirements()),
        }
