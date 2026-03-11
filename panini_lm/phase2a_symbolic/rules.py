"""
Grammar rules for determining valid token relationships.

Each rule implements the GrammarRule interface and determines whether
a pair of tokens can have a grammatical relationship (attention allowed).

Rules are applied in priority order; first matching rule wins.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from panini_lm.core.types import MorphToken


class GrammarRule(ABC):
    """
    Abstract base class for grammar rules.
    
    Each rule checks whether a specific grammatical relationship
    exists between two tokens.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this rule (e.g., 'kartā-kriyā')."""
        pass
    
    @property
    @abstractmethod
    def priority(self) -> int:
        """
        Priority of this rule (lower = higher priority).
        
        When multiple rules could apply, higher priority rules
        are checked first.
        """
        pass
    
    @abstractmethod
    def check(self, head: MorphToken, dep: MorphToken, head_idx: int, dep_idx: int) -> bool:
        """
        Check if head can attend to dep according to this rule.
        
        Args:
            head: The token doing the attending (query position)
            dep: The token being attended to (key position)
            head_idx: Index of head in sequence
            dep_idx: Index of dep in sequence
            
        Returns:
            True if attention is allowed, False otherwise
        """
        pass
    
    def get_link_type(self, head: MorphToken, dep: MorphToken) -> str:
        """Get the type of link this rule represents."""
        return self.name


class SelfAttentionRule(GrammarRule):
    """
    Self-attention rule: every token can attend to itself.
    
    This is always allowed and has highest priority.
    """
    
    @property
    def name(self) -> str:
        return "sva-sambandha"  # Self-relation
    
    @property
    def priority(self) -> int:
        return 0  # Highest priority
    
    def check(self, head: MorphToken, dep: MorphToken, head_idx: int, dep_idx: int) -> bool:
        return head_idx == dep_idx


class SubjectVerbRule(GrammarRule):
    """
    Kartā-kriyā (subject-verb) agreement rule.
    
    A nominative noun (vibhakti=1) can link to a verb if they agree in number.
    """
    
    @property
    def name(self) -> str:
        return "kartā-kriyā"
    
    @property
    def priority(self) -> int:
        return 10
    
    def check(self, head: MorphToken, dep: MorphToken, head_idx: int, dep_idx: int) -> bool:
        # Head must be a noun in nominative case
        if head["type"] != "subanta":
            return False
        if head["attributes"].get("vibhakti") != 1:
            return False
        
        # Dep must be a verb
        if dep["type"] != "tinanta":
            return False
        
        # Check number agreement
        head_vacana = head["attributes"].get("vacana")
        dep_vacana = dep["attributes"].get("vacana")
        
        if head_vacana is not None and dep_vacana is not None:
            return head_vacana == dep_vacana
        
        # If number not specified, allow the link
        return True
    
    def get_link_type(self, head: MorphToken, dep: MorphToken) -> str:
        return "subject-verb"


class ObjectVerbRule(GrammarRule):
    """
    Karma-kriyā (object-verb) rule.
    
    An accusative noun (vibhakti=2) can link to a verb.
    """
    
    @property
    def name(self) -> str:
        return "karma-kriyā"
    
    @property
    def priority(self) -> int:
        return 20
    
    def check(self, head: MorphToken, dep: MorphToken, head_idx: int, dep_idx: int) -> bool:
        # Head must be a noun in accusative case
        if head["type"] != "subanta":
            return False
        if head["attributes"].get("vibhakti") != 2:
            return False
        
        # Dep must be a verb
        return dep["type"] == "tinanta"
    
    def get_link_type(self, head: MorphToken, dep: MorphToken) -> str:
        return "object-verb"


class InstrumentVerbRule(GrammarRule):
    """
    Karaṇa-kriyā (instrument-verb) rule.
    
    An instrumental noun (vibhakti=3) can link to a verb.
    """
    
    @property
    def name(self) -> str:
        return "karaṇa-kriyā"
    
    @property
    def priority(self) -> int:
        return 30
    
    def check(self, head: MorphToken, dep: MorphToken, head_idx: int, dep_idx: int) -> bool:
        # Head must be a noun in instrumental case
        if head["type"] != "subanta":
            return False
        if head["attributes"].get("vibhakti") != 3:
            return False
        
        return dep["type"] == "tinanta"
    
    def get_link_type(self, head: MorphToken, dep: MorphToken) -> str:
        return "instrument-verb"


class DativeVerbRule(GrammarRule):
    """
    Sampradāna-kriyā (recipient-verb) rule.
    
    A dative noun (vibhakti=4) can link to a verb.
    """
    
    @property
    def name(self) -> str:
        return "sampradāna-kriyā"
    
    @property
    def priority(self) -> int:
        return 40
    
    def check(self, head: MorphToken, dep: MorphToken, head_idx: int, dep_idx: int) -> bool:
        if head["type"] != "subanta":
            return False
        if head["attributes"].get("vibhakti") != 4:
            return False
        
        return dep["type"] == "tinanta"
    
    def get_link_type(self, head: MorphToken, dep: MorphToken) -> str:
        return "dative-verb"


class VerbSubjectRule(GrammarRule):
    """
    Reverse of SubjectVerbRule: verb can attend to its subject.
    
    This allows bidirectional information flow in the dependency.
    """
    
    @property
    def name(self) -> str:
        return "kriyā-kartā"
    
    @property
    def priority(self) -> int:
        return 15
    
    def check(self, head: MorphToken, dep: MorphToken, head_idx: int, dep_idx: int) -> bool:
        # Head must be a verb
        if head["type"] != "tinanta":
            return False
        
        # Dep must be nominative noun
        if dep["type"] != "subanta":
            return False
        if dep["attributes"].get("vibhakti") != 1:
            return False
        
        # Check number agreement
        head_vacana = head["attributes"].get("vacana")
        dep_vacana = dep["attributes"].get("vacana")
        
        if head_vacana is not None and dep_vacana is not None:
            return head_vacana == dep_vacana
        
        return True


class VerbObjectRule(GrammarRule):
    """
    Reverse of ObjectVerbRule: verb can attend to its object.
    """
    
    @property
    def name(self) -> str:
        return "kriyā-karma"
    
    @property
    def priority(self) -> int:
        return 25
    
    def check(self, head: MorphToken, dep: MorphToken, head_idx: int, dep_idx: int) -> bool:
        # Head must be a verb
        if head["type"] != "tinanta":
            return False
        
        # Dep must be accusative noun
        if dep["type"] != "subanta":
            return False
        
        return dep["attributes"].get("vibhakti") == 2


class ParticleRule(GrammarRule):
    """
    Particles (avyaya) can attend to adjacent tokens.
    
    Particles like 'ca', 'api', 'eva' modify nearby words.
    """
    
    @property
    def name(self) -> str:
        return "avyaya-sambandha"
    
    @property
    def priority(self) -> int:
        return 50
    
    def check(self, head: MorphToken, dep: MorphToken, head_idx: int, dep_idx: int) -> bool:
        # If head is a particle, allow attention to adjacent tokens
        if head["type"] == "avyaya":
            return abs(head_idx - dep_idx) <= 1
        
        # If dep is a particle, allow attention from adjacent tokens
        if dep["type"] == "avyaya":
            return abs(head_idx - dep_idx) <= 1
        
        return False


class AdjacentRule(GrammarRule):
    """
    Fallback: allow attention between adjacent tokens.
    
    This is a weak rule that allows local context when
    no stronger grammatical rule applies.
    """
    
    @property
    def name(self) -> str:
        return "sannihita"  # Adjacent
    
    @property
    def priority(self) -> int:
        return 100  # Low priority (fallback)
    
    def check(self, head: MorphToken, dep: MorphToken, head_idx: int, dep_idx: int) -> bool:
        return abs(head_idx - dep_idx) <= 1


def get_default_rules() -> List[GrammarRule]:
    """
    Get the default set of grammar rules.
    
    Returns rules sorted by priority (highest priority first).
    """
    rules = [
        SelfAttentionRule(),
        SubjectVerbRule(),
        VerbSubjectRule(),
        ObjectVerbRule(),
        VerbObjectRule(),
        InstrumentVerbRule(),
        DativeVerbRule(),
        ParticleRule(),
        AdjacentRule(),
    ]
    
    # Sort by priority
    return sorted(rules, key=lambda r: r.priority)
