"""
Shared Enums for the Password Guesser project.

Centralized enum definitions to avoid circular dependencies between modules.
"""

from enum import Enum


class ExpertType(Enum):
    """Types of penetration testing experts."""
    RECONNAISSANCE = "reconnaissance"
    VULNERABILITY = "vulnerability"
    EXPLOITATION = "exploitation"
    POST_EXPLOITATION = "post_exploitation"
    CREDENTIAL = "credential"
    LATERAL_MOVEMENT = "lateral_movement"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    PERSISTENCE = "persistence"
    EVASION = "evasion"