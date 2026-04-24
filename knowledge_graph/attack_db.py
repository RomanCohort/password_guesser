"""
MITRE ATT&CK Database

Provides access to the MITRE ATT&CK framework for classifying
attack techniques, tactics, and procedures.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
import json
import os
import logging

logger = logging.getLogger(__name__)


@dataclass
class ATTACKTechnique:
    """Represents a MITRE ATT&CK technique."""
    technique_id: str
    name: str
    description: str
    tactics: List[str] = field(default_factory=list)
    platforms: List[str] = field(default_factory=list)
    permissions_required: List[str] = field(default_factory=list)
    detection: str = ""
    mitigation: List[str] = field(default_factory=list)
    subtechniques: List[str] = field(default_factory=list)
    parent_technique: Optional[str] = None

    @property
    def is_subtechnique(self) -> bool:
        return "." in self.technique_id

    def to_dict(self) -> dict:
        return {
            "technique_id": self.technique_id,
            "name": self.name,
            "description": self.description,
            "tactics": self.tactics,
            "platforms": self.platforms,
            "permissions_required": self.permissions_required,
            "detection": self.detection,
            "mitigation": self.mitigation,
            "subtechniques": self.subtechniques,
            "parent_technique": self.parent_technique,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ATTACKTechnique":
        return cls(
            technique_id=data["technique_id"],
            name=data.get("name", ""),
            description=data.get("description", ""),
            tactics=data.get("tactics", []),
            platforms=data.get("platforms", []),
            permissions_required=data.get("permissions_required", []),
            detection=data.get("detection", ""),
            mitigation=data.get("mitigation", []),
            subtechniques=data.get("subtechniques", []),
            parent_technique=data.get("parent_technique"),
        )


# ATT&CK Tactics
ATTACK_TACTICS = [
    "Reconnaissance",
    "Resource Development",
    "Initial Access",
    "Execution",
    "Persistence",
    "Privilege Escalation",
    "Defense Evasion",
    "Credential Access",
    "Discovery",
    "Lateral Movement",
    "Collection",
    "Command and Control",
    "Exfiltration",
    "Impact",
]

# Pre-built mapping of common techniques for offline use
COMMON_TECHNIQUES = [
    ATTACKTechnique(
        technique_id="T1190",
        name="Exploit Public-Facing Application",
        description="Adversaries may exploit weaknesses in public-facing applications.",
        tactics=["Initial Access"],
        platforms=["Linux", "Windows", "macOS"],
        detection="Monitor application logs for exploitation attempts.",
    ),
    ATTACKTechnique(
        technique_id="T1133",
        name="External Remote Services",
        description="Adversaries may leverage external remote services.",
        tactics=["Initial Access", "Persistence"],
        platforms=["Linux", "Windows", "macOS"],
    ),
    ATTACKTechnique(
        technique_id="T1078",
        name="Valid Accounts",
        description="Adversaries may use valid accounts to gain access.",
        tactics=["Initial Access", "Persistence", "Privilege Escalation", "Defense Evasion"],
        platforms=["Linux", "Windows", "macOS"],
        permissions_required=["User", "Administrator"],
    ),
    ATTACKTechnique(
        technique_id="T1059",
        name="Command and Scripting Interpreter",
        description="Adversaries may abuse command and script interpreters.",
        tactics=["Execution"],
        platforms=["Linux", "Windows", "macOS"],
        subtechniques=["T1059.001", "T1059.003", "T1059.004", "T1059.005", "T1059.006"],
    ),
    ATTACKTechnique(
        technique_id="T1053",
        name="Scheduled Task/Job",
        description="Adversaries may abuse task scheduling functionality.",
        tactics=["Execution", "Persistence", "Privilege Escalation"],
        platforms=["Windows", "Linux", "macOS"],
    ),
    ATTACKTechnique(
        technique_id="T1068",
        name="Exploitation for Privilege Escalation",
        description="Adversaries may exploit software vulnerabilities for privilege escalation.",
        tactics=["Privilege Escalation"],
        platforms=["Linux", "Windows", "macOS"],
    ),
    ATTACKTechnique(
        technique_id="T1548",
        name="Abuse Elevation Control Mechanism",
        description="Adversaries may circumvent elevation control mechanisms.",
        tactics=["Privilege Escalation", "Defense Evasion"],
        platforms=["Linux", "Windows", "macOS"],
    ),
    ATTACKTechnique(
        technique_id="T1055",
        name="Process Injection",
        description="Adversaries may inject code into processes.",
        tactics=["Privilege Escalation", "Defense Evasion"],
        platforms=["Windows", "Linux", "macOS"],
    ),
    ATTACKTechnique(
        technique_id="T1552",
        name="Unsecured Credentials",
        description="Adversaries may search for unsecured credentials.",
        tactics=["Credential Access"],
        platforms=["Linux", "Windows", "macOS"],
    ),
    ATTACKTechnique(
        technique_id="T1110",
        name="Brute Force",
        description="Adversaries may use brute force techniques to gain access.",
        tactics=["Credential Access"],
        platforms=["Linux", "Windows", "macOS"],
    ),
    ATTACKTechnique(
        technique_id="T1046",
        name="Network Service Discovery",
        description="Adversaries may discover network services.",
        tactics=["Discovery"],
        platforms=["Linux", "Windows", "macOS"],
    ),
    ATTACKTechnique(
        technique_id="T1021",
        name="Remote Services",
        description="Adversaries may use remote services for lateral movement.",
        tactics=["Lateral Movement"],
        platforms=["Linux", "Windows", "macOS"],
        subtechniques=["T1021.001", "T1021.002", "T1021.004", "T1021.006"],
    ),
    ATTACKTechnique(
        technique_id="T1087",
        name="Account Discovery",
        description="Adversaries may discover accounts on a system.",
        tactics=["Discovery"],
        platforms=["Linux", "Windows", "macOS"],
    ),
    ATTACKTechnique(
        technique_id="T1003",
        name="OS Credential Dumping",
        description="Adversaries may dump OS credentials.",
        tactics=["Credential Access"],
        platforms=["Linux", "Windows", "macOS"],
    ),
    ATTACKTechnique(
        technique_id="T1048",
        name="Exfiltration Over Alternative Protocol",
        description="Adversaries may exfiltrate data over alternative protocols.",
        tactics=["Exfiltration"],
        platforms=["Linux", "Windows", "macOS"],
    ),
]


class ATTACKDatabase:
    """
    MITRE ATT&CK framework database.

    Loads technique data from:
    1. Pre-built common techniques (offline)
    2. Enterprise ATT&CK JSON file (optional)
    """

    def __init__(self, data_path: Optional[str] = None):
        self.techniques: Dict[str, ATTACKTechnique] = {}
        self.tactic_map: Dict[str, List[str]] = {t: [] for t in ATTACK_TACTICS}

        # Load pre-built techniques
        for tech in COMMON_TECHNIQUES:
            self.techniques[tech.technique_id] = tech
            for tactic in tech.tactics:
                self.tactic_map.setdefault(tactic, []).append(tech.technique_id)

        # Load from file if available
        if data_path and os.path.exists(data_path):
            self.load_from_file(data_path)

    def load_from_file(self, path: str) -> None:
        """Load techniques from MITRE ATT&CK STIX JSON file."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            technique_objects = {}
            for obj in data.get("objects", []):
                obj_type = obj.get("type", "")
                if obj_type == "attack-pattern":
                    ext_refs = obj.get("external_references", [])
                    tech_id = None
                    for ref in ext_refs:
                        if ref.get("source_name") == "mitre-attack":
                            tech_id = ref.get("external_id")
                            break

                    if tech_id:
                        # Extract tactics from kill chain phases
                        tactics = []
                        for phase in obj.get("kill_chain_phases", []):
                            if phase.get("kill_chain_name") == "mitre-attack":
                                tactics.append(phase.get("phase_name", ""))

                        # Extract platforms
                        platforms = obj.get("x_mitre_platforms", [])

                        # Subtechnique check
                        parent = None
                        subtechniques = []
                        if "." in tech_id:
                            parent = tech_id.split(".")[0]

                        technique = ATTACKTechnique(
                            technique_id=tech_id,
                            name=obj.get("name", ""),
                            description=obj.get("description", ""),
                            tactics=tactics,
                            platforms=platforms,
                            detection=obj.get("x_mitre_detection", ""),
                        )

                        technique_objects[tech_id] = technique

                        # Build parent-child relationships
                        if parent and parent in technique_objects:
                            technique_objects[parent].subtechniques.append(tech_id)
                            technique.parent_technique = parent

            self.techniques.update(technique_objects)

            # Rebuild tactic map
            self.tactic_map = {t: [] for t in ATTACK_TACTICS}
            for tech_id, tech in self.techniques.items():
                for tactic in tech.tactics:
                    self.tactic_map.setdefault(tactic, []).append(tech_id)

            logger.info(f"Loaded {len(self.techniques)} ATT&CK techniques")

        except Exception as e:
            logger.error(f"Failed to load ATT&CK data: {e}")

    def get_technique(self, technique_id: str) -> Optional[ATTACKTechnique]:
        """Get a technique by ID (e.g., 'T1190')."""
        return self.techniques.get(technique_id)

    def search_by_tactic(self, tactic: str) -> List[ATTACKTechnique]:
        """Get all techniques for a specific tactic."""
        tactic_lower = tactic.lower()
        # Match case-insensitively
        matched_tactic = None
        for t in ATTACK_TACTICS:
            if t.lower() == tactic_lower:
                matched_tactic = t
                break

        if not matched_tactic:
            return []

        tech_ids = self.tactic_map.get(matched_tactic, [])
        return [self.techniques[tid] for tid in tech_ids if tid in self.techniques]

    def search_by_platform(self, platform: str) -> List[ATTACKTechnique]:
        """Get all techniques applicable to a platform."""
        platform_lower = platform.lower()
        return [
            tech for tech in self.techniques.values()
            if any(p.lower() == platform_lower for p in tech.platforms)
        ]

    def get_related_techniques(self, technique_id: str) -> List[str]:
        """Get techniques related to a given technique."""
        tech = self.techniques.get(technique_id)
        if not tech:
            return []

        related = set()

        # Same tactics
        for tactic in tech.tactics:
            for tid in self.tactic_map.get(tactic, []):
                if tid != technique_id:
                    related.add(tid)

        # Subtechniques/parent
        if tech.parent_technique:
            related.add(tech.parent_technique)
        related.update(tech.subtechniques)

        return list(related)

    def get_all_techniques(self) -> List[ATTACKTechnique]:
        """Get all loaded techniques."""
        return list(self.techniques.values())

    def get_tactics_summary(self) -> Dict[str, int]:
        """Get count of techniques per tactic."""
        return {tactic: len(techs) for tactic, techs in self.tactic_map.items()}
