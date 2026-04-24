"""
Penetration Test Action Space

Defines the actions available to the RL agent and
manages the action space with validity checking.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Dict, Set
import numpy as np
import hashlib
import json


class ActionType(Enum):
    SCAN_NETWORK = "scan_network"
    SCAN_PORT = "scan_port"
    ENUMERATE_SERVICE = "enum_service"
    EXPLOIT_VULN = "exploit_vuln"
    BRUTE_FORCE = "brute_force"
    LATERAL_MOVE = "lateral_move"
    PRIV_ESCALATE = "priv_escalate"
    DUMP_CREDS = "dump_creds"
    EXFILTRATE = "exfiltrate"


# Maximum number of each action type for indexing
ACTION_DIM = len(ActionType)
MAX_TARGETS = 100  # Maximum target slots


@dataclass
class PenTestAction:
    """Represents a penetration test action."""
    type: ActionType
    target: str
    parameters: Dict = None
    cve_id: Optional[str] = None
    technique_id: Optional[str] = None

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "target": self.target,
            "parameters": self.parameters,
            "cve_id": self.cve_id,
            "technique_id": self.technique_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PenTestAction":
        return cls(
            type=ActionType(data["type"]),
            target=data["target"],
            parameters=data.get("parameters", {}),
            cve_id=data.get("cve_id"),
            technique_id=data.get("technique_id"),
        )

    @property
    def action_id(self) -> str:
        """Unique identifier for this action."""
        key = f"{self.type.value}:{self.target}:{json.dumps(self.parameters, sort_keys=True)}"
        return hashlib.md5(key.encode()).hexdigest()[:12]

    def __str__(self) -> str:
        desc = f"{self.type.value} -> {self.target}"
        if self.cve_id:
            desc += f" (CVE: {self.cve_id})"
        return desc


class ActionSpace:
    """
    Manages the action space for the RL agent.

    Maps between PenTestAction objects and integer indices
    used by the policy network.
    """

    def __init__(self):
        # Known targets and their indices
        self._target_to_idx: Dict[str, int] = {}
        self._idx_to_target: Dict[int, str] = {}
        self._next_target_idx = 0

        # Cached valid actions per state hash
        self._valid_actions_cache: Dict[int, List[PenTestAction]] = {}

    def register_target(self, target: str) -> int:
        """Register a target and get its index."""
        if target in self._target_to_idx:
            return self._target_to_idx[target]

        idx = self._next_target_idx
        self._target_to_idx[target] = idx
        self._idx_to_target[idx] = target
        self._next_target_idx += 1
        return idx

    def get_target_index(self, target: str) -> int:
        """Get index for a target, registering if needed."""
        return self.register_target(target)

    @property
    def total_action_dim(self) -> int:
        """Total action dimension (action types * targets)."""
        return ACTION_DIM * MAX_TARGETS

    def action_to_index(self, action: PenTestAction) -> int:
        """Convert action to a flat index."""
        type_idx = list(ActionType).index(action.type)
        target_idx = self.get_target_index(action.target) % MAX_TARGETS
        return type_idx * MAX_TARGETS + target_idx

    def index_to_action(self, index: int) -> PenTestAction:
        """Convert flat index back to action."""
        type_idx = index // MAX_TARGETS
        target_idx = index % MAX_TARGETS
        action_type = list(ActionType)[type_idx]
        target = self._idx_to_target.get(target_idx, f"target_{target_idx}")
        return PenTestAction(type=action_type, target=target)

    def get_valid_actions(self, state) -> List[PenTestAction]:
        """
        Get all valid actions for the current state.

        Uses rules based on current knowledge:
        - Can always scan discovered networks
        - Can scan ports on discovered hosts
        - Can exploit known vulnerabilities
        - Can brute force services with credentials
        - Can move laterally from compromised hosts
        - Can escalate privileges on compromised hosts
        - Can dump credentials with sufficient access
        - Can exfiltrate from hosts with data access
        """
        from .state import PenTestState

        if not isinstance(state, PenTestState):
            return []

        state_hash = hash((
            frozenset(state.discovered_hosts),
            frozenset(state.compromised_hosts),
            tuple(sorted(state.vulnerabilities.items())),
        ))

        if state_hash in self._valid_actions_cache:
            return self._valid_actions_cache[state_hash]

        actions: List[PenTestAction] = []

        # Always allow network scanning
        actions.append(PenTestAction(
            type=ActionType.SCAN_NETWORK,
            target="network",
            parameters={"scan_type": "ping"},
            technique_id="T1046",
        ))

        # Port scan discovered but uncompromised hosts
        for host in sorted(state.discovered_hosts - state.compromised_hosts):
            actions.append(PenTestAction(
                type=ActionType.SCAN_PORT,
                target=host,
                parameters={"scan_type": "tcp"},
                technique_id="T1046",
            ))

        # Enumerate services on hosts with open ports
        for host, ports in state.open_ports.items():
            if host not in state.compromised_hosts:
                for port in ports[:10]:  # Limit to first 10 ports
                    actions.append(PenTestAction(
                        type=ActionType.ENUMERATE_SERVICE,
                        target=f"{host}:{port}",
                        parameters={"port": port},
                    ))

        # Exploit known vulnerabilities
        for host, vulns in state.vulnerabilities.items():
            for cve_id in vulns:
                self.register_target(host)
                actions.append(PenTestAction(
                    type=ActionType.EXPLOIT_VULN,
                    target=host,
                    parameters={"cve_id": cve_id},
                    cve_id=cve_id,
                ))

        # Brute force services on discovered hosts
        for host, ports in state.open_ports.items():
            if host not in state.compromised_hosts:
                for port in ports[:5]:
                    if port in (22, 80, 443, 445, 3389, 3306, 5432):
                        svc = {22: "ssh", 445: "smb", 3389: "rdp",
                               3306: "mysql", 5432: "postgresql"}.get(port, "unknown")
                        actions.append(PenTestAction(
                            type=ActionType.BRUTE_FORCE,
                            target=f"{host}:{port}",
                            parameters={"port": port, "service": svc},
                            technique_id="T1110",
                        ))

        # Lateral movement from compromised hosts
        for compromised in state.compromised_hosts:
            for target_host in sorted(state.discovered_hosts - state.compromised_hosts):
                actions.append(PenTestAction(
                    type=ActionType.LATERAL_MOVE,
                    target=target_host,
                    parameters={"source": compromised},
                    technique_id="T1021",
                ))

        # Privilege escalation on compromised hosts without root
        for host in state.compromised_hosts:
            priv = state.privileges.get(host, "user")
            if priv not in ("root", "system", "admin"):
                actions.append(PenTestAction(
                    type=ActionType.PRIV_ESCALATE,
                    target=host,
                    technique_id="T1068",
                ))

        # Dump credentials on compromised hosts with sufficient access
        for host in state.compromised_hosts:
            priv = state.privileges.get(host, "user")
            if priv in ("user", "admin", "root", "system"):
                actions.append(PenTestAction(
                    type=ActionType.DUMP_CREDS,
                    target=host,
                    technique_id="T1003",
                ))

        # Exfiltrate data from compromised hosts
        for host in state.compromised_hosts:
            actions.append(PenTestAction(
                type=ActionType.EXFILTRATE,
                target=host,
                parameters={"method": "encrypted_channel"},
                technique_id="T1048",
            ))

        self._valid_actions_cache[state_hash] = actions
        return actions

    def get_action_mask(self, state) -> np.ndarray:
        """
        Get a binary mask for valid actions.

        Shape: (total_action_dim,) where 1 = valid, 0 = invalid.
        """
        valid_actions = self.get_valid_actions(state)
        mask = np.zeros(self.total_action_dim, dtype=np.float32)
        for action in valid_actions:
            idx = self.action_to_index(action)
            if 0 <= idx < self.total_action_dim:
                mask[idx] = 1.0
        return mask

    def clear_cache(self) -> None:
        """Clear the valid actions cache."""
        self._valid_actions_cache.clear()
