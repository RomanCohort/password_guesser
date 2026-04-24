"""
Penetration Test State Representation

Defines the state space for the RL agent, representing
the current knowledge and access level of the pen test.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional
import numpy as np
import json


@dataclass
class PenTestState:
    """
    Represents the current state of a penetration test.

    Tracks discovered hosts, open ports, vulnerabilities,
    credentials, privileges, and compromised hosts.
    """
    discovered_hosts: Set[str] = field(default_factory=set)
    open_ports: Dict[str, List[int]] = field(default_factory=dict)
    vulnerabilities: Dict[str, List[str]] = field(default_factory=dict)
    services: Dict[str, Dict[int, str]] = field(default_factory=dict)
    credentials: Dict[str, str] = field(default_factory=dict)
    privileges: Dict[str, str] = field(default_factory=dict)
    compromised_hosts: Set[str] = field(default_factory=set)
    discovered_data: Set[str] = field(default_factory=set)

    def to_vector(self, dim: int = 256) -> np.ndarray:
        """
        Encode state as a fixed-size vector for neural network input.

        Encoding scheme:
        - [0:32] - discovered host density (one-hot-ish encoding)
        - [32:64] - vulnerability density per host
        - [64:96] - credential count / privilege levels
        - [96:128] - compromise ratio
        - [128:160] - service diversity
        - [160:256] - padding / reserved
        """
        vec = np.zeros(dim, dtype=np.float32)

        # Host discovery ratio
        n_hosts = len(self.discovered_hosts)
        vec[0] = min(1.0, n_hosts / 254.0)

        # Average open ports per host
        if n_hosts > 0:
            avg_ports = sum(len(ports) for ports in self.open_ports.values()) / n_hosts
            vec[1] = min(1.0, avg_ports / 100.0)

        # Total vulnerabilities found
        total_vulns = sum(len(v) for v in self.vulnerabilities.values())
        vec[2] = min(1.0, total_vulns / 100.0)

        # Compromise ratio
        if n_hosts > 0:
            vec[3] = len(self.compromised_hosts) / n_hosts

        # Credential count
        vec[4] = min(1.0, len(self.credentials) / 20.0)

        # High privilege ratio
        priv_hosts = sum(
            1 for p in self.privileges.values()
            if p in ("root", "system", "admin", "administrator")
        )
        if n_hosts > 0:
            vec[5] = priv_hosts / n_hosts

        # Data exfiltrated
        vec[6] = min(1.0, len(self.discovered_data) / 50.0)

        # Per-host features (up to 24 hosts, 1 value each)
        for i, host in enumerate(sorted(self.discovered_hosts)):
            if i >= 24:
                break
            host_score = 0.0
            host_score += len(self.open_ports.get(host, [])) / 100.0 * 0.3
            host_score += len(self.vulnerabilities.get(host, [])) / 20.0 * 0.3
            host_score += 0.2 if host in self.compromised_hosts else 0.0
            host_score += 0.2 if self.privileges.get(host, "") in ("root", "system") else 0.0
            vec[32 + i] = min(1.0, host_score)

        # Vulnerability severity distribution (binned)
        for i, host in enumerate(sorted(self.discovered_hosts)):
            if i >= 24:
                break
            vulns = self.vulnerabilities.get(host, [])
            vec[64 + i] = min(1.0, len(vulns) / 10.0)

        # Service type distribution
        service_counts: Dict[str, int] = {}
        for host, ports in self.services.items():
            for port, svc in ports.items():
                service_counts[svc] = service_counts.get(svc, 0) + 1

        common_services = ["http", "https", "ssh", "ftp", "smb", "rdp", "mysql", "postgresql", "redis", "mongodb"]
        for i, svc in enumerate(common_services):
            vec[96 + i] = min(1.0, service_counts.get(svc, 0) / 10.0)

        # Credential types
        cred_types = {
            "ssh": len([k for k in self.credentials if "ssh" in k.lower()]),
            "smb": len([k for k in self.credentials if "smb" in k.lower()]),
            "http": len([k for k in self.credentials if "http" in k.lower()]),
            "database": len([k for k in self.credentials if any(
                db in k.lower() for db in ["mysql", "postgres", "redis", "mongo"]
            )]),
        }
        for i, (_, count) in enumerate(cred_types.items()):
            vec[128 + i] = min(1.0, count / 5.0)

        return vec

    def to_dict(self) -> dict:
        """Serialize state to dictionary."""
        return {
            "discovered_hosts": sorted(self.discovered_hosts),
            "open_ports": {k: v for k, v in self.open_ports.items()},
            "vulnerabilities": {k: v for k, v in self.vulnerabilities.items()},
            "services": {k: {str(p): s for p, s in v.items()} for k, v in self.services.items()},
            "credentials": {k: "***" for k in self.credentials},  # Don't expose actual creds
            "privileges": {k: v for k, v in self.privileges.items()},
            "compromised_hosts": sorted(self.compromised_hosts),
            "discovered_data": sorted(self.discovered_data),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PenTestState":
        """Restore state from dictionary."""
        services = {}
        for k, v in data.get("services", {}).items():
            services[k] = {int(p): s for p, s in v.items()}

        return cls(
            discovered_hosts=set(data.get("discovered_hosts", [])),
            open_ports=data.get("open_ports", {}),
            vulnerabilities=data.get("vulnerabilities", {}),
            services=services,
            credentials=data.get("credentials", {}),
            privileges=data.get("privileges", {}),
            compromised_hosts=set(data.get("compromised_hosts", [])),
            discovered_data=set(data.get("discovered_data", [])),
        )

    @property
    def total_vulnerabilities(self) -> int:
        return sum(len(v) for v in self.vulnerabilities.values())

    @property
    def compromise_ratio(self) -> float:
        if not self.discovered_hosts:
            return 0.0
        return len(self.compromised_hosts) / len(self.discovered_hosts)

    @property
    def has_root_access(self) -> bool:
        return any(
            p in ("root", "system", "admin", "administrator")
            for p in self.privileges.values()
        )

    def summary(self) -> str:
        """Get human-readable state summary."""
        lines = [
            f"Hosts: {len(self.discovered_hosts)} discovered, {len(self.compromised_hosts)} compromised",
            f"Vulnerabilities: {self.total_vulnerabilities}",
            f"Credentials: {len(self.credentials)}",
            f"Root/Admin access: {'Yes' if self.has_root_access else 'No'}",
        ]
        return "\n".join(lines)
