"""
Penetration Test RL Environment

Gymnasium-style environment for training RL agents
on simulated penetration testing scenarios.
"""

from typing import Tuple, List, Dict, Optional
import random
import logging

from .state import PenTestState
from .action import PenTestAction, ActionType, ActionSpace

logger = logging.getLogger(__name__)


class PenTestEnvironment:
    """
    Simulated penetration testing environment for RL training.

    Follows OpenAI Gymnasium interface:
    - reset() -> initial state
    - step(action) -> (next_state, reward, done, info)
    """

    def __init__(
        self,
        hosts: Optional[List[dict]] = None,
        network_config: Optional[dict] = None,
    ):
        """
        Initialize environment with target network configuration.

        Args:
            hosts: List of host configurations
                {
                    "ip": "192.168.1.100",
                    "os": "Linux",
                    "ports": {22: "ssh", 80: "http"},
                    "vulnerabilities": ["CVE-2021-44228"],
                    "services": {"ssh": {"users": ["admin"]}},
                    "credential": {"admin": "password123"},
                }
            network_config: Network configuration
                {
                    "subnet": "192.168.1.0/24",
                    "gateway": "192.168.1.1",
                }
        """
        self.hosts = hosts or self._default_network()
        self.network_config = network_config or {"subnet": "192.168.1.0/24"}
        self.action_space = ActionSpace()
        self.state = PenTestState()
        self.step_count = 0
        self.max_steps = 100
        self.history: List[Dict] = []

    def _default_network(self) -> List[dict]:
        """Generate a default target network for testing."""
        return [
            {
                "ip": "192.168.1.100",
                "os": "Linux",
                "ports": {22: "ssh", 80: "http", 443: "https"},
                "vulnerabilities": ["CVE-2021-44228"],
                "services": {
                    "ssh": {"users": ["admin", "user"]},
                    "http": {"framework": "django", "version": "3.2"},
                },
                "credential": {"admin": "admin123"},
                "data": ["database_dump.sql", "config.py"],
            },
            {
                "ip": "192.168.1.101",
                "os": "Windows",
                "ports": {445: "smb", 3389: "rdp"},
                "vulnerabilities": ["CVE-2017-0144"],
                "services": {
                    "smb": {"version": "SMBv1"},
                },
                "credential": {"administrator": "P@ssw0rd"},
                "data": ["passwords.xlsx", "network_diagram.pdf"],
            },
            {
                "ip": "192.168.1.102",
                "os": "Linux",
                "ports": {22: "ssh", 3306: "mysql", 6379: "redis"},
                "vulnerabilities": [],
                "services": {
                    "mysql": {"users": ["root"], "version": "5.7"},
                    "redis": {"version": "6.0", "auth": False},
                },
                "credential": {"root": "toor"},
                "data": ["customer_data.sql"],
            },
        ]

    def reset(self) -> PenTestState:
        """Reset environment to initial state."""
        self.state = PenTestState()
        self.step_count = 0
        self.history = []
        self.action_space.clear_cache()

        # Initial discovery: know the network exists
        self.state.discovered_hosts.add("192.168.1.0/24")

        return self.state

    def step(self, action: PenTestAction) -> Tuple[PenTestState, float, bool, dict]:
        """
        Execute an action and return results.

        Returns:
            (next_state, reward, done, info)
        """
        self.step_count += 1
        info: Dict = {
            "action": action.to_dict(),
            "step": self.step_count,
        }

        # Execute action based on type
        reward = 0.0
        done = False

        if action.type == ActionType.SCAN_NETWORK:
            reward, done, info = self._execute_scan_network(action, info)
        elif action.type == ActionType.SCAN_PORT:
            reward, done, info = self._execute_scan_port(action, info)
        elif action.type == ActionType.ENUMERATE_SERVICE:
            reward, done, info = self._execute_enum_service(action, info)
        elif action.type == ActionType.EXPLOIT_VULN:
            reward, done, info = self._execute_exploit(action, info)
        elif action.type == ActionType.BRUTE_FORCE:
            reward, done, info = self._execute_brute_force(action, info)
        elif action.type == ActionType.LATERAL_MOVE:
            reward, done, info = self._execute_lateral_move(action, info)
        elif action.type == ActionType.PRIV_ESCALATE:
            reward, done, info = self._execute_priv_escalate(action, info)
        elif action.type == ActionType.DUMP_CREDS:
            reward, done, info = self._execute_dump_creds(action, info)
        elif action.type == ActionType.EXFILTRATE:
            reward, done, info = self._execute_exfiltrate(action, info)

        # Step penalty to encourage efficiency
        reward -= 0.1

        # Check if max steps reached
        if self.step_count >= self.max_steps:
            done = True
            info["reason"] = "max_steps_reached"

        # Record history
        self.history.append({
            "step": self.step_count,
            "action": action.to_dict(),
            "reward": reward,
            "done": done,
        })

        info["state_summary"] = self.state.summary()
        return self.state, reward, done, info

    def _find_host(self, ip: str) -> Optional[dict]:
        """Find host config by IP."""
        for host in self.hosts:
            if host["ip"] == ip:
                return host
        return None

    def _execute_scan_network(self, action: PenTestAction, info: Dict) -> Tuple[float, bool, Dict]:
        """Execute network scan to discover hosts."""
        discovered = 0
        for host in self.hosts:
            if host["ip"] not in self.state.discovered_hosts:
                self.state.discovered_hosts.add(host["ip"])
                discovered += 1
                self.action_space.register_target(host["ip"])

        reward = discovered * 0.5
        info["hosts_discovered"] = discovered
        return reward, False, info

    def _execute_scan_port(self, action: PenTestAction, info: Dict) -> Tuple[float, bool, Dict]:
        """Execute port scan on a host."""
        host_ip = action.target
        host = self._find_host(host_ip)

        if host is None:
            info["error"] = "Host not found"
            return -0.5, False, info

        ports_found = []
        host_ports = host.get("ports", {})
        # Handle both dict {22: "ssh"} and list [{port: 22, service: "ssh"}] formats
        if isinstance(host_ports, list):
            for port_entry in host_ports:
                if isinstance(port_entry, dict):
                    port = port_entry.get("port", 0)
                    service = port_entry.get("service", "unknown")
                else:
                    port = int(port_entry)
                    service = "unknown"
                if port and port not in self.state.open_ports.get(host_ip, []):
                    self.state.open_ports.setdefault(host_ip, []).append(port)
                    self.state.services.setdefault(host_ip, {})[port] = service
                    ports_found.append(port)
        elif isinstance(host_ports, dict):
            for port, service in host_ports.items():
                if port not in self.state.open_ports.get(host_ip, []):
                    self.state.open_ports.setdefault(host_ip, []).append(port)
                    self.state.services.setdefault(host_ip, {})[port] = service
                    ports_found.append(port)

        reward = len(ports_found) * 0.3
        info["ports_found"] = ports_found
        return reward, False, info

    def _execute_enum_service(self, action: PenTestAction, info: Dict) -> Tuple[float, bool, Dict]:
        """Enumerate a service for more details."""
        reward = 0.2  # Small reward for enumeration
        info["details_found"] = True
        return reward, False, info

    def _execute_exploit(self, action: PenTestAction, info: Dict) -> Tuple[float, bool, Dict]:
        """Execute a vulnerability exploit."""
        host_ip = action.target
        cve_id = action.cve_id or action.parameters.get("cve_id", "")
        host = self._find_host(host_ip)

        if host is None:
            info["error"] = "Host not found"
            return -1.0, False, info

        if cve_id not in host.get("vulnerabilities", []):
            info["error"] = f"Vulnerability {cve_id} not present on host"
            return -0.5, False, info

        # Exploit succeeds
        self.state.compromised_hosts.add(host_ip)
        self.state.privileges[host_ip] = "user"
        self.state.vulnerabilities.setdefault(host_ip, []).append(cve_id)

        reward = 5.0  # Big reward for successful exploitation
        info["exploit_success"] = True
        info["cve_id"] = cve_id
        return reward, False, info

    def _execute_brute_force(self, action: PenTestAction, info: Dict) -> Tuple[float, bool, Dict]:
        """Execute brute force attack."""
        host_ip = action.target.split(":")[0] if ":" in action.target else action.target
        host = self._find_host(host_ip)

        if host is None:
            info["error"] = "Host not found"
            return -0.5, False, info

        creds = host.get("credential", {})
        if creds:
            # Simulate finding correct credential (with some probability)
            success_prob = 0.3 if host_ip not in self.state.compromised_hosts else 0.5
            if random.random() < success_prob:
                username = list(creds.keys())[0]
                self.state.credentials[f"{host_ip}:{action.parameters.get('service', 'unknown')}"] = username
                self.state.compromised_hosts.add(host_ip)
                self.state.privileges[host_ip] = "user"
                reward = 3.0
                info["brute_force_success"] = True
                return reward, False, info

        info["error"] = "Brute force failed"
        return -0.3, False, info

    def _execute_lateral_move(self, action: PenTestAction, info: Dict) -> Tuple[float, bool, Dict]:
        """Execute lateral movement."""
        source = action.parameters.get("source", "")
        target = action.target

        if source not in self.state.compromised_hosts:
            info["error"] = "Source host not compromised"
            return -0.5, False, info

        host = self._find_host(target)
        if host is None:
            info["error"] = "Target host not found"
            return -0.3, False, info

        # Try to move using compromised credentials
        success = False
        if self.state.credentials:
            # Higher success rate with more credentials
            success_prob = min(0.8, len(self.state.credentials) * 0.2)
            if random.random() < success_prob:
                success = True

        if success:
            self.state.compromised_hosts.add(target)
            self.state.privileges[target] = "user"
            reward = 4.0
            info["lateral_move_success"] = True
        else:
            reward = -0.3
            info["error"] = "Lateral movement failed"

        return reward, False, info

    def _execute_priv_escalate(self, action: PenTestAction, info: Dict) -> Tuple[float, bool, Dict]:
        """Execute privilege escalation."""
        host_ip = action.target

        if host_ip not in self.state.compromised_hosts:
            info["error"] = "Host not compromised"
            return -0.5, False, info

        host = self._find_host(host_ip)
        if host is None:
            info["error"] = "Host not found"
            return -0.3, False, info

        current_priv = self.state.privileges.get(host_ip, "user")
        if current_priv in ("root", "system", "admin", "administrator"):
            info["error"] = "Already at highest privilege"
            return -0.1, False, info

        # Success probability based on vulnerabilities
        vulns = host.get("vulnerabilities", [])
        success_prob = 0.4 + len(vulns) * 0.1

        if random.random() < success_prob:
            os_type = host.get("os", "").lower()
            new_priv = "root" if "linux" in os_type else "system"
            self.state.privileges[host_ip] = new_priv
            reward = 3.0
            info["priv_escalate_success"] = True
            info["new_privilege"] = new_priv
        else:
            reward = -0.3
            info["error"] = "Privilege escalation failed"

        return reward, False, info

    def _execute_dump_creds(self, action: PenTestAction, info: Dict) -> Tuple[float, bool, Dict]:
        """Dump credentials from a compromised host."""
        host_ip = action.target

        if host_ip not in self.state.compromised_hosts:
            info["error"] = "Host not compromised"
            return -0.5, False, info

        host = self._find_host(host_ip)
        if host is None:
            return -0.3, False, info

        creds = host.get("credential", {})
        found = 0
        for username in creds:
            key = f"{host_ip}:{username}"
            if key not in self.state.credentials:
                self.state.credentials[key] = creds[username]
                found += 1

        reward = found * 1.0
        info["credentials_found"] = found
        return reward, False, info

    def _execute_exfiltrate(self, action: PenTestAction, info: Dict) -> Tuple[float, bool, Dict]:
        """Exfiltrate data from a compromised host."""
        host_ip = action.target

        if host_ip not in self.state.compromised_hosts:
            info["error"] = "Host not compromised"
            return -0.5, False, info

        host = self._find_host(host_ip)
        if host is None:
            return -0.3, False, info

        data = host.get("data", [])
        found = 0
        for item in data:
            if item not in self.state.discovered_data:
                self.state.discovered_data.add(item)
                found += 1

        reward = found * 2.0
        info["data_exfiltrated"] = found

        # Check if all data collected from all compromised hosts
        all_collected = True
        for h in self.hosts:
            if h["ip"] in self.state.compromised_hosts:
                for d in h.get("data", []):
                    if d not in self.state.discovered_data:
                        all_collected = False
                        break

        if all_collected and len(self.state.compromised_hosts) == len(self.hosts):
            reward += 10.0  # Bonus for complete compromise
            return reward, True, info

        return reward, False, info

    def calculate_reward(self, action: PenTestAction, success: bool) -> float:
        """Calculate reward for an action, mapping DE fitness_fn pattern."""
        base_rewards = {
            ActionType.SCAN_NETWORK: 0.5,
            ActionType.SCAN_PORT: 0.3,
            ActionType.ENUMERATE_SERVICE: 0.2,
            ActionType.EXPLOIT_VULN: 5.0,
            ActionType.BRUTE_FORCE: 3.0,
            ActionType.LATERAL_MOVE: 4.0,
            ActionType.PRIV_ESCALATE: 3.0,
            ActionType.DUMP_CREDS: 1.0,
            ActionType.EXFILTRATE: 2.0,
        }
        base = base_rewards.get(action.type, 0.1)
        return base if success else -0.3

    def render(self) -> str:
        """Render current state as text."""
        lines = [
            f"=== Penetration Test State (Step {self.step_count}) ===",
            self.state.summary(),
            "",
            "Compromised Hosts:",
        ]
        for host in sorted(self.state.compromised_hosts):
            priv = self.state.privileges.get(host, "unknown")
            lines.append(f"  {host} [{priv}]")

        lines.append("\nDiscovered Vulnerabilities:")
        for host, vulns in self.state.vulnerabilities.items():
            for cve in vulns:
                lines.append(f"  {host}: {cve}")

        return "\n".join(lines)
