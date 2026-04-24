"""
Environment Bridge Layer

Bridges the RL environment with real penetration testing tools.
When in real mode, executes actual tools and parses their output.
When in simulation mode, uses the existing abstract simulation.
"""

import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

from rl_agent.environment import PenTestEnvironment, PenTestState
from rl_agent.action import PenTestAction, ActionType
from pentest.executor import AttackExecutor
from pentest.output_parser import (
    NmapParser, HydraParser, MetasploitParser,
    parse_tool_output, ParsedHost, ParsedCredential
)

logger = logging.getLogger(__name__)


@dataclass
class EnvironmentConfig:
    """Configuration for the bridge layer."""
    real_mode: bool = False  # True = use real tools, False = simulation
    tool_timeout: int = 300  # Seconds to wait for tool execution
    auto_retry: bool = True  # Auto retry failed actions
    max_retries: int = 3
    unsafe_mode: bool = False  # Allow actual attacks (requires confirmation)


class EnvironmentBridge:
    """
    Bridge between RL environment and real penetration tools.

    Wraps PenTestEnvironment with optional real tool execution.
    When real_mode=True, actions trigger actual tool calls instead of simulation.
    """

    def __init__(
        self,
        base_env: PenTestEnvironment,
        executor: Optional[AttackExecutor] = None,
        config: Optional[EnvironmentConfig] = None
    ):
        self.base_env = base_env
        self.executor = executor
        self.config = config or EnvironmentConfig()

        # Results from real tool execution
        self.last_scan_results: List[ParsedHost] = []
        self.last_credentials: List[ParsedCredential] = []
        self.last_exploit_result: Optional[dict] = None

        # Safety confirmation
        self._confirmed_for_real_mode = False

    def step(self, action: PenTestAction) -> Tuple[PenTestState, float, bool, dict]:
        """
        Execute action in either real or simulation mode.

        Args:
            action: The action to execute

        Returns:
            Tuple of (state, reward, done, info)
        """
        if not self.config.real_mode or self.executor is None:
            # Use simulation
            return self.base_env.step(action)

        # Real mode - execute with actual tools
        return self._step_real(action)

    def _step_real(self, action: PenTestAction) -> Tuple[PenTestState, float, bool, dict]:
        """Execute action using real tools."""
        action_type = action.type

        # Check safety
        if self._requires_safety_confirmation(action_type) and not self._confirmed_for_real_mode:
            logger.warning(f"Action {action_type} requires safety confirmation in real mode")
            return self._safety_block_response(action)

        try:
            if action_type == ActionType.SCAN_PORT:
                return self._real_scan_port(action)
            elif action_type == ActionType.SCAN_SERVICE:
                return self._real_scan_service(action)
            elif action_type == ActionType.SCAN_VULN:
                return self._real_scan_vulnerability(action)
            elif action_type == ActionType.EXPLOIT_VULN:
                return self._real_exploit(action)
            elif action_type == ActionType.BRUTE_FORCE:
                return self._real_brute_force(action)
            elif action_type == ActionType.DUMP_CREDS:
                return self._real_dump_creds(action)
            elif action_type == ActionType.LATERAL_MOVE:
                return self._real_lateral_move(action)
            elif action_type == ActionType.PRIV_ESCALATE:
                return self._real_privesc(action)
            else:
                # Fall back to simulation for unsupported actions
                logger.info(f"Action {action_type} not supported in real mode, using simulation")
                return self.base_env.step(action)

        except Exception as e:
            logger.error(f"Real mode execution failed for {action_type}: {e}")
            if self.config.auto_retry:
                return self._retry_action(action)
            return self._error_response(action, str(e))

    def _real_scan_port(self, action: PenTestAction) -> Tuple[PenTestState, float, bool, dict]:
        """Execute real port scan."""
        target = action.target
        ports = action.parameters.get("ports", "1-1000")

        logger.info(f"[REAL] Scanning ports {ports} on {target}")

        result = self.executor.execute_scan(target, ports=ports)
        info = {"tool": "nmap", "target": target, "raw_output": result.stdout}

        if result.success:
            # Parse results
            parsed = NmapParser.parse_grepable(result.stdout)
            self.last_scan_results = parsed

            # Update state
            for host in parsed:
                self.base_env.state.discovered_hosts.add(host.ip)
                if host.ports:
                    self.base_env.state.open_ports[host.ip] = list(host.ports.keys())

            reward = 2.0
            info["parsed_hosts"] = [vars(h) for h in parsed]
            info["host_count"] = len(parsed)
        else:
            reward = -1.0
            info["error"] = result.stderr

        return self.base_env.state, reward, False, info

    def _real_scan_service(self, action: PenTestAction) -> Tuple[PenTestState, float, bool, dict]:
        """Execute real service scan."""
        target = action.target
        port = action.parameters.get("port", 80)

        logger.info(f"[REAL] Scanning services on {target}:{port}")

        result = self.executor.execute_scan(target, ports=str(port), scan_type="service")

        if result.success:
            # Parse service info
            parsed = NmapParser.parse_grepable(result.stdout)
            for host in parsed:
                if host.ip == target and port in host.ports:
                    service_info = host.ports[port]
                    self.base_env.state.services[target] = {
                        port: service_info.get("service", "unknown")
                    }
                    info = {
                        "tool": "nmap",
                        "service": service_info.get("service", ""),
                        "version": service_info.get("version", ""),
                    }
                    return self.base_env.state, 1.5, False, info

        return self.base_env.state, -0.5, False, {"error": "Service scan failed"}

    def _real_scan_vulnerability(self, action: PenTestAction) -> Tuple[PenTestState, float, bool, dict]:
        """Execute vulnerability scan."""
        target = action.target

        logger.info(f"[REAL] Vulnerability scan on {target}")

        result = self.executor.execute_scan(target, scan_type="vuln")

        if result.success:
            parsed = NmapParser.parse_xml(result.stdout)
            vulns_found = []
            for host in parsed:
                for vuln in host.vulnerabilities:
                    self.base_env.state.vulnerabilities.setdefault(host.ip, []).append(
                        vuln.get("cve_id", "")
                    )
                    vulns_found.append(vuln)

            info = {"vulnerabilities": vulns_found, "count": len(vulns_found)}
            return self.base_env.state, 3.0 if vulns_found else 0.5, False, info

        return self.base_env.state, -1.0, False, {"error": "Vuln scan failed"}

    def _real_exploit(self, action: PenTestAction) -> Tuple[PenTestState, float, bool, dict]:
        """Execute real exploit."""
        target = action.target
        cve_id = action.parameters.get("cve_id", "")

        logger.info(f"[REAL] Attempting exploit {cve_id} on {target}")

        result = self.executor.execute_exploit(target, cve_id)

        if result.success:
            self.base_env.state.compromised_hosts.add(target)
            self.base_env.state.privileges[target] = "user"
            info = {"exploit": cve_id, "session": result.session_id if hasattr(result, 'session_id') else None}
            return self.base_env.state, 5.0, False, info
        else:
            info = {"error": result.stderr, "exploit": cve_id}
            return self.base_env.state, -2.0, False, info

    def _real_brute_force(self, action: PenTestAction) -> Tuple[PenTestState, float, bool, dict]:
        """Execute real brute force attack."""
        target = action.target
        service = action.parameters.get("service", "ssh")
        port = action.parameters.get("port", 22)

        logger.info(f"[REAL] Brute forcing {service} on {target}:{port}")

        result = self.executor.execute_brute_force(
            target, service, port,
            wordlist=action.parameters.get("wordlist", "default")
        )

        if result.success:
            # Parse credentials
            creds = HydraParser.parse_stdout(result.stdout)
            self.last_credentials = creds

            for cred in creds:
                if cred.host == target:
                    self.base_env.state.credentials[cred.username] = cred.password

            info = {"credentials": [vars(c) for c in creds], "count": len(creds)}
            return self.base_env.state, 4.0 if creds else 1.0, False, info

        return self.base_env.state, -1.5, False, {"error": "Brute force failed"}

    def _real_dump_creds(self, action: PenTestAction) -> Tuple[PenTestState, float, bool, dict]:
        """Dump credentials from compromised host."""
        target = action.target

        logger.info(f"[REAL] Dumping credentials from {target}")

        result = self.executor.execute_mimikatz(target) if hasattr(self.executor, 'execute_mimikatz') else None

        if result and result.success:
            info = {"dumped_creds": result.stdout}
            return self.base_env.state, 3.0, False, info

        return self.base_env.state, -1.0, False, {"error": "Credential dump failed"}

    def _real_lateral_move(self, action: PenTestAction) -> Tuple[PenTestState, float, bool, dict]:
        """Execute lateral movement."""
        target = action.target
        source = action.parameters.get("source", "")

        logger.info(f"[REAL] Lateral movement from {source} to {target}")

        result = self.executor.execute_psexec(target, source) if hasattr(self.executor, 'execute_psexec') else None

        if result and result.success:
            self.base_env.state.compromised_hosts.add(target)
            info = {"moved_from": source, "target": target}
            return self.base_env.state, 4.0, False, info

        return self.base_env.state, -2.0, False, {"error": "Lateral move failed"}

    def _real_privesc(self, action: PenTestAction) -> Tuple[PenTestState, float, bool, dict]:
        """Execute privilege escalation."""
        target = action.target

        logger.info(f"[REAL] Privilege escalation on {target}")

        result = self.executor.execute_privesc(target) if hasattr(self.executor, 'execute_privesc') else None

        if result and result.success:
            self.base_env.state.privileges[target] = "admin"
            info = {"privilege_level": "admin"}
            return self.base_env.state, 5.0, False, info

        return self.base_env.state, -1.0, False, {"error": "Privilege escalation failed"}

    def _retry_action(self, action: PenTestAction) -> Tuple[PenTestState, float, bool, dict]:
        """Retry failed action with backoff."""
        for attempt in range(self.config.max_retries):
            logger.info(f"Retry attempt {attempt + 1} for {action.type}")
            time.sleep(2 ** attempt)  # Exponential backoff

            result = self._step_real(action)
            if result[1] > 0:  # If positive reward, consider it a success
                return result

        return self._error_response(action, f"Failed after {self.config.max_retries} retries")

    def _error_response(self, action: PenTestAction, error: str) -> Tuple[PenTestState, float, bool, dict]:
        """Generate error response."""
        info = {
            "action": action.type.value,
            "error": error,
            "error_type": "execution_failure",
        }
        return self.base_env.state, -1.0, False, info

    def _safety_block_response(self, action: PenTestAction) -> Tuple[PenTestState, float, bool, dict]:
        """Block action pending safety confirmation."""
        logger.warning(f"Safety block: {action.type} requires confirmation")
        info = {
            "action": action.type.value,
            "blocked": True,
            "requires_confirmation": True,
            "hint": "Call confirm_real_mode() to enable dangerous actions",
        }
        return self.base_env.state, 0.0, False, info

    def _requires_safety_confirmation(self, action_type: ActionType) -> bool:
        """Check if action requires safety confirmation."""
        dangerous_actions = {
            ActionType.EXPLOIT_VULN,
            ActionType.BRUTE_FORCE,
            ActionType.LATERAL_MOVE,
            ActionType.PRIV_ESCALATE,
            ActionType.EXFILTRATE,
        }
        return action_type in dangerous_actions

    def confirm_real_mode(self) -> None:
        """Confirm real mode to allow dangerous actions."""
        if not self.config.unsafe_mode:
            logger.warning("Safety override requested - this is DANGEROUS")
        self._confirmed_for_real_mode = True
        logger.info("Real mode confirmed - dangerous actions enabled")

    def reset(self) -> PenTestState:
        """Reset the environment."""
        state = self.base_env.reset()
        self.last_scan_results = []
        self.last_credentials = []
        self.last_exploit_result = None
        return state

    def enable_real_mode(self, executor: AttackExecutor, unsafe: bool = False) -> None:
        """
        Enable real mode with specified executor.

        Args:
            executor: AttackExecutor instance with real tool paths
            unsafe: If True, skip dangerous action warnings
        """
        self.executor = executor
        self.config.real_mode = True
        self.config.unsafe_mode = unsafe

        if unsafe:
            self._confirmed_for_real_mode = True

        logger.info(f"Real mode enabled (unsafe={unsafe})")

    def disable_real_mode(self) -> None:
        """Switch back to simulation mode."""
        self.config.real_mode = False
        self._confirmed_for_real_mode = False
        logger.info("Real mode disabled, using simulation")

    def get_last_scan_results(self) -> List[ParsedHost]:
        """Get results from last scan."""
        return self.last_scan_results

    def get_last_credentials(self) -> List[ParsedCredential]:
        """Get credentials from last brute force."""
        return self.last_credentials

    def get_state(self) -> PenTestState:
        """Get current environment state."""
        return self.base_env.state

    def get_tool_results(self, tool: str) -> dict:
        """Get cached results for a specific tool."""
        if tool == "nmap":
            return {"hosts": [vars(h) for h in self.last_scan_results]}
        elif tool == "hydra":
            return {"credentials": [vars(c) for c in self.last_credentials]}
        else:
            return {}