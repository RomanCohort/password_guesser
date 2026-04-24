"""
Dual-Mode Environment

Extends PenTestEnvironment with real mode support while maintaining
simulation mode for safe training. Includes safety mechanisms and
confirmation prompts for dangerous real-mode operations.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass

from rl_agent.environment import PenTestEnvironment, PenTestState
from rl_agent.action import PenTestAction, ActionType
from rl_agent.environment_bridge import EnvironmentBridge, EnvironmentConfig
from pentest.executor import AttackExecutor
from pentest.network import ConnectionPool, NetworkScanner

logger = logging.getLogger(__name__)


@dataclass
class DualModeConfig:
    """Configuration for dual-mode environment."""
    default_mode: str = "simulation"  # "simulation" or "real"
    allow_real_mode: bool = True     # Allow switching to real mode
    require_confirmation: bool = True  # Require confirmation for real mode
    safety_dangerous_actions: List[ActionType] = None
    auto_save_on_real: bool = True    # Auto-save state before real actions

    def __post_init__(self):
        if self.safety_dangerous_actions is None:
            self.safety_dangerous_actions = [
                ActionType.EXPLOIT_VULN,
                ActionType.BRUTE_FORCE,
                ActionType.LATERAL_MOVE,
                ActionType.PRIV_ESCALATE,
                ActionType.EXFILTRATE,
            ]


class DualModeEnvironment:
    """
    Environment that supports both simulation and real modes.

    - Simulation mode: Uses abstract state transitions (existing behavior)
    - Real mode: Executes real tools and parses their output

    Safety features:
    - Confirmation required for dangerous actions in real mode
    - State snapshots before real operations
    - Automatic fallback to simulation on failure
    """

    def __init__(
        self,
        hosts: List[Dict] = None,
        config: Optional[DualModeConfig] = None,
        executor: Optional[AttackExecutor] = None,
    ):
        self.config = config or DualModeConfig()

        # Create base simulation environment
        self.sim_env = PenTestEnvironment(hosts=hosts or [])

        # Create bridge if executor provided
        self.bridge: Optional[EnvironmentBridge] = None
        if executor:
            self.bridge = EnvironmentBridge(
                base_env=self.sim_env,
                executor=executor,
            )

        # Current mode
        self._mode = self.config.default_mode
        self._confirmed = False

        # Safety: state snapshots
        self._snapshots: List[Dict] = []

        # Network layer for Python-native operations
        self._network: Optional[NetworkScanner] = None

        # Callbacks
        self._on_real_action: Optional[Callable] = None
        self._on_mode_switch: Optional[Callable] = None

    @property
    def mode(self) -> str:
        """Current execution mode."""
        return self._mode

    @property
    def is_real_mode(self) -> bool:
        """Check if in real mode."""
        return self._mode == "real" and self.bridge is not None

    def step(self, action: PenTestAction) -> Tuple[PenTestState, float, bool, dict]:
        """
        Execute action in current mode.

        Args:
            action: The action to execute

        Returns:
            Tuple of (state, reward, done, info)
        """
        # Safety check for dangerous actions
        if self.is_real_mode and self.config.require_confirmation:
            if action.type in self.config.safety_dangerous_actions:
                if not self._confirmed:
                    logger.warning(
                        f"Dangerous action {action.type.value} requires confirmation"
                    )
                    return self._block_dangerous_action(action)

        # Pre-action snapshot in real mode
        if self.is_real_mode and self.config.auto_save_on_real:
            self._save_snapshot()

        # Execute based on mode
        if self._mode == "real" and self.bridge:
            return self.bridge.step(action)
        else:
            return self.sim_env.step(action)

    def _save_snapshot(self) -> None:
        """Save current state snapshot for safety."""
        self._snapshots.append({
            "state": self.sim_env.state.to_dict(),
            "mode": self._mode,
            "timestamp": asyncio.get_event_loop().time(),
        })

        # Keep only last 10 snapshots
        if len(self._snapshots) > 10:
            self._snapshots.pop(0)

    def restore_snapshot(self, index: int = -1) -> bool:
        """
        Restore a previous state snapshot.

        Args:
            index: Snapshot index (-1 = last)

        Returns:
            True if restore successful
        """
        if not self._snapshots:
            logger.warning("No snapshots to restore")
            return False

        snapshot = self._snapshots[index]
        state_dict = snapshot["state"]

        # Restore state
        self.sim_env.state = PenTestState.from_dict(state_dict)

        logger.info(f"Restored snapshot from {snapshot['timestamp']}")
        return True

    def _block_dangerous_action(self, action: PenTestAction) -> Tuple[PenTestState, float, bool, dict]:
        """Block dangerous action pending confirmation."""
        logger.warning(
            f"BLOCKED: {action.type.value} on {action.target} "
            f"(need confirmation, use confirm_dangerous_actions())"
        )

        info = {
            "blocked": True,
            "action": action.type.value,
            "reason": "confirmation_required",
            "hint": "Call confirm_dangerous_actions() before executing dangerous actions",
        }

        return self.sim_env.state, 0.0, False, info

    def confirm_dangerous_actions(self) -> None:
        """Confirm to allow dangerous real-mode actions."""
        if not self.config.allow_real_mode:
            raise PermissionError("Real mode is disabled")

        logger.warning("=" * 60)
        logger.warning("CONFIRMATION: Enabling dangerous real-mode actions")
        logger.warning("=" * 60)

        self._confirmed = True

        if self.bridge:
            self.bridge.confirm_real_mode()

    def enable_real_mode(
        self,
        executor: AttackExecutor,
        unsafe: bool = False,
    ) -> None:
        """
        Enable real mode with actual tool execution.

        Args:
            executor: AttackExecutor with configured tool paths
            unsafe: If True, skip safety confirmation (DANGEROUS)
        """
        if not self.config.allow_real_mode:
            logger.error("Real mode is disabled in configuration")
            return

        if self.bridge is None:
            self.bridge = EnvironmentBridge(
                base_env=self.sim_env,
                executor=executor,
            )

        self.bridge.enable_real_mode(executor, unsafe=unsafe)
        self._mode = "real"

        if self._on_mode_switch:
            self._on_mode_switch("real")

        logger.info("=" * 60)
        logger.info("REAL MODE ENABLED - Using actual penetration tools")
        logger.warning("Ensure you have authorization for all target systems!")
        logger.info("=" * 60)

    def disable_real_mode(self) -> None:
        """Switch back to simulation mode."""
        self._mode = "simulation"
        self._confirmed = False

        if self.bridge:
            self.bridge.disable_real_mode()

        if self._on_mode_switch:
            self._on_mode_switch("simulation")

        logger.info("Simulation mode enabled")

    def toggle_mode(self, executor: Optional[AttackExecutor] = None) -> str:
        """
        Toggle between simulation and real mode.

        Args:
            executor: Required when switching to real mode

        Returns:
            New mode name
        """
        if self._mode == "simulation":
            if executor is None and self.bridge is None:
                logger.error("Need executor to enable real mode")
                return "simulation"
            if executor:
                self.enable_real_mode(executor)
            else:
                self._mode = "real"
                if self.bridge:
                    self.bridge.config.real_mode = True
        else:
            self.disable_real_mode()

        return self._mode

    def reset(self) -> PenTestState:
        """Reset environment to initial state."""
        if self.bridge:
            return self.bridge.reset()
        return self.sim_env.reset()

    @property
    def state(self) -> PenTestState:
        """Get current state."""
        if self.bridge and self.is_real_mode:
            return self.bridge.get_state()
        return self.sim_env.state

    def get_network_scanner(self) -> NetworkScanner:
        """Get Python-native network scanner."""
        if self._network is None:
            self._network = NetworkScanner(ConnectionPool())
        return self._network

    def get_last_real_results(self) -> Dict[str, Any]:
        """Get results from last real-mode action."""
        if self.bridge:
            return {
                "scan_results": self.bridge.get_last_scan_results(),
                "credentials": self.bridge.get_last_credentials(),
                "state": self.bridge.get_state().to_dict(),
            }
        return {}

    def register_callback(self, event: str, callback: Callable) -> None:
        """
        Register callback for events.

        Events:
        - "mode_switch": Called when mode changes
        - "real_action": Called before each real-mode action
        """
        if event == "mode_switch":
            self._on_mode_switch = callback
        elif event == "real_action":
            self._on_mode_switch = callback

    def get_safety_status(self) -> Dict[str, Any]:
        """Get current safety status."""
        return {
            "mode": self._mode,
            "confirmed": self._confirmed,
            "bridge_available": self.bridge is not None,
            "real_mode_active": self.is_real_mode,
            "snapshots_count": len(self._snapshots),
            "dangerous_actions_blocked": self.config.safety_dangerous_actions,
        }

    def simulate_attack_path(self, path: List[ActionType]) -> Dict[str, Any]:
        """
        Simulate an attack path in simulation mode.

        Useful for testing and planning without real execution.

        Args:
            path: List of action types to execute

        Returns:
            Dict with simulation results
        """
        self._mode = "simulation"
        if self.bridge:
            self.bridge.config.real_mode = False

        state = self.reset()
        steps = []
        total_reward = 0.0

        for i, action_type in enumerate(path):
            # Create dummy action
            action = PenTestAction(
                type=action_type,
                target=list(state.discovered_hosts)[0] if state.discovered_hosts else "unknown",
            )

            new_state, reward, done, info = self.step(action)

            steps.append({
                "step": i + 1,
                "action": action_type.value,
                "reward": reward,
                "done": done,
            })

            total_reward += reward
            state = new_state

            if done:
                break

        return {
            "total_steps": len(steps),
            "total_reward": total_reward,
            "steps": steps,
            "final_state": state.to_dict(),
            "reached_goal": total_reward > 0,
        }

    def estimate_realistic_duration(self, path: List[ActionType]) -> float:
        """
        Estimate realistic time for an attack path.

        Based on typical tool execution times:
        - port scan: 30s - 5min
        - service scan: 10s - 2min
        - vuln scan: 1min - 10min
        - brute force: 1min - 30min
        - exploit: 10s - 2min

        Returns:
            Estimated duration in seconds
        """
        durations = {
            ActionType.SCAN_PORT: 60,           # 1 min for quick scan
            ActionType.SCAN_NETWORK: 30,        # 30s per service
            ActionType.ENUMERATE_SERVICE: 120,  # 2 min for enumeration
            ActionType.EXPLOIT_VULN: 60,        # 1 min for exploit attempt
            ActionType.BRUTE_FORCE: 600,        # 10 min for brute force
            ActionType.DUMP_CREDS: 30,          # 30s for credential dump
            ActionType.LATERAL_MOVE: 120,       # 2 min for lateral move
            ActionType.PRIV_ESCALATE: 180,      # 3 min for privesc
            ActionType.EXFILTRATE: 60,          # 1 min for exfil
        }

        total = sum(durations.get(at, 60) for at in path)
        return total


# Convenience functions

def create_dual_mode_env(
    hosts: List[Dict] = None,
    executor: Optional[AttackExecutor] = None,
    mode: str = "simulation",
) -> DualModeEnvironment:
    """
    Create a dual-mode environment.

    Args:
        hosts: Target host definitions
        executor: Attack executor for real mode
        mode: Initial mode ("simulation" or "real")

    Returns:
        Configured DualModeEnvironment
    """
    config = DualModeConfig(
        default_mode=mode,
        allow_real_mode=executor is not None,
    )

    env = DualModeEnvironment(hosts=hosts, config=config, executor=executor)

    if mode == "real" and executor:
        env.enable_real_mode(executor)

    return env


def quick_test_with_real(target: str, ports: List[int]) -> Dict[str, Any]:
    """
    Quick test using Python-native tools without full environment setup.

    Args:
        target: Target hostname/IP
        ports: List of ports to scan

    Returns:
        Scan results
    """
    from pentest.windows_compat import WindowsToolkit

    toolkit = WindowsToolkit()

    async def _scan():
        results = await toolkit.scan(target, ports)
        return results

    return asyncio.run(_scan())