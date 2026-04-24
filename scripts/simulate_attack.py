"""
Simulated Target Attack Test

Tests the full attack chain against a simulated vulnerable target.
This uses the simulation environment to validate the complete attack
pipeline without needing a real target.

Can also connect to a real target when provided with VPN access.

Usage:
    # Pure simulation (no network needed)
    python scripts/simulate_attack.py

    # With a real target (after VPN connection)
    python scripts/simulate_attack.py --target 10.10.x.x --real

    # Target TryHackMe-style vulnerable VM
    python scripts/simulate_attack.py --target 10.10.x.x --real --full-chain
"""

import asyncio
import json
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from pentest.orchestrator import PenTestOrchestrator, PenTestConfig
from models.llm_provider import LLMConfig
from rl_agent.dual_mode_env import DualModeEnvironment, DualModeConfig
from rl_agent.action import ActionType, PenTestAction
from pentest.windows_compat import WindowsToolkit

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


# Common vulnerable VM configurations for simulation
VULN_TARGETS = {
    "metasploitable2": {
        "name": "Metasploitable 2",
        "description": "Intentionally vulnerable Ubuntu Linux",
        "hosts": [{
            "ip": "192.168.1.100",
            "os": "Linux",
            "ports": [
                {"port": 21, "service": "ftp"},
                {"port": 22, "service": "ssh"},
                {"port": 23, "service": "telnet"},
                {"port": 25, "service": "smtp"},
                {"port": 53, "service": "dns"},
                {"port": 80, "service": "http"},
                {"port": 111, "service": "rpcbind"},
                {"port": 139, "service": "netbios"},
                {"port": 445, "service": "smb"},
                {"port": 512, "service": "rexec"},
                {"port": 513, "service": "rlogin"},
                {"port": 514, "service": "rsh"},
                {"port": 1099, "service": "java-rmi"},
                {"port": 1524, "service": "ingreslock"},
                {"port": 2049, "service": "nfs"},
                {"port": 2121, "service": "ftp"},
                {"port": 3306, "service": "mysql"},
                {"port": 5432, "service": "postgresql"},
                {"port": 5900, "service": "vnc"},
                {"port": 6000, "service": "x11"},
                {"port": 6667, "service": "irc"},
                {"port": 8009, "service": "ajp"},
                {"port": 8180, "service": "tomcat"},
                {"port": 8787, "service": "java-debug"},
            ],
            "vulnerabilities": [
                {"cve_id": "CVE-2011-2523", "name": "vsftpd backdoor"},
                {"cve_id": "CVE-2008-4250", "name": "Samba CVE-2008-4250"},
                {"cve_id": "CVE-2012-1823", "name": "PHP-CGI"},
                {"cve_id": "CVE-2010-2861", "name": "ColdFusion"},
                {"cve_id": "CVE-2011-3192", "name": "Apache"},
                {"cve_id": "CVE-2010-0738", "name": "Tomcat"},
            ],
        }]
    },
    "dvwa": {
        "name": "DVWA",
        "description": "Damn Vulnerable Web Application",
        "hosts": [{
            "ip": "192.168.1.101",
            "os": "Linux",
            "ports": [
                {"port": 80, "service": "http"},
                {"port": 3306, "service": "mysql"},
            ],
            "vulnerabilities": [
                {"cve_id": "SQLi", "name": "SQL Injection"},
                {"cve_id": "XSS", "name": "Cross-Site Scripting"},
                {"cve_id": "RCE", "name": "Remote Code Execution"},
            ],
        }]
    },
    "kioptrix": {
        "name": "Kioptrix Level 1",
        "description": "Beginner vulnerable VM",
        "hosts": [{
            "ip": "192.168.1.102",
            "os": "Linux",
            "ports": [
                {"port": 22, "service": "ssh"},
                {"port": 80, "service": "http"},
                {"port": 111, "service": "rpcbind"},
                {"port": 139, "service": "netbios"},
                {"port": 443, "service": "https"},
                {"port": 1024, "service": "unknown"},
            ],
            "vulnerabilities": [
                {"cve_id": "CVE-2009-3103", "name": "Samba"},
                {"cve_id": "CVE-2003-0201", "name": "Apache mod_ssl"},
            ],
        }]
    },
    "windows_server": {
        "name": "Vulnerable Windows Server",
        "description": "Unpatched Windows Server 2008",
        "hosts": [{
            "ip": "192.168.1.103",
            "os": "Windows",
            "ports": [
                {"port": 80, "service": "http"},
                {"port": 135, "service": "msrpc"},
                {"port": 139, "service": "netbios"},
                {"port": 443, "service": "https"},
                {"port": 445, "service": "smb"},
                {"port": 1433, "service": "mssql"},
                {"port": 3389, "service": "rdp"},
                {"port": 5985, "service": "winrm"},
                {"port": 8080, "service": "http-proxy"},
            ],
            "vulnerabilities": [
                {"cve_id": "CVE-2017-0144", "name": "EternalBlue (MS17-010)"},
                {"cve_id": "CVE-2019-0708", "name": "BlueKeep"},
                {"cve_id": "CVE-2021-27065", "name": "Exchange ProxyLogon"},
            ],
        }]
    },
}


class SimulatedAttackTest:
    """
    Full simulated attack test against a configured target.

    Flow:
    1. Initialize orchestrator with target config
    2. Run RL agent in simulation mode
    3. Log all actions and results
    4. Generate attack report
    """

    def __init__(
        self,
        target_key: str = "metasploitable2",
        max_steps: int = 30,
        verbose: bool = True,
    ):
        self.target_config = VULN_TARGETS.get(target_key, VULN_TARGETS["metasploitable2"])
        self.max_steps = max_steps
        self.verbose = verbose
        self.report: Dict[str, Any] = {
            "target": self.target_config["name"],
            "start_time": datetime.now().isoformat(),
            "steps": [],
            "findings": [],
        }

    def run_simulation(self) -> Dict[str, Any]:
        """
        Run full simulated attack using the orchestrator.
        """
        print("\n" + "="*70)
        print(f"  SIMULATED ATTACK: {self.target_config['name']}")
        print(f"  Description: {self.target_config['description']}")
        print("="*70)

        # Create orchestrator config
        config = PenTestConfig(
            max_steps=self.max_steps,
            enable_attack_team=False,  # Simplified for simulation
            enable_self_improvement=False,
        )

        # Initialize orchestrator
        orchestrator = PenTestOrchestrator(config)

        # Load target data
        scan_data = {
            "format": "manual",
            "data": self.target_config["hosts"],
        }

        print(f"\n[*] Initializing with {len(self.target_config['hosts'])} host(s)")
        for host in self.target_config["hosts"]:
            print(f"    Host: {host['ip']} ({host['os']})")
            print(f"    Ports: {len(host['ports'])} open")
            print(f"    Vulnerabilities: {len(host.get('vulnerabilities', []))}")

        orchestrator.initialize_from_scan(scan_data)

        # Run autonomous attack
        print(f"\n[*] Starting autonomous attack (max {self.max_steps} steps)")
        print("-"*70)

        results = orchestrator.run_autonomous(
            target_goal="full_compromise",
            max_steps=self.max_steps,
            verbose=self.verbose,
        )

        # Process results
        self._process_results(results)

        return self.report

    def run_dual_mode_test(self) -> Dict[str, Any]:
        """
        Test dual-mode environment (simulation + real mode capability).
        """
        print("\n" + "="*70)
        print(f"  DUAL-MODE TEST: {self.target_config['name']}")
        print("="*70)

        # Create dual-mode environment
        hosts = self.target_config["hosts"]
        config = DualModeConfig(default_mode="simulation")
        env = DualModeEnvironment(hosts=hosts, config=config)

        print(f"\n[*] Mode: {env.mode}")
        safety = env.get_safety_status()
        safety["dangerous_actions_blocked"] = [a.value for a in safety.get("dangerous_actions_blocked", [])]
        print(f"[*] Safety status: {json.dumps(safety, indent=2)}")

        # Simulate attack path
        attack_path = [
            ActionType.SCAN_NETWORK,
            ActionType.SCAN_PORT,
            ActionType.ENUMERATE_SERVICE,
            ActionType.EXPLOIT_VULN,
            ActionType.PRIV_ESCALATE,
            ActionType.DUMP_CREDS,
            ActionType.LATERAL_MOVE,
        ]

        print(f"\n[*] Simulating attack path ({len(attack_path)} steps)")
        print("-"*70)

        sim_result = env.simulate_attack_path(attack_path)

        print(f"\n  Total steps: {sim_result['total_steps']}")
        print(f"  Total reward: {sim_result['total_reward']:.2f}")
        print(f"  Reached goal: {sim_result['reached_goal']}")

        for step in sim_result["steps"]:
            print(f"    Step {step['step']}: {step['action']:20s} reward={step['reward']:.2f}")

        # Estimate realistic duration
        duration = env.estimate_realistic_duration(attack_path)
        print(f"\n  Estimated real duration: {duration:.0f}s ({duration/60:.1f} min)")

        self.report["dual_mode"] = {
            "simulation": sim_result,
            "estimated_duration": duration,
            "safety_status": env.get_safety_status(),
        }

        return self.report

    def _process_results(self, results) -> None:
        """Process and display attack results."""
        if results is None:
            print("\n[!] No results returned")
            return

        print("\n" + "="*70)
        print("  ATTACK RESULTS")
        print("="*70)

        if isinstance(results, dict):
            total_reward = results.get("total_reward", 0)
            steps_taken = results.get("steps_taken", 0)
            goal_achieved = results.get("goal_achieved", False)

            print(f"  Total reward: {total_reward:.2f}")
            print(f"  Steps taken: {steps_taken}")
            print(f"  Goal achieved: {goal_achieved}")

            # Show action history
            if "actions" in results:
                print(f"\n  Action history ({len(results['actions'])} actions):")
                for i, action in enumerate(results["actions"][-10:]):
                    print(f"    {i+1:3d}. {action}")

            self.report["results"] = results
        else:
            print(f"  Results: {results}")
            self.report["results"] = str(results)

    def print_report(self) -> None:
        """Print final report."""
        self.report["end_time"] = datetime.now().isoformat()

        print("\n" + "="*70)
        print("  FINAL REPORT")
        print("="*70)
        print(json.dumps(self.report, indent=2, default=str))
        print("="*70)

    def save_report(self, path: str) -> None:
        """Save report to file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.report, f, indent=2, default=str)
        print(f"\nReport saved to {path}")


async def real_target_scan(target_ip: str) -> Dict[str, Any]:
    """
    Perform real scan on a target IP (requires network access).
    """
    print("\n" + "="*70)
    print(f"  REAL TARGET SCAN: {target_ip}")
    print("="*70)

    toolkit = WindowsToolkit(timeout=5.0)

    # Scan common ports
    print("\n[*] Scanning common ports...")
    results = await toolkit.scan(target_ip)

    print(f"\n  Open ports: {results['open_count']}")
    for port, info in results["open_ports"].items():
        print(f"    {port:5d} | {info['service']:15s} | {info.get('banner', '')[:50]}")

    if results["open_count"] > 0:
        # Enumerate services
        print("\n[*] Enumerating services...")
        for port in results["open_ports"]:
            sv = await toolkit.nmap.scan_service_version(target_ip, port)
            print(f"  Port {port}: {sv}")

    return {
        "target": target_ip,
        "scan_results": results,
        "timestamp": datetime.now().isoformat(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Simulated and real attack testing"
    )
    parser.add_argument(
        "--target", "-t",
        choices=list(VULN_TARGETS.keys()),
        default="metasploitable2",
        help="Simulated target type"
    )
    parser.add_argument(
        "--real-target",
        help="Real target IP for scanning (requires network access)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Maximum attack steps"
    )
    parser.add_argument(
        "--mode",
        choices=["simulation", "dual-mode", "real-scan"],
        default="simulation",
        help="Test mode"
    )
    parser.add_argument(
        "--output", "-o",
        default="data/simulated_attack_report.json",
        help="Output report path"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )

    args = parser.parse_args()

    if args.mode == "real-scan" or args.real_target:
        # Real target scan
        target = args.real_target
        if not target:
            print("Error: --real-target required for real-scan mode")
            sys.exit(1)
        result = asyncio.run(real_target_scan(target))
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nReport saved to {args.output}")
        return

    # Simulation modes
    tester = SimulatedAttackTest(
        target_key=args.target,
        max_steps=args.steps,
        verbose=not args.quiet,
    )

    if args.mode == "simulation":
        tester.run_simulation()
    elif args.mode == "dual-mode":
        tester.run_dual_mode_test()

    tester.print_report()
    tester.save_report(args.output)


if __name__ == "__main__":
    main()
