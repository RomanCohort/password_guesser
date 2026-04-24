"""
Local Network Scan Test

Tests the Windows-native Python scanning tools against local network targets.
This verifies the toolkit works correctly before attempting real attacks.

Usage:
    python scripts/local_scan_test.py
    python scripts/local_scan_test.py --target localhost
    python scripts/local_scan_test.py --target 127.0.0.1 --full-scan
"""

import asyncio
import argparse
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from pentest.windows_compat import WindowsToolkit, PythonNmap
from pentest.network import ConnectionPool, NetworkScanner
from pentest.evasion import EvasionManager, EvasionConfig, EvasionLevel


async def test_python_nmap(toolkit: WindowsToolkit, target: str, ports):
    """Test Python-native nmap scanner."""
    print("\n[1] Testing PythonNmap (port scanner)")
    print("-" * 40)

    results = await toolkit.nmap.scan_ports(target, ports)

    print(f"Scanned {len(results)} ports on {target}")
    open_count = sum(1 for r in results if r.status == "open")
    print(f"Open: {open_count}, Closed/Filtered: {len(results) - open_count}")

    for r in results:
        if r.status == "open":
            print(f"  PORT {r.port:5d}: {r.service:12s} | Banner: {r.banner[:40]}")

    return results


async def test_network_scanner(pool: NetworkScanner, target: str):
    """Test NetworkScanner high-level interface."""
    print("\n[2] Testing NetworkScanner (high-level)")
    print("-" * 40)

    results = await pool.quick_scan(target, common_ports=True)

    print(f"Host: {results['host']}")
    print(f"Open ports: {list(results['open_ports'].keys())}")
    print(f"Scan duration: {results['scan_duration']:.2f}s")

    return results


async def test_evasion(evasion_level: EvasionLevel):
    """Test evasion manager."""
    print(f"\n[3] Testing EvasionManager (level: {evasion_level.name})")
    print("-" * 40)

    evasion = EvasionManager(EvasionConfig(level=evasion_level))

    # Test delays
    delays = []
    for action in ["scan", "exploit", "brute_force"]:
        import time
        start = time.time()
        await evasion.pre_action_delay(action)
        delay = time.time() - start
        delays.append((action, delay))
        print(f"  {action:12s}: {delay:.3f}s delay")

    # Test headers
    headers = evasion.get_headers()
    print(f"\n  User-Agent: {headers.get('User-Agent', 'N/A')[:60]}...")

    return evasion


async def test_service_version(toolkit: WindowsToolkit, target: str, port: int):
    """Test service version detection."""
    print(f"\n[4] Testing Service Version Detection (port {port})")
    print("-" * 40)

    result = await toolkit.nmap.scan_service_version(target, port)

    print(f"Status: {result.get('status', 'unknown')}")
    if "service" in result:
        print(f"Service: {result.get('service')}")
    if "banner" in result:
        print(f"Banner: {result.get('banner', '')[:80]}")
    if "version_info" in result:
        print(f"Version info: {result.get('version_info')}")

    return result


async def test_tcp_connector(pool: ConnectionPool, target: str, port: int):
    """Test low-level TCP connector."""
    print(f"\n[5] Testing TCPConnector (direct connect)")
    print("-" * 40)

    result = await pool._tcp.connect(target, port)

    print(f"Protocol: {result.protocol}")
    print(f"Host: {result.host}:{result.port}")
    print(f"Success: {result.success}")
    print(f"Duration: {result.duration:.3f}s")
    if result.banner:
        print(f"Banner: {result.banner[:80]}")

    return result


async def test_platform_check(toolkit: WindowsToolkit):
    """Test platform capability check."""
    print("\n[6] Testing Platform Capability Check")
    print("-" * 40)

    info = toolkit.check_platform()

    print(f"Platform: {info['platform']}")
    print(f"Python: {info['python_version']}")
    print("\nAvailable tools:")
    for tool, available in info['tools'].items():
        status = "OK" if available else "MISSING"
        print(f"  {tool:20s}: {status}")

    if info.get('recommendation'):
        print(f"\nRecommendation: {info['recommendation']}")

    return info


async def run_local_tests(target: str = "localhost", ports = None, full_scan: bool = False):
    """Run all local tests."""
    print("="*60)
    print("LOCAL SCAN TEST SUITE")
    print("="*60)
    print(f"Target: {target}")
    print(f"Mode: {'Full scan (1000 ports)' if full_scan else 'Quick scan'}")
    print(f"Started: {datetime.now().isoformat()}")
    print("="*60)

    # Default ports
    if ports is None:
        if full_scan:
            ports = list(range(1, 1001))
        else:
            ports = [21, 22, 23, 25, 53, 80, 110, 135, 139, 143, 443, 445,
                     993, 995, 3306, 3389, 5432, 5900, 8080, 8443]

    # Initialize tools
    toolkit = WindowsToolkit(timeout=5.0, max_concurrent=50)
    pool = NetworkScanner()

    results = {}

    # Run tests
    try:
        results["nmap"] = await test_python_nmap(toolkit, target, ports)

        results["network_scanner"] = await test_network_scanner(pool, target)

        results["evasion"] = await test_evasion(EvasionLevel.LOW)

        results["platform"] = await test_platform_check(toolkit)

        # Test service version if we found open ports
        if results["nmap"]:
            open_ports = [r.port for r in results["nmap"] if r.status == "open"]
            if open_ports:
                port = open_ports[0]
                results["service_version"] = await test_service_version(toolkit, target, port)

        # Test TCP connector
        if open_ports:
            results["tcp_connector"] = await test_tcp_connector(NetworkScanner().pool, target, open_ports[0])

    except Exception as e:
        print(f"\n[!] Test error: {e}")
        import traceback
        traceback.print_exc()
        results["error"] = str(e)

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    if "nmap" in results:
        open_ports = [r.port for r in results["nmap"] if r.status == "open"]
        print(f"Open ports found: {len(open_ports)}")
        if open_ports:
            print(f"Ports: {open_ports}")

    if "evasion" in results:
        stats = results["evasion"].get_stats()
        print(f"Evasion level: {stats['level']}")
        print(f"Actions performed: {stats['actions_performed']}")

    print(f"Completed: {datetime.now().isoformat()}")
    print("="*60)

    return results


async def main():
    parser = argparse.ArgumentParser(
        description="Test Windows-native penetration tools locally"
    )
    parser.add_argument(
        "--target", "-t",
        default="localhost",
        help="Target (default: localhost)"
    )
    parser.add_argument(
        "--ports", "-p",
        help="Ports to scan (comma-separated)"
    )
    parser.add_argument(
        "--full-scan",
        action="store_true",
        help="Scan all 1000 ports"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        default=True,
        help="Quick scan (default)"
    )

    args = parser.parse_args()

    # Parse ports
    ports = None
    if args.ports:
        ports = [int(p.strip()) for p in args.ports.split(",")]

    await run_local_tests(
        target=args.target,
        ports=ports,
        full_scan=args.full_scan
    )


if __name__ == "__main__":
    asyncio.run(main())
