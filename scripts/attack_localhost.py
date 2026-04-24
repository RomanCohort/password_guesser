"""
Local Machine Attack Test

Comprehensive penetration test against the local machine.
Discovers services, enumerates details, and tests for vulnerabilities.

WARNING: Only run on machines you own or have authorization to test.
"""

import asyncio
import json
import sys
import time
import socket
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from pentest.windows_compat import WindowsToolkit, PythonNmap
from pentest.network import ConnectionPool, TCPConnector, HTTPConnector
from pentest.evasion import EvasionManager, EvasionConfig, EvasionLevel


# Colors for terminal output
class C:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'


def banner(text, color=C.CYAN):
    print(f"\n{color}{C.BOLD}{'='*60}\n  {text}\n{'='*60}{C.END}")


def info(msg):
    print(f"  {C.BLUE}[*]{C.END} {msg}")


def success(msg):
    print(f"  {C.GREEN}[+]{C.END} {msg}")


def warn(msg):
    print(f"  {C.YELLOW}[!]{C.END} {msg}")


def fail(msg):
    print(f"  {C.RED}[-]{C.END} {msg}")


def vuln(msg):
    print(f"  {C.RED}{C.BOLD}[VULN]{C.END} {msg}")


async def phase1_wide_scan(target: str) -> Dict[str, Any]:
    """Phase 1: Wide port scan to discover all open ports."""
    banner("PHASE 1: Wide Port Scan")

    toolkit = WindowsToolkit(timeout=3.0, max_concurrent=100)

    # Scan top 1000 + common service ports
    common = [
        21, 22, 23, 25, 53, 80, 110, 111, 135, 139, 143, 161, 389, 443, 445,
        465, 514, 587, 636, 993, 995, 1080, 1433, 1521, 2049, 3306, 3389,
        5432, 5900, 5985, 5986, 6379, 8080, 8443, 8888, 9090, 9200, 27017,
    ]
    # Add dynamic/service ports
    extended = list(range(49152, 49200)) + list(range(8000, 8100))
    all_ports = sorted(set(common + extended))

    info(f"Scanning {len(all_ports)} ports on {target}...")
    start = time.time()

    results = await toolkit.nmap.scan_ports(target, all_ports)

    open_ports = [r for r in results if r.status == "open"]
    closed = sum(1 for r in results if r.status == "closed")
    filtered = sum(1 for r in results if r.status == "filtered")

    elapsed = time.time() - start

    print(f"\n  Scan completed in {elapsed:.1f}s")
    print(f"  {C.GREEN}{len(open_ports)} open{C.END} | {closed} closed | {filtered} filtered\n")

    if open_ports:
        print(f"  {'PORT':>8s} | {'STATUS':10s} | {'SERVICE':15s} | BANNER")
        print(f"  {'-'*8} | {'-'*10} | {'-'*15} | {'-'*40}")
        for r in sorted(open_ports, key=lambda x: x.port):
            print(f"  {r.port:>8d} | {r.status:10s} | {r.service:15s} | {r.banner[:40]}")

    return {"open_ports": open_ports, "scan_time": elapsed}


async def phase2_service_enum(target: str, open_ports: List) -> Dict[str, Any]:
    """Phase 2: Service enumeration and banner grabbing."""
    banner("PHASE 2: Service Enumeration")

    toolkit = WindowsToolkit(timeout=5.0)
    findings = {}

    for port_result in open_ports:
        port = port_result.port
        service = port_result.service

        info(f"Enumerating port {port} ({service})...")

        # Service version detection
        sv = await toolkit.nmap.scan_service_version(target, port)
        findings[port] = sv

        if sv.get("banner"):
            success(f"Port {port} banner: {sv['banner'][:80]}")
        if sv.get("version_info"):
            for k, v in sv["version_info"].items():
                if v and v != "unknown":
                    success(f"Port {port} {k}: {v}")

        # HTTP-specific checks
        if port in (80, 443, 8080, 8443):
            await check_http(target, port, findings)

        # SMB check
        if port in (139, 445):
            await check_smb(target, port, findings)

        # SSH check
        if port == 22:
            await check_ssh(target, port, findings)

    return findings


async def check_http(target: str, port: int, findings: Dict):
    """Check HTTP service details."""
    info(f"  HTTP check on port {port}...")
    try:
        import httpx
        protocol = "https" if port in (443, 8443) else "http"
        url = f"{protocol}://{target}:{port}/"

        async with httpx.AsyncClient(timeout=5.0, verify=False, follow_redirects=True) as client:
            resp = await client.get(url)

            server = resp.headers.get("server", "")
            powered = resp.headers.get("x-powered-by", "")

            if server:
                success(f"  Server header: {server}")
            if powered:
                success(f"  X-Powered-By: {powered}")
                warn(f"  Info leakage: X-Powered-By exposes technology stack")

            if resp.status_code == 200:
                success(f"  HTTP {resp.status_code} - Page served")

                # Check for common paths
                for path in ["/robots.txt", "/.env", "/admin", "/wp-admin",
                             "/server-status", "/.git/config"]:
                    try:
                        check = await client.get(f"{protocol}://{target}:{port}{path}")
                        if check.status_code == 200:
                            if path in ["/.env", "/.git/config"]:
                                vuln(f"  SENSITIVE FILE EXPOSED: {path} (200 OK)")
                            else:
                                success(f"  Found: {path} ({check.status_code})")
                    except Exception:
                        pass
            else:
                info(f"  HTTP {resp.status_code}")

    except Exception as e:
        info(f"  HTTP check failed: {e}")


async def check_smb(target: str, port: int, findings: Dict):
    """Check SMB service."""
    info(f"  SMB check on port {port}...")
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(target, 445),
            timeout=5.0
        )
        # SMB negotiate protocol
        # Minimal SMBv1 negotiate request
        smb_negotiate = bytes([
            0x00, 0x00, 0x00, 0x85,  # NetBIOS session
            0xFF, 0x53, 0x4D, 0x42,  # SMB magic
            0x72,                     # Negotiate
            0x00, 0x00, 0x00, 0x00, 0x18, 0x53, 0xC8,
            0x00, 0x00,
        ])
        writer.write(smb_negotiate)
        await writer.drain()

        try:
            resp = await asyncio.wait_for(reader.read(1024), timeout=3.0)
            if resp and len(resp) > 4:
                # Check SMB dialect
                if b'SMB' in resp:
                    success(f"  SMB responding - dialect negotiation supported")
                    if b'\x00\x02' in resp[:10]:
                        warn(f"  SMBv1 enabled - vulnerable to multiple attacks")
                    findings[port] = {"smb_responding": True, "raw_size": len(resp)}
        except asyncio.TimeoutError:
            pass

        writer.close()
        await writer.wait_closed()

    except Exception as e:
        info(f"  SMB check: {e}")


async def check_ssh(target: str, port: int, findings: Dict):
    """Check SSH service and get banner."""
    info(f"  SSH check on port {port}...")
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(target, port),
            timeout=5.0
        )
        banner = await asyncio.wait_for(reader.read(1024), timeout=3.0)
        banner_str = banner.decode('utf-8', errors='ignore').strip()

        if banner_str:
            success(f"  SSH banner: {banner_str}")
            if "OpenSSH" in banner_str:
                # Extract version
                import re
                version = re.search(r'OpenSSH_([\d.]+)', banner_str)
                if version:
                    ver = version.group(1)
                    success(f"  OpenSSH version: {ver}")

        writer.close()
        await writer.wait_closed()
    except Exception as e:
        info(f"  SSH check: {e}")


async def phase3_vuln_check(target: str, open_ports: List, findings: Dict) -> List[Dict]:
    """Phase 3: Vulnerability checks."""
    banner("PHASE 3: Vulnerability Assessment")

    vulnerabilities = []
    services = {r.port: r.service for r in open_ports}

    # Check each service for known issues
    for port, service in services.items():
        if service in ("microsoft-ds", "smb") or port == 445:
            # SMB vulnerability checks
            info("Checking SMB vulnerabilities...")
            # Check for null session
            try:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(target, 445), timeout=5.0
                )
                writer.close()
                await writer.wait_closed()
                warn("  SMB port accessible - check for MS17-010 (EternalBlue)")
                vulnerabilities.append({
                    "port": 445,
                    "service": "smb",
                    "vuln": "Potential MS17-010 / SMB vulnerabilities",
                    "severity": "high",
                    "detail": "SMB port is open and accessible",
                })
            except Exception:
                pass

        if service in ("msrpc",) or port == 135:
            info("Checking MS-RPC endpoint...")
            warn("  MS-RPC endpoint mapper accessible")
            vulnerabilities.append({
                "port": 135,
                "service": "msrpc",
                "vuln": "MS-RPC endpoint exposed",
                "severity": "medium",
                "detail": "RPC endpoint mapper can enumerate services",
            })

        if service in ("http", "https") or port in (80, 443, 8080):
            info(f"Checking web vulnerabilities on port {port}...")
            # Check for common HTTP issues
            try:
                import httpx
                protocol = "https" if port in (443, 8443) else "http"
                async with httpx.AsyncClient(timeout=5.0, verify=False) as client:
                    resp = await client.get(f"{protocol}://{target}:{port}/")

                    headers = resp.headers

                    # Missing security headers
                    missing_headers = []
                    security_headers = [
                        "X-Frame-Options", "X-Content-Type-Options",
                        "X-XSS-Protection", "Content-Security-Policy",
                        "Strict-Transport-Security"
                    ]
                    for h in security_headers:
                        if h not in headers:
                            missing_headers.append(h)

                    if missing_headers:
                        warn(f"  Missing security headers: {', '.join(missing_headers)}")
                        vulnerabilities.append({
                            "port": port,
                            "service": service,
                            "vuln": f"Missing security headers",
                            "severity": "low",
                            "detail": f"Missing: {', '.join(missing_headers[:3])}",
                        })

                    # Server version disclosure
                    server = headers.get("server", "")
                    if server and len(server) > 3:
                        warn(f"  Server version disclosed: {server}")
            except Exception:
                pass

    # Print vulnerability summary
    print(f"\n  {C.BOLD}Vulnerability Summary:{C.END}")
    print(f"  {'-'*55}")

    if not vulnerabilities:
        success("No obvious vulnerabilities found in basic checks")
    else:
        sev_colors = {"high": C.RED, "medium": C.YELLOW, "low": C.BLUE}
        for v in vulnerabilities:
            color = sev_colors.get(v["severity"], "")
            print(f"  {color}[{v['severity'].upper():6s}]{C.END} Port {v['port']:5d} ({v['service']:10s}) - {v['vuln']}")

    return vulnerabilities


async def phase4_brute_test(target: str, open_ports: List) -> Dict:
    """Phase 4: Test common credentials (safe - no actual exploitation)."""
    banner("PHASE 4: Credential Testing")

    services = {r.port: r.service for r in open_ports}
    results = {}

    # Test common usernames/passwords against SSH if available
    if 22 in services:
        info("SSH credential testing (top 5 common passwords)...")
        toolkit = WindowsToolkit(timeout=5.0)
        common_creds = [
            ("root", "root"), ("admin", "admin"), ("root", "toor"),
            ("admin", "password"), ("user", "user"),
        ]

        for username, password in common_creds:
            info(f"  Testing {username}:{password}...")
            # This will actually try to connect via paramiko if available
            # otherwise it falls back to a port check
            result = await toolkit.hydra._try_ssh_login(target, 22, username, password)
            if result:
                vuln(f"  CREDENTIAL FOUND: {username}:{password}")
                results[f"ssh_{username}"] = {"found": True, "password": password}
            else:
                info(f"  {username}:{password} - failed")

        if not results:
            info("No SSH credentials found (paramiko may not be installed)")

    # HTTP auth check
    for port in [80, 443, 8080]:
        if port in services:
            info(f"HTTP auth testing on port {port}...")
            toolkit = WindowsToolkit(timeout=5.0)
            common_creds = [
                ("admin", "admin"), ("admin", "password"),
                ("admin", "admin123"), ("root", "root"),
            ]

            protocol = "https" if port in (443, 8443) else "http"
            url = f"{protocol}://{target}:{port}/"

            for username, password in common_creds:
                result = await toolkit.hydra._try_http_login(url, username, password)
                if result:
                    success(f"  HTTP auth possible: {username}:{password}")
                    results[f"http_{port}_{username}"] = {"found": True, "password": password}

            if not any(k.startswith(f"http_{port}") for k in results):
                info(f"  No HTTP auth found on port {port}")

    # Windows local check
    if 445 in services or 135 in services:
        info("Windows local account check...")
        try:
            import subprocess
            # List local users (Windows)
            proc = await asyncio.create_subprocess_exec(
                "net", "user",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            if stdout:
                users = stdout.decode('utf-8', errors='ignore')
                info(f"  Local users found:\n{users}")
                results["windows_users"] = users
        except Exception:
            info("  Could not enumerate local users")

    return results


async def phase5_network_info() -> Dict:
    """Phase 5: Gather local network information."""
    banner("PHASE 5: Network Reconnaissance")

    info_map = {}

    # Get hostname
    hostname = socket.gethostname()
    success(f"Hostname: {hostname}")
    info_map["hostname"] = hostname

    # Get local IP
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        success(f"Local IP: {local_ip}")
        info_map["local_ip"] = local_ip
    except Exception:
        local_ip = "127.0.0.1"

    # Get all interface addresses
    try:
        addrs = socket.getaddrinfo(hostname, None)
        ips = list(set(addr[4][0] for addr in addrs if addr[0] == socket.AF_INET))
        success(f"Interfaces: {', '.join(ips)}")
        info_map["interfaces"] = ips
    except Exception:
        pass

    # Check active connections
    try:
        proc = await asyncio.create_subprocess_exec(
            "netstat", "-an",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
        if stdout:
            lines = stdout.decode('utf-8', errors='ignore').split('\n')
            listening = [l.strip() for l in lines if 'LISTENING' in l]
            info(f"Listening ports ({len(listening)} total):")
            for line in listening[:15]:
                print(f"    {line}")
            if len(listening) > 15:
                print(f"    ... and {len(listening)-15} more")
            info_map["listening_ports"] = len(listening)
    except Exception:
        info("  netstat not available")

    return info_map


async def run_full_attack():
    """Execute full local attack test."""
    target = "localhost"

    print(f"\n{C.RED}{C.BOLD}")
    print("  ╔══════════════════════════════════════════════════════════╗")
    print("  ║     LOCAL MACHINE PENETRATION TEST                      ║")
    print("  ║     Target: localhost (this machine)                    ║")
    print("  ║     Started: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "                          ║")
    print("  ╚══════════════════════════════════════════════════════════╝")
    print(f"{C.END}")

    report = {"target": target, "start_time": datetime.now().isoformat()}

    # Phase 1: Wide scan
    scan_result = await phase1_wide_scan(target)
    report["scan"] = {
        "open_count": len(scan_result["open_ports"]),
        "open_ports": [r.port for r in scan_result["open_ports"]],
        "scan_time": scan_result["scan_time"],
    }

    if not scan_result["open_ports"]:
        warn("No open ports found. Exiting.")
        return report

    # Phase 2: Service enumeration
    findings = await phase2_service_enum(target, scan_result["open_ports"])
    report["services"] = {str(k): v for k, v in findings.items()}

    # Phase 3: Vulnerability check
    vulns = await phase3_vuln_check(target, scan_result["open_ports"], findings)
    report["vulnerabilities"] = vulns

    # Phase 4: Credential testing
    creds = await phase4_brute_test(target, scan_result["open_ports"])
    report["credentials"] = creds

    # Phase 5: Network info
    net_info = await phase5_network_info()
    report["network"] = net_info

    # Final summary
    banner("ATTACK SUMMARY", C.RED)
    print(f"  Target:              {target}")
    print(f"  Open ports:          {report['scan']['open_count']}")
    print(f"  Vulnerabilities:     {len(vulns)}")
    print(f"  Credentials found:   {sum(1 for v in creds.values() if isinstance(v, dict) and v.get('found'))}")
    print(f"  Scan duration:       {report['scan']['scan_time']:.1f}s")
    print(f"  Completed:           {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    report["end_time"] = datetime.now().isoformat()

    # Save report
    report_path = f"data/local_attack_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    Path("data").mkdir(exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report saved: {report_path}")
    print(f"{'='*60}")

    return report


if __name__ == "__main__":
    asyncio.run(run_full_attack())
