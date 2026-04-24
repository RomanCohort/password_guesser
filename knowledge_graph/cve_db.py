"""
CVE/NVD Database Interface

Provides access to CVE (Common Vulnerabilities and Exposures) data
from the NVD (National Vulnerability Database) API.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
import requests
import time
import json
import logging
import os

logger = logging.getLogger(__name__)


@dataclass
class CVEEntry:
    """Represents a CVE vulnerability entry."""
    cve_id: str
    description: str
    cvss_score: float
    cvss_vector: str
    affected_products: List[str] = field(default_factory=list)
    cwe_ids: List[str] = field(default_factory=list)
    exploit_available: bool = False
    references: List[str] = field(default_factory=list)
    published_date: str = ""
    modified_date: str = ""

    @property
    def severity(self) -> str:
        if self.cvss_score >= 9.0:
            return "critical"
        elif self.cvss_score >= 7.0:
            return "high"
        elif self.cvss_score >= 4.0:
            return "medium"
        elif self.cvss_score > 0.0:
            return "low"
        return "none"

    def to_dict(self) -> dict:
        return {
            "cve_id": self.cve_id,
            "description": self.description,
            "cvss_score": self.cvss_score,
            "cvss_vector": self.cvss_vector,
            "affected_products": self.affected_products,
            "cwe_ids": self.cwe_ids,
            "exploit_available": self.exploit_available,
            "references": self.references,
            "published_date": self.published_date,
            "modified_date": self.modified_date,
            "severity": self.severity,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CVEEntry":
        return cls(
            cve_id=data["cve_id"],
            description=data.get("description", ""),
            cvss_score=data.get("cvss_score", 0.0),
            cvss_vector=data.get("cvss_vector", ""),
            affected_products=data.get("affected_products", []),
            cwe_ids=data.get("cwe_ids", []),
            exploit_available=data.get("exploit_available", False),
            references=data.get("references", []),
            published_date=data.get("published_date", ""),
            modified_date=data.get("modified_date", ""),
        )


class CVEDatabase:
    """
    CVE database interface using NVD API.

    Features:
    - Online lookup via NVD REST API
    - Local caching to avoid rate limiting
    - Offline cache file support
    """

    def __init__(
        self,
        api_base: str = "https://services.nvd.nist.gov/rest/json/cves/2.0",
        cache_path: Optional[str] = None,
    ):
        self.api_base = api_base
        self.cache: Dict[str, CVEEntry] = {}
        self._session = requests.Session()
        self._last_request_time: float = 0
        self._min_request_interval: float = 0.6  # NVD rate limit

        if cache_path and os.path.exists(cache_path):
            self._load_cache(cache_path)

    def _load_cache(self, path: str) -> None:
        """Load cached CVE entries from JSON file."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for cve_data in data:
                entry = CVEEntry.from_dict(cve_data)
                self.cache[entry.cve_id] = entry
            logger.info(f"Loaded {len(self.cache)} CVE entries from cache")
        except Exception as e:
            logger.warning(f"Failed to load CVE cache: {e}")

    def save_cache(self, path: str) -> None:
        """Save cached CVE entries to JSON file."""
        data = [entry.to_dict() for entry in self.cache.values()]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(data)} CVE entries to cache")

    def _rate_limit(self) -> None:
        """Enforce NVD API rate limits."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    def get_cve(self, cve_id: str) -> Optional[CVEEntry]:
        """Get a CVE entry by ID, checking cache first."""
        if cve_id in self.cache:
            return self.cache[cve_id]

        return self._fetch_cve(cve_id)

    def _fetch_cve(self, cve_id: str) -> Optional[CVEEntry]:
        """Fetch a CVE entry from NVD API."""
        self._rate_limit()

        try:
            params = {"cveId": cve_id}
            response = self._session.get(
                self.api_base,
                params=params,
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            vulnerabilities = data.get("vulnerabilities", [])
            if not vulnerabilities:
                return None

            cve_data = vulnerabilities[0].get("cve", {})
            return self._parse_nvd_entry(cve_data)

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch CVE {cve_id}: {e}")
            return None

    def _parse_nvd_entry(self, cve_data: dict) -> CVEEntry:
        """Parse NVD API response into CVEEntry."""
        cve_id = cve_data.get("id", "")

        # Description
        descriptions = cve_data.get("descriptions", [])
        description = ""
        for desc in descriptions:
            if desc.get("lang") == "en":
                description = desc.get("value", "")
                break

        # CVSS
        cvss_score = 0.0
        cvss_vector = ""
        metrics = cve_data.get("metrics", {})
        for version in ["cvssMetricV31", "cvssMetricV30", "cvssMetricV2"]:
            if version in metrics and metrics[version]:
                cvss_data = metrics[version][0].get("cvssData", {})
                cvss_score = cvss_data.get("baseScore", 0.0)
                cvss_vector = cvss_data.get("vectorString", "")
                break

        # Affected products (CPE)
        affected_products: List[str] = []
        configurations = cve_data.get("configurations", {})
        for config in configurations.get("nodes", []):
            for cpe_match in config.get("cpeMatch", []):
                cpe_uri = cpe_match.get("criteria", "")
                if cpe_uri:
                    affected_products.append(cpe_uri)

        # CWE
        cwe_ids: List[str] = []
        weaknesses = cve_data.get("weaknesses", [])
        for weakness in weaknesses:
            for desc in weakness.get("description", []):
                cwe = desc.get("value", "")
                if cwe and cwe.startswith("CWE-"):
                    cwe_ids.append(cwe)

        # References
        references = [
            ref.get("url", "")
            for ref in cve_data.get("references", [])
        ]

        entry = CVEEntry(
            cve_id=cve_id,
            description=description,
            cvss_score=cvss_score,
            cvss_vector=cvss_vector,
            affected_products=affected_products,
            cwe_ids=cwe_ids,
            references=references,
            published_date=cve_data.get("published", ""),
            modified_date=cve_data.get("lastModified", ""),
        )

        self.cache[cve_id] = entry
        return entry

    def search_by_product(self, product: str) -> List[CVEEntry]:
        """Search CVEs affecting a specific product."""
        self._rate_limit()

        try:
            params = {"keywordSearch": product}
            response = self._session.get(
                self.api_base,
                params=params,
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            results = []
            for vuln in data.get("vulnerabilities", []):
                entry = self._parse_nvd_entry(vuln.get("cve", {}))
                if entry:
                    results.append(entry)

            return results

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to search CVEs for {product}: {e}")
            return []

    def search_by_cvss_range(
        self,
        min_score: float = 0.0,
        max_score: float = 10.0,
        product: Optional[str] = None,
    ) -> List[CVEEntry]:
        """Search CVEs within a CVSS score range."""
        results = []

        if product:
            all_cves = self.search_by_product(product)
        else:
            all_cves = list(self.cache.values())

        for cve in all_cves:
            if min_score <= cve.cvss_score <= max_score:
                results.append(cve)

        return sorted(results, key=lambda x: x.cvss_score, reverse=True)

    def get_exploitable_cves(self) -> List[CVEEntry]:
        """Get cached CVEs that have known exploits."""
        return [
            cve for cve in self.cache.values()
            if cve.exploit_available or cve.cvss_score >= 7.0
        ]

    def preload_common_cves(self, limit: int = 1000) -> None:
        """
        Preload recent high-severity CVEs into cache.

        Note: This uses NVD API and is subject to rate limits.
        """
        logger.info(f"Preloading up to {limit} recent CVEs...")

        try:
            params = {
                "resultsPerPage": min(limit, 100),
                "cvssV3Severity": "HIGH",
            }
            response = self._session.get(
                self.api_base,
                params=params,
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            count = 0
            for vuln in data.get("vulnerabilities", []):
                entry = self._parse_nvd_entry(vuln.get("cve", {}))
                if entry:
                    count += 1

            logger.info(f"Preloaded {count} CVE entries")

        except Exception as e:
            logger.error(f"Failed to preload CVEs: {e}")

    def batch_get(self, cve_ids: List[str]) -> Dict[str, CVEEntry]:
        """Get multiple CVE entries, using cache where possible."""
        results = {}
        for cve_id in cve_ids:
            entry = self.get_cve(cve_id)
            if entry:
                results[cve_id] = entry
        return results
