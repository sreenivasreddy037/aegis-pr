"""Hybrid (dense + sparse) retrieval over repo ADRs and built-in security policies.

* Dense vectors: fastembed (`BAAI/bge-small-en-v1.5`) when available, otherwise a
  deterministic feature-hashing embedder so CI and `MOCK_LLM=true` never hit the network.
* Sparse vectors: hashed term frequencies with Qdrant's server-side IDF modifier (BM25-style).
* Fusion: Qdrant `query_points` with two prefetches fused by Reciprocal Rank Fusion.

Runs against an in-process Qdrant (`QDRANT_MODE=local`) or a Qdrant server (`remote`).
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import uuid
from collections import Counter
from pathlib import Path
from typing import Protocol

from qdrant_client import QdrantClient, models

from app.config import Settings, get_settings
from app.schemas import PolicySnippet

logger = logging.getLogger(__name__)

DENSE = "dense"
SPARSE = "sparse"
HASH_DIM = 384
SPARSE_BUCKETS = 2**18
_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "for",
        "is",
        "are",
        "be",
        "by",
        "with",
        "as",
        "at",
        "it",
        "this",
        "that",
        "from",
        "must",
        "any",
        "not",
    ]
)

BUILTIN_POLICIES: list[dict] = [
    {
        "source": "builtin:owasp-a03",
        "title": "OWASP A03:2021 Injection / CWE-89 SQL Injection",
        "tags": ["security", "CWE-89", "sql"],
        "text": "Never build SQL statements with f-strings, % formatting, str.format or string "
        "concatenation of variables. Use parameterised queries with placeholders (? or %s) and "
        "pass values separately to cursor.execute. Dynamic filters such as status, ORDER BY or "
        "LIMIT must be bound parameters or validated against an allow-list.",
    },
    {
        "source": "builtin:cwe-78",
        "title": "CWE-78 OS Command Injection",
        "tags": ["security", "CWE-78", "subprocess", "shell"],
        "text": "Do not call subprocess with shell=True or os.system on strings that contain "
        "interpolated input. Pass an argument list to subprocess.run without a shell and validate "
        "file names and bucket names.",
    },
    {
        "source": "builtin:cwe-95",
        "title": "CWE-95 Eval Injection",
        "tags": ["security", "CWE-95", "eval"],
        "text": "eval and exec on externally influenced strings allow arbitrary code execution. "
        "Use ast.literal_eval, a parser for a small expression grammar, or configuration data.",
    },
    {
        "source": "builtin:cwe-798",
        "title": "CWE-798 Hard-coded Credentials",
        "tags": ["security", "CWE-798", "secret"],
        "text": "API keys, signing secrets, passwords and tokens must never appear as literals in "
        "source code or as fallback defaults for environment variables. Fail closed when the "
        "secret is missing.",
    },
    {
        "source": "builtin:cwe-532",
        "title": "CWE-532 Sensitive Information in Log Files",
        "tags": ["security", "CWE-532", "logging"],
        "text": "Tokens, passwords, secrets and signatures must not be logged at any level.",
    },
    {
        "source": "builtin:cwe-369",
        "title": "CWE-369 Divide By Zero / CWE-476 NULL Pointer Dereference",
        "tags": ["regression", "CWE-369", "CWE-476", "division"],
        "text": "Arithmetic on caller-supplied counts, divisors or optional parameters must guard "
        "against zero and None. Validate inputs and raise a domain error such as ValueError "
        "instead of letting ZeroDivisionError or TypeError escape.",
    },
    {
        "source": "builtin:api-contract",
        "title": "Backward-compatible public function signatures",
        "tags": ["breaking_change", "contract", "signature"],
        "text": "Adding a required parameter, removing or renaming a parameter, reordering "
        "positional parameters, or making parameters keyword-only breaks existing callers. New "
        "parameters need defaults, or every caller must be updated in the same change.",
    },
]


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1]


def _bucket(token: str, buckets: int) -> int:
    return int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "big") % buckets


class HashingEmbedder:
    """Deterministic dense embedder: hashed unigrams + character trigrams, L2-normalised."""

    name = "hash"

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * HASH_DIM
            tokens = tokenize(text)
            feats = tokens + [t[i : i + 3] for t in tokens for i in range(max(1, len(t) - 2))]
            for f in feats:
                h = _bucket(f, 2 * HASH_DIM)
                vec[h % HASH_DIM] += 1.0 if h < HASH_DIM else -1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


class FastEmbedEmbedder:
    def __init__(self, model_name: str) -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name)
        self.name = f"fastembed:{model_name}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.embed(texts)]


def make_embedder(settings: Settings) -> Embedder:
    backend = settings.embedding_backend
    if backend == "auto":
        backend = "hash" if settings.offline else "fastembed"
    if backend == "fastembed":
        try:
            return FastEmbedEmbedder(settings.fastembed_model)
        except Exception as exc:  # model download blocked, missing ONNX runtime, ...
            logger.warning("fastembed unavailable (%s); falling back to hashing embedder", exc)
    return HashingEmbedder()


def sparse_vector(text: str) -> models.SparseVector:
    counts = Counter(_bucket(t, SPARSE_BUCKETS) for t in tokenize(text))
    indices = sorted(counts)
    return models.SparseVector(indices=indices, values=[float(counts[i]) for i in indices])


def chunk_markdown(text: str, source: str) -> list[dict]:
    """Split a markdown document on `##` headings into retrievable chunks."""
    chunks: list[dict] = []
    title, buf = source, []
    for line in text.splitlines():
        if line.startswith("## "):
            if "".join(buf).strip():
                chunks.append({"source": source, "title": title, "text": "\n".join(buf).strip()})
            title, buf = line[3:].strip(), []
        elif not line.startswith("# "):
            buf.append(line)
    if "".join(buf).strip():
        chunks.append({"source": source, "title": title, "text": "\n".join(buf).strip()})
    for c in chunks:
        c["tags"] = ["adr"] if "adr" in c["title"].lower() else ["docs"]
    return chunks


def discover_repo_docs(repo: Path) -> list[dict]:
    docs: list[dict] = []
    candidates = [*repo.glob("*.md"), *repo.glob("docs/**/*.md"), *repo.glob("adr/**/*.md")]
    for path in sorted(set(candidates)):
        if path.name.lower() in {"readme.md", "changelog.md"}:
            continue
        docs.extend(
            chunk_markdown(path.read_text(errors="replace"), path.relative_to(repo).as_posix())
        )
    return docs


class PolicyStore:
    def __init__(
        self,
        settings: Settings | None = None,
        embedder: Embedder | None = None,
        client: QdrantClient | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.embedder = embedder or make_embedder(self.settings)
        if client is not None:
            self.client = client
        elif self.settings.qdrant_mode == "remote":
            self.client = QdrantClient(url=self.settings.qdrant_url, timeout=10)
        else:
            self.client = QdrantClient(location=":memory:")
        # Per-index collection name avoids cross-talk between concurrent reviews on a server.
        self.collection = f"{self.settings.qdrant_collection}_{uuid.uuid4().hex[:8]}"
        self._dim: int | None = None

    def _ensure_collection(self, dim: int) -> None:
        if self._dim == dim:
            return
        self.client.create_collection(
            self.collection,
            vectors_config={DENSE: models.VectorParams(size=dim, distance=models.Distance.COSINE)},
            sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        self._dim = dim

    def index(self, docs: list[dict]) -> int:
        if not docs:
            return 0
        texts = [f"{d['title']}\n{d['text']}" for d in docs]
        dense = self.embedder.embed(texts)
        self._ensure_collection(len(dense[0]))
        self.client.upsert(
            self.collection,
            points=[
                models.PointStruct(
                    id=i,
                    vector={DENSE: dense[i], SPARSE: sparse_vector(texts[i])},
                    payload=docs[i],
                )
                for i in range(len(docs))
            ],
        )
        return len(docs)

    def index_repo(self, repo: Path) -> int:
        return self.index(BUILTIN_POLICIES + discover_repo_docs(repo))

    def search(self, query: str, top_k: int | None = None) -> list[PolicySnippet]:
        if self._dim is None or not query.strip():
            return []
        k = top_k or self.settings.policy_top_k
        result = self.client.query_points(
            self.collection,
            prefetch=[
                models.Prefetch(query=self.embedder.embed([query])[0], using=DENSE, limit=k * 3),
                models.Prefetch(query=sparse_vector(query), using=SPARSE, limit=k * 3),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=k,
            with_payload=True,
        )
        return [
            PolicySnippet(
                source=p.payload["source"],
                title=p.payload["title"],
                text=p.payload["text"],
                score=round(float(p.score), 4),
                tags=p.payload.get("tags", []),
            )
            for p in result.points
            if p.payload
        ]

    def close(self) -> None:
        try:
            self.client.delete_collection(self.collection)
        except Exception:  # pragma: no cover - best effort cleanup
            logger.debug("failed to delete collection %s", self.collection)
