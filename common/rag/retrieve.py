"""
Retriever for the paper RAG index.

加载 papers_index.npz (L2-normalized) + papers_meta.jsonl, 支持 sync / async 检索.
- 余弦相似度 = 归一化后内积 (np.float32 dot product).
- section_boost: 软加权 (results/discussion 略加分, body/intro 略减分).
- section_filter: 硬过滤, 只允许特定 section.

CLI 用法:
  python common/rag/retrieve.py "Cu Mg ratio S phase precipitation" -k 5
  python common/rag/retrieve.py "hot cracking Cu>5%" -k 8 --section results --section discussion
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI

try:
    from common.rag.identity import (
        BM25_DOCUMENT_VERSION,
        CHUNK_ID_VERSION,
        INDEX_ID_SCHEMA_VERSION,
        ordered_id_fingerprint,
        validate_current_chunk_records,
        validate_current_id_list,
    )
    from common.rag.tokenization import (
        BM25_TOKENIZATION_VERSION,
        tokenize_bm25,
    )
except ModuleNotFoundError:  # direct ``python common/rag/retrieve.py``
    from identity import (
        BM25_DOCUMENT_VERSION,
        CHUNK_ID_VERSION,
        INDEX_ID_SCHEMA_VERSION,
        ordered_id_fingerprint,
        validate_current_chunk_records,
        validate_current_id_list,
    )
    from tokenization import BM25_TOKENIZATION_VERSION, tokenize_bm25

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

RAG_DIR = Path(__file__).resolve().parent
load_dotenv(RAG_DIR.parent / "llm" / ".env", override=False)

VEC_PATH = RAG_DIR / "papers_index.npz"
META_PATH = RAG_DIR / "papers_meta.jsonl"
BM25_PATH = RAG_DIR / "papers_bm25.pkl"

MODEL = "text-embedding-v4"
EMBED_DIM = 1024
BASE_URL = os.environ.get(
    "DASHSCOPE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)

DEFAULT_SECTION_BOOST: dict[str, float] = {
    "results": 1.10,
    "discussion": 1.10,
    "conclusion": 1.05,
    "methods": 1.00,
    "experimental": 1.00,
    "body": 0.95,
    "introduction": 0.92,
}


class Retriever:
    """Hybrid retriever: dense (cosine) + sparse (BM25) over embedded paper chunks."""

    def __init__(self) -> None:
        if not VEC_PATH.exists() or not META_PATH.exists():
            raise FileNotFoundError(
                "Missing index files. Run build_index.py first.\n"
                f"  expected: {VEC_PATH}\n  expected: {META_PATH}"
            )
        try:
            with np.load(VEC_PATH, allow_pickle=False) as archive:
                required = {
                    "embeddings",
                    "chunk_ids",
                    "chunk_id_version",
                    "ordered_id_fingerprint",
                    "index_id_schema_version",
                }
                missing = sorted(required - set(archive.files))
                if missing:
                    raise RuntimeError(
                        "Dense index lacks ordered identity metadata "
                        f"{missing}; run build_index.py --migrate-current-meta-ids"
                    )
                self.embeddings = np.asarray(archive["embeddings"], dtype=np.float32)
                dense_ids_raw = np.asarray(archive["chunk_ids"]).reshape(-1).tolist()
                dense_id_version = str(
                    np.asarray(archive["chunk_id_version"]).reshape(-1)[0]
                )
                dense_fingerprint = str(
                    np.asarray(archive["ordered_id_fingerprint"]).reshape(-1)[0]
                )
                dense_schema = str(
                    np.asarray(archive["index_id_schema_version"]).reshape(-1)[0]
                )
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"Cannot load dense RAG index: {VEC_PATH}") from exc
        self.meta: list[dict] = [
            json.loads(l)
            for l in META_PATH.read_text(encoding="utf-8").splitlines()
            if l.strip()
        ]
        if self.embeddings.ndim != 2:
            raise RuntimeError(
                f"Dense embeddings must be 2-D, got {self.embeddings.shape}"
            )
        if self.embeddings.shape[0] != len(self.meta):
            raise RuntimeError(
                "index/meta size mismatch: "
                f"{self.embeddings.shape[0]} vs {len(self.meta)}"
            )
        if not np.all(np.isfinite(self.embeddings)):
            raise RuntimeError("Dense RAG index contains non-finite values")
        meta_ids = validate_current_chunk_records(
            self.meta, context="retriever metadata"
        )
        dense_ids = validate_current_id_list(
            dense_ids_raw, context="retriever dense chunk_ids"
        )
        if dense_ids != meta_ids:
            raise RuntimeError("Dense row IDs do not match ordered metadata IDs")
        if dense_id_version != CHUNK_ID_VERSION:
            raise RuntimeError(
                f"Dense chunk ID version mismatch: {dense_id_version!r} != "
                f"{CHUNK_ID_VERSION!r}"
            )
        if dense_schema != INDEX_ID_SCHEMA_VERSION:
            raise RuntimeError(
                f"Dense ID schema mismatch: {dense_schema!r} != "
                f"{INDEX_ID_SCHEMA_VERSION!r}"
            )
        expected_fingerprint = ordered_id_fingerprint(meta_ids)
        if dense_fingerprint != expected_fingerprint:
            raise RuntimeError(
                "Dense ordered ID fingerprint does not match metadata row order"
            )

        # ── BM25 ──
        if not BM25_PATH.exists():
            raise FileNotFoundError(
                f"Missing required hybrid-retrieval artifact: {BM25_PATH}. "
                "Run build_index.py --bm25-only."
            )
        with open(BM25_PATH, "rb") as f:
            bm25_data = pickle.load(f)
        stored_tokenization = bm25_data.get("tokenization_version")
        if stored_tokenization != BM25_TOKENIZATION_VERSION:
            raise RuntimeError(
                "BM25 tokenization version mismatch: "
                f"{stored_tokenization!r} != {BM25_TOKENIZATION_VERSION!r}. "
                "Run build_index.py --bm25-only."
            )
        if bm25_data.get("document_version") != BM25_DOCUMENT_VERSION:
            raise RuntimeError(
                "BM25 document schema mismatch; rebuild with title + section + text "
                "using build_index.py --bm25-only."
            )
        if bm25_data.get("chunk_id_version") != CHUNK_ID_VERSION:
            raise RuntimeError(
                "BM25 chunk ID version mismatch; run build_index.py --bm25-only."
            )
        if bm25_data.get("index_id_schema_version") != INDEX_ID_SCHEMA_VERSION:
            raise RuntimeError(
                "BM25 ordered-ID schema mismatch; run build_index.py --bm25-only."
            )
        bm25_ids = validate_current_id_list(
            bm25_data.get("chunk_ids", []), context="BM25 chunk_ids"
        )
        if bm25_ids != meta_ids:
            raise RuntimeError(
                "BM25 row IDs differ from dense/meta order; rebuild BM25."
            )
        if bm25_data.get("ordered_id_fingerprint") != expected_fingerprint:
            raise RuntimeError(
                "BM25 ordered ID fingerprint differs from dense/meta artifacts."
            )
        self._bm25 = bm25_data["bm25"]
        self._bm25_docs = bm25_data["tokenized"]
        if len(self._bm25_docs) != self.embeddings.shape[0]:
            raise RuntimeError(
                "BM25 index size mismatch: "
                f"{len(self._bm25_docs)} docs vs "
                f"{self.embeddings.shape[0]} dense rows. Rebuild the index."
            )

        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("DASHSCOPE_API_KEY not set (check common/llm/.env)")
        self._sync_client = OpenAI(api_key=api_key, base_url=BASE_URL)
        self._async_client = AsyncOpenAI(api_key=api_key, base_url=BASE_URL)

    @property
    def size(self) -> int:
        return int(self.embeddings.shape[0])

    @property
    def has_bm25(self) -> bool:
        return self._bm25 is not None

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v

    def _bm25_scores(
        self, query: str, normalization_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        if self._bm25 is None:
            raise RuntimeError(
                "BM25 is required by the current hybrid-retrieval protocol. "
                "Rebuild it with build_index.py --bm25-only."
            )
        tokens = tokenize_bm25(query)
        if not tokens:
            return np.zeros(self.embeddings.shape[0], dtype=np.float32)
        raw = np.array(self._bm25.get_scores(tokens), dtype=np.float32)
        if len(raw) != self.embeddings.shape[0]:
            raise RuntimeError(
                "BM25 score length mismatch: "
                f"{len(raw)} scores vs {self.embeddings.shape[0]} dense rows. "
                "Rebuild the index instead of silently dropping sparse retrieval."
            )
        # 归一化到 [0, 1]
        values = raw if normalization_mask is None else raw[normalization_mask]
        rmax = float(values.max()) if values.size else 0.0
        return raw / rmax if rmax > 0 else raw

    def _score(
        self,
        q_vec: np.ndarray,
        top_k: int,
        section_filter: Optional[list[str]],
        section_boost: Optional[dict[str, float]],
        bm25_weight: float = 0.5,
        bm25_query: str = "",
        max_per_source: Optional[int] = 2,
    ) -> list[dict]:
        """Fuse dense/BM25 scores and enforce paper-level result diversity."""
        if top_k <= 0:
            return []
        if not 0.0 <= bm25_weight <= 1.0:
            raise ValueError("bm25_weight must be within [0, 1]")
        if max_per_source is not None and max_per_source <= 0:
            raise ValueError("max_per_source must be positive or None")

        q_vec = np.asarray(q_vec, dtype=np.float32).reshape(-1)
        if q_vec.shape[0] != self.embeddings.shape[1]:
            raise ValueError(
                f"Query dimension {q_vec.shape[0]} != index dimension "
                f"{self.embeddings.shape[1]}"
            )
        dense_cosine = self.embeddings @ q_vec
        dense_component = np.array(dense_cosine, copy=True)
        bm25_component = np.zeros(self.embeddings.shape[0], dtype=np.float32)

        if section_filter:
            allowed_sections = set(section_filter)
            allowed_mask = np.asarray(
                [
                    meta.get("section", "body") in allowed_sections
                    for meta in self.meta
                ],
                dtype=bool,
            )
        else:
            allowed_mask = np.ones(self.embeddings.shape[0], dtype=bool)
        if not np.any(allowed_mask):
            return []

        if bm25_weight > 0 and bm25_query:
            bm25_component = self._bm25_scores(
                bm25_query, normalization_mask=allowed_mask
            )
            allowed_dense = dense_cosine[allowed_mask]
            d_min, d_max = float(allowed_dense.min()), float(allowed_dense.max())
            if d_max > d_min:
                dense_component = (dense_cosine - d_min) / (d_max - d_min)
            else:
                dense_component = np.zeros_like(dense_cosine)
            fused_scores = (
                (1 - bm25_weight) * dense_component
                + bm25_weight * bm25_component
            )
        else:
            fused_scores = np.array(dense_component, copy=True)

        final_scores = np.array(fused_scores, copy=True)
        boost = section_boost if section_boost is not None else DEFAULT_SECTION_BOOST
        if boost:
            multipliers = np.asarray(
                [boost.get(m.get("section", "body"), 1.0) for m in self.meta],
                dtype=np.float32,
            )
            final_scores *= multipliers

        final_scores = np.where(allowed_mask, final_scores, -np.inf)

        hits: list[dict] = []
        paper_counts: dict[str, int] = {}
        # Full sorting is cheap at this corpus scale and lets selection continue
        # past rows rejected by the per-paper cap.
        for raw_index in np.argsort(-final_scores, kind="stable"):
            index = int(raw_index)
            score = float(final_scores[index])
            if not np.isfinite(score):
                continue
            meta = self.meta[index]
            paper_key = str(
                meta.get("source_file")
                or meta.get("paper_title")
                or f"row:{index}"
            )
            if max_per_source is not None:
                count = paper_counts.get(paper_key, 0)
                if count >= max_per_source:
                    continue
                paper_counts[paper_key] = count + 1
            hits.append(
                {
                    "score": score,
                    "dense_score": float(dense_component[index]),
                    "dense_cosine": float(dense_cosine[index]),
                    "bm25_score": float(bm25_component[index]),
                    "fused_score": float(fused_scores[index]),
                    "text": meta.get("text", ""),
                    "paper_title": meta.get("paper_title", ""),
                    "source_file": meta.get("source_file", ""),
                    "section": meta.get("section", ""),
                    "chunk_id": meta.get("chunk_id", ""),
                    "chunk_idx": meta.get("chunk_idx"),
                    "source_chunk_order": meta.get("source_chunk_order"),
                    "word_count": meta.get("word_count"),
                }
            )
            if len(hits) >= top_k:
                break
        return hits

    def _embed_query_sync(self, query: str) -> np.ndarray:
        resp = self._sync_client.embeddings.create(
            model=MODEL,
            input=[query],
            dimensions=EMBED_DIM,
            encoding_format="float",
        )
        return self._normalize(np.asarray(resp.data[0].embedding, dtype=np.float32))

    async def _embed_query_async(self, query: str) -> np.ndarray:
        resp = await self._async_client.embeddings.create(
            model=MODEL,
            input=[query],
            dimensions=EMBED_DIM,
            encoding_format="float",
        )
        return self._normalize(np.asarray(resp.data[0].embedding, dtype=np.float32))

    def search(
        self,
        query: str,
        top_k: int = 5,
        section_filter: Optional[list[str]] = None,
        section_boost: Optional[dict[str, float]] = None,
        bm25_weight: float = 0.5,
        bm25_query: str = "",
        max_per_source: Optional[int] = 2,
        max_chunks_per_paper: Optional[int] = None,
    ) -> list[dict]:
        if max_chunks_per_paper is not None:
            if max_per_source != 2 and max_per_source != max_chunks_per_paper:
                raise ValueError("Conflicting per-source diversity limits")
            max_per_source = max_chunks_per_paper
        q = self._embed_query_sync(query)
        return self._score(q, top_k, section_filter, section_boost,
                           bm25_weight=bm25_weight,
                           bm25_query=bm25_query or query,
                           max_per_source=max_per_source)

    async def asearch(
        self,
        query: str,
        top_k: int = 5,
        section_filter: Optional[list[str]] = None,
        section_boost: Optional[dict[str, float]] = None,
        bm25_weight: float = 0.5,
        bm25_query: str = "",
        max_per_source: Optional[int] = 2,
        max_chunks_per_paper: Optional[int] = None,
    ) -> list[dict]:
        if max_chunks_per_paper is not None:
            if max_per_source != 2 and max_per_source != max_chunks_per_paper:
                raise ValueError("Conflicting per-source diversity limits")
            max_per_source = max_chunks_per_paper
        q = await self._embed_query_async(query)
        return self._score(q, top_k, section_filter, section_boost,
                           bm25_weight=bm25_weight,
                           bm25_query=bm25_query or query,
                           max_per_source=max_per_source)


def _fmt_hit(h: dict, snippet_chars: int = 320) -> str:
    text = h["text"].replace("\n", " ").strip()
    if len(text) > snippet_chars:
        text = text[:snippet_chars] + "..."
    title = h["paper_title"] or h["source_file"]
    return (
        f"  score={h['score']:.4f}  [{h['section']}]  {title}\n"
        f"    {text}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the paper RAG index.")
    parser.add_argument("query", help="Search query (Chinese or English)")
    parser.add_argument("-k", "--top_k", type=int, default=5)
    parser.add_argument(
        "--max-per-paper",
        type=int,
        default=2,
        help="Maximum chunks returned from one source paper (default: 2)",
    )
    parser.add_argument(
        "--section",
        action="append",
        default=None,
        help="Hard-filter to a section (repeatable). E.g. --section results --section discussion",
    )
    parser.add_argument(
        "--no-boost",
        action="store_true",
        help="Disable default section boost (give every section equal weight).",
    )
    args = parser.parse_args()

    r = Retriever()
    print(f"Index: {r.size} chunks")
    hits = r.search(
        args.query,
        top_k=args.top_k,
        section_filter=args.section,
        section_boost={} if args.no_boost else None,
        max_per_source=args.max_per_paper,
    )
    print(f"Query: {args.query}\n" + "=" * 70)
    for i, h in enumerate(hits, 1):
        print(f"[{i}]")
        print(_fmt_hit(h))
        print()


if __name__ == "__main__":
    main()
