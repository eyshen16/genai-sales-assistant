from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
from fastembed import TextEmbedding

from ingest import PROJECT_ROOT, ingest_documents


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Selected empirically for the current synthetic corpus based on k-sensitivity
# testing. Re-evaluate if the corpus size or composition changes.
DEFAULT_RETRIEVAL_TOP_K = 6
DEFAULT_RETRIEVAL_MODE = "semantic"
ALLOWED_RETRIEVAL_MODES = {"semantic", "lexical", "hybrid"}
BM25_K1 = 1.5
BM25_B = 0.75
RRF_K = 60


@dataclass(frozen=True)
class RetrievalIndex:
    model_name: str
    chunks: list[dict[str, str]]
    embeddings: np.ndarray


def ingest_and_build_index(
    *,
    project_root: Path | str = PROJECT_ROOT,
    as_of_date: date | None = None,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> RetrievalIndex:
    chunks = ingest_documents(project_root=project_root, as_of_date=as_of_date)
    return build_retrieval_index(chunks=chunks, model_name=model_name)


def build_retrieval_index(
    *,
    chunks: list[dict[str, str]],
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> RetrievalIndex:
    if not chunks:
        return RetrievalIndex(
            model_name=model_name,
            chunks=[],
            embeddings=np.empty((0, 0), dtype=np.float32),
        )

    embedder = TextEmbedding(model_name=model_name)
    embedding_inputs = [_chunk_text_for_embedding(chunk) for chunk in chunks]
    embeddings = np.vstack(list(embedder.embed(embedding_inputs))).astype(np.float32)

    return RetrievalIndex(
        model_name=model_name,
        chunks=chunks,
        embeddings=embeddings,
    )


def retrieve_relevant_chunks(
    *,
    query: str,
    index: RetrievalIndex,
    top_k: int = DEFAULT_RETRIEVAL_TOP_K,
    authoritative_source_ids: list[str] | None = None,
    include_diagnostics: bool = False,
    retrieval_mode: str = DEFAULT_RETRIEVAL_MODE,
) -> list[dict[str, Any]] | dict[str, Any]:
    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("Query must not be empty.")
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    if retrieval_mode not in ALLOWED_RETRIEVAL_MODES:
        raise ValueError(f"Unsupported retrieval mode: {retrieval_mode}")
    if not index.chunks:
        empty_result: list[dict[str, Any]] = []
        if include_diagnostics:
            return {
                "chunks": empty_result,
                "requested_authoritative_source_ids": list(authoritative_source_ids or []),
                "retrieved_authoritative_source_ids": [],
                "authority_gap": bool(authoritative_source_ids),
                "retrieval_mode": retrieval_mode,
            }
        return empty_result

    scores = _retrieval_scores(query=normalized_query, index=index, retrieval_mode=retrieval_mode)

    requested_authoritative_ids = list(dict.fromkeys(authoritative_source_ids or []))
    ranked_indices = _assemble_ranked_indices(
        chunks=index.chunks,
        scores=scores,
        top_k=top_k,
        authoritative_source_ids=requested_authoritative_ids,
    )

    results: list[dict[str, Any]] = []
    for chunk_index in ranked_indices:
        chunk = dict(index.chunks[int(chunk_index)])
        chunk["score"] = float(scores[int(chunk_index)])
        results.append(chunk)

    if not include_diagnostics:
        return results

    retrieved_authoritative_source_ids = list(
        dict.fromkeys(
            chunk["source_id"]
            for chunk in results
            if chunk["source_id"] in set(requested_authoritative_ids)
        )
    )
    has_eligible_authoritative_chunks = any(
        chunk["source_id"] in set(requested_authoritative_ids) for chunk in index.chunks
    )
    authority_gap = bool(requested_authoritative_ids) and not has_eligible_authoritative_chunks
    return {
        "chunks": results,
        "requested_authoritative_source_ids": requested_authoritative_ids,
        "retrieved_authoritative_source_ids": retrieved_authoritative_source_ids,
        "authority_gap": authority_gap,
        "retrieval_mode": retrieval_mode,
    }


def _retrieval_scores(*, query: str, index: RetrievalIndex, retrieval_mode: str) -> np.ndarray:
    if retrieval_mode == "lexical":
        return bm25_scores(query=query, chunks=index.chunks)

    embedder = TextEmbedding(model_name=index.model_name)
    query_embedding = np.vstack(list(embedder.embed([query]))).astype(np.float32)[0]
    dense_scores = cosine_similarity(query_embedding=query_embedding, document_embeddings=index.embeddings)
    if retrieval_mode == "semantic":
        return dense_scores

    lexical_scores = bm25_scores(query=query, chunks=index.chunks)
    return reciprocal_rank_fusion(
        [dense_scores, lexical_scores],
        candidate_masks=[None, lexical_scores > 0],
        rrf_k=RRF_K,
    )


def tokenize_for_lexical_retrieval(text: str) -> list[str]:
    """Tokenize deterministically while retaining identifiers and versions."""
    tokens: list[str] = []
    for token in re.findall(r"[a-z0-9]+(?:[.-][a-z0-9]+)*", text.casefold()):
        tokens.append(token)
        if "-" in token:
            tokens.extend(part for part in token.split("-") if part)
    return tokens


def bm25_scores(
    *, query: str, chunks: list[dict[str, str]], k1: float = BM25_K1, b: float = BM25_B,
) -> np.ndarray:
    if not chunks:
        return np.empty(0, dtype=np.float32)
    documents = [tokenize_for_lexical_retrieval(_chunk_text_for_embedding(chunk)) for chunk in chunks]
    query_tokens = list(dict.fromkeys(tokenize_for_lexical_retrieval(query)))
    document_lengths = [len(document) for document in documents]
    average_length = sum(document_lengths) / len(document_lengths) if document_lengths else 0.0
    document_frequency = {
        token: sum(token in set(document) for document in documents)
        for token in query_tokens
    }
    scores = np.zeros(len(documents), dtype=np.float32)
    for index, document in enumerate(documents):
        frequencies = {token: document.count(token) for token in query_tokens}
        for token in query_tokens:
            frequency = frequencies[token]
            if not frequency:
                continue
            frequency_in_documents = document_frequency[token]
            inverse_document_frequency = math.log(
                1 + (len(documents) - frequency_in_documents + 0.5) / (frequency_in_documents + 0.5)
            )
            length_normalization = 1 - b + b * document_lengths[index] / max(average_length, 1e-12)
            scores[index] += inverse_document_frequency * (
                frequency * (k1 + 1) / (frequency + k1 * length_normalization)
            )
    return scores


def reciprocal_rank_fusion(
    score_sets: list[np.ndarray],
    *,
    candidate_masks: list[np.ndarray | None] | None = None,
    rrf_k: int = RRF_K,
) -> np.ndarray:
    if not score_sets:
        return np.empty(0, dtype=np.float32)
    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive.")
    size = len(score_sets[0])
    if any(len(scores) != size for scores in score_sets):
        raise ValueError("All score sets must have the same length.")
    masks = candidate_masks or [None] * len(score_sets)
    if len(masks) != len(score_sets):
        raise ValueError("candidate_masks must align with score_sets.")
    fused = np.zeros(size, dtype=np.float32)
    for scores, mask in zip(score_sets, masks):
        candidates = range(size) if mask is None else np.flatnonzero(mask)
        ranked = sorted(candidates, key=lambda index: (-float(scores[index]), int(index)))
        for rank, index in enumerate(ranked, start=1):
            fused[index] += 1.0 / (rrf_k + rank)
    return fused


def cosine_similarity(
    *,
    query_embedding: np.ndarray,
    document_embeddings: np.ndarray,
) -> np.ndarray:
    query_norm = np.linalg.norm(query_embedding)
    document_norms = np.linalg.norm(document_embeddings, axis=1)
    safe_denominator = np.maximum(document_norms * query_norm, 1e-12)
    return (document_embeddings @ query_embedding) / safe_denominator


def _chunk_text_for_embedding(chunk: dict[str, str]) -> str:
    # Include title and section labels so retrieval can use document structure
    # without changing the stored chunk text returned to downstream consumers.
    return "\n".join(
        [
            f"Title: {chunk['title']}",
            f"Section: {chunk['section']}",
            f"Source: {chunk['source_name']}",
            chunk["text"],
        ]
    )


def _assemble_ranked_indices(
    *,
    chunks: list[dict[str, str]],
    scores: np.ndarray,
    top_k: int,
    authoritative_source_ids: list[str],
) -> list[int]:
    all_ranked = list(np.argsort(scores)[::-1])
    if not authoritative_source_ids:
        return [int(index) for index in all_ranked[: min(top_k, len(chunks))]]

    requested_set = set(authoritative_source_ids)
    authoritative_ranked = [
        int(index)
        for index in all_ranked
        if chunks[int(index)]["source_id"] in requested_set
    ]
    secondary_ranked = [
        int(index)
        for index in all_ranked
        if chunks[int(index)]["source_id"] not in requested_set
    ]

    if not authoritative_ranked:
        return secondary_ranked[: min(top_k, len(secondary_ranked))]

    selected = [authoritative_ranked[0]]
    remaining_candidates = authoritative_ranked[1:] + secondary_ranked
    remaining_candidates.sort(key=lambda idx: float(scores[idx]), reverse=True)
    for index in remaining_candidates:
        if len(selected) >= min(top_k, len(chunks)):
            break
        selected.append(index)
    return selected
