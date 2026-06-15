#!/usr/bin/env python3
"""Experiment bridge for RIC synthetic pilots.

This runner implements the first R001-R004 experiments from
refine-logs/EXPERIMENT_PLAN.md without requiring FAISS, sentence-transformers,
or a GPU. It uses a controlled synthetic retrieval benchmark with ground-truth
topic relevance, so evaluation metrics are computed against dataset labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


EPS = 1e-9


@dataclass
class SyntheticRetrievalData:
    old_docs: np.ndarray
    new_docs: np.ndarray
    old_queries: np.ndarray
    new_queries: np.ndarray
    doc_topics: np.ndarray
    query_topics: np.ndarray
    qrels: List[set[int]]
    sparse_scores: np.ndarray
    query_frequency: np.ndarray
    topic_drift: np.ndarray


@dataclass
class Adapter:
    name: str
    matrix: np.ndarray
    bias: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return l2_normalize(x @ self.matrix + self.bias)


@dataclass
class SearchResult:
    indices: np.ndarray
    scores: np.ndarray


@dataclass
class BeirTextRecords:
    corpus_texts: List[str]
    query_texts: List[str]
    qrels: List[set[int]]
    n_docs: int
    n_queries: int


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, EPS)


def unit_scale(values: np.ndarray) -> np.ndarray:
    lo = float(values.min())
    hi = float(values.max())
    if hi - lo < EPS:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def topk(scores: np.ndarray, k: int) -> SearchResult:
    k = min(k, scores.shape[1])
    raw = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    raw_scores = np.take_along_axis(scores, raw, axis=1)
    order = np.argsort(-raw_scores, axis=1)
    idx = np.take_along_axis(raw, order, axis=1)
    vals = np.take_along_axis(scores, idx, axis=1)
    return SearchResult(indices=idx, scores=vals)


def cosine_search(queries: np.ndarray, docs: np.ndarray, k: int) -> SearchResult:
    return topk(queries @ docs.T, k)


def mismatched_space_search(queries: np.ndarray, docs: np.ndarray, k: int) -> SearchResult:
    if queries.shape[1] == docs.shape[1]:
        return cosine_search(queries, docs, k)
    common_dim = min(queries.shape[1], docs.shape[1])
    return cosine_search(
        l2_normalize(queries[:, :common_dim]),
        l2_normalize(docs[:, :common_dim]),
        k,
    )


def random_orthogonal(dim: int, rng: np.random.Generator) -> np.ndarray:
    a = rng.normal(size=(dim, dim))
    q, _ = np.linalg.qr(a)
    return q


def generate_synthetic_data(
    seed: int,
    n_docs: int,
    n_queries: int,
    dim: int,
    n_topics: int,
    relevant_per_query: int,
    drift_strength: float,
    hard_topic_fraction: float,
) -> SyntheticRetrievalData:
    rng = np.random.default_rng(seed)
    centers = l2_normalize(rng.normal(size=(n_topics, dim)))
    transform = random_orthogonal(dim, rng)
    nonlinear_basis = l2_normalize(rng.normal(size=(n_topics, dim)))

    hard_topics = set(
        rng.choice(n_topics, size=max(1, int(n_topics * hard_topic_fraction)), replace=False)
    )
    topic_drift = rng.uniform(0.05, drift_strength, size=n_topics)
    for topic in hard_topics:
        topic_drift[topic] *= 2.5

    doc_topics = np.repeat(np.arange(n_topics), math.ceil(n_docs / n_topics))[:n_docs]
    rng.shuffle(doc_topics)
    query_topics = rng.choice(n_topics, size=n_queries, replace=True)

    latent_docs = centers[doc_topics] + 0.28 * rng.normal(size=(n_docs, dim))
    latent_queries = centers[query_topics] + 0.32 * rng.normal(size=(n_queries, dim))
    latent_docs = l2_normalize(latent_docs)
    latent_queries = l2_normalize(latent_queries)

    old_docs = l2_normalize(latent_docs + 0.04 * rng.normal(size=(n_docs, dim)))
    old_queries = l2_normalize(latent_queries + 0.04 * rng.normal(size=(n_queries, dim)))

    doc_drift = topic_drift[doc_topics, None] * nonlinear_basis[doc_topics]
    query_drift = topic_drift[query_topics, None] * nonlinear_basis[query_topics]
    new_docs = l2_normalize(latent_docs @ transform + doc_drift + 0.04 * rng.normal(size=(n_docs, dim)))
    new_queries = l2_normalize(
        latent_queries @ transform + query_drift + 0.04 * rng.normal(size=(n_queries, dim))
    )

    qrels: List[set[int]] = []
    semantic_scores = latent_queries @ latent_docs.T
    for qi, topic in enumerate(query_topics):
        candidates = np.flatnonzero(doc_topics == topic)
        if len(candidates) <= relevant_per_query:
            relevant = candidates
        else:
            topic_scores = semantic_scores[qi, candidates]
            order = np.argsort(-topic_scores)[:relevant_per_query]
            relevant = candidates[order]
        qrels.append(set(int(x) for x in relevant))

    sparse_scores = np.zeros((n_queries, n_docs), dtype=np.float64)
    for qi, topic in enumerate(query_topics):
        sparse_scores[qi] = rng.normal(loc=0.0, scale=0.35, size=n_docs)
        same_topic = doc_topics == topic
        sparse_scores[qi, same_topic] += rng.normal(loc=1.2, scale=0.25, size=same_topic.sum())
    query_frequency = rng.zipf(a=1.4, size=n_queries).astype(np.float64)
    query_frequency /= query_frequency.mean()

    return SyntheticRetrievalData(
        old_docs=old_docs,
        new_docs=new_docs,
        old_queries=old_queries,
        new_queries=new_queries,
        doc_topics=doc_topics,
        query_topics=query_topics,
        qrels=qrels,
        sparse_scores=sparse_scores,
        query_frequency=query_frequency,
        topic_drift=topic_drift,
    )


def fit_procrustes(new_anchor: np.ndarray, old_anchor: np.ndarray) -> Adapter:
    x_mean = new_anchor.mean(axis=0, keepdims=True)
    y_mean = old_anchor.mean(axis=0, keepdims=True)
    x = new_anchor - x_mean
    y = old_anchor - y_mean
    u, _, vt = np.linalg.svd(x.T @ y, full_matrices=False)
    r = u @ vt
    bias = (y_mean - x_mean @ r).reshape(-1)
    return Adapter("procrustes", r, bias)


def fit_affine(new_anchor: np.ndarray, old_anchor: np.ndarray, alpha: float = 1e-3) -> Adapter:
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(new_anchor, old_anchor)
    return Adapter("affine", model.coef_.T, model.intercept_)


def metric_values(result: SearchResult, qrels: List[set[int]], k: int) -> Dict[str, float]:
    recalls = []
    mrrs = []
    ndcgs = []
    for qi, rel in enumerate(qrels):
        retrieved = list(map(int, result.indices[qi, :k]))
        hits = [1 if doc in rel else 0 for doc in retrieved]
        recalls.append(sum(hits) / max(1, len(rel)))
        reciprocal = 0.0
        for rank, hit in enumerate(hits, start=1):
            if hit:
                reciprocal = 1.0 / rank
                break
        mrrs.append(reciprocal)
        dcg = sum(hit / math.log2(rank + 1) for rank, hit in enumerate(hits, start=1))
        ideal_hits = min(len(rel), k)
        idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
        ndcgs.append(dcg / max(idcg, EPS))
    return {
        f"recall@{k}": float(np.mean(recalls)),
        f"mrr@{k}": float(np.mean(mrrs)),
        f"ndcg@{k}": float(np.mean(ndcgs)),
    }


def per_query_recall(result: SearchResult, qrels: List[set[int]], k: int) -> np.ndarray:
    values = []
    for qi, rel in enumerate(qrels):
        hits = sum(1 for doc in result.indices[qi, :k] if int(doc) in rel)
        values.append(hits / max(1, len(rel)))
    return np.asarray(values, dtype=np.float64)


def per_query_ndcg(result: SearchResult, qrels: List[set[int]], k: int) -> np.ndarray:
    values = []
    for qi, rel in enumerate(qrels):
        hits = [1 if int(doc) in rel else 0 for doc in result.indices[qi, :k]]
        dcg = sum(hit / math.log2(rank + 1) for rank, hit in enumerate(hits, start=1))
        ideal_hits = min(len(rel), k)
        idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
        values.append(dcg / max(idcg, EPS))
    return np.asarray(values, dtype=np.float64)


def make_failure_labels(
    primary: SearchResult,
    full_new: SearchResult,
    qrels: List[set[int]],
    k: int,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, Dict[str, float | str]]:
    if args.failure_label == "absolute_recall":
        primary_recall = per_query_recall(primary, qrels, k)
        labels = (primary_recall < args.failure_recall_threshold).astype(int)
        meta: Dict[str, float | str] = {
            "failure_label": args.failure_label,
            "failure_recall_threshold": float(args.failure_recall_threshold),
        }
        return labels, meta
    if args.failure_label == "oracle_delta_recall":
        primary_values = per_query_recall(primary, qrels, k)
        oracle_values = per_query_recall(full_new, qrels, k)
        metric = "recall"
    elif args.failure_label == "oracle_delta_ndcg":
        primary_values = per_query_ndcg(primary, qrels, k)
        oracle_values = per_query_ndcg(full_new, qrels, k)
        metric = "ndcg"
    else:
        raise ValueError(f"Unsupported failure label: {args.failure_label}")
    loss = oracle_values - primary_values
    labels = (loss > args.oracle_delta).astype(int)
    meta = {
        "failure_label": args.failure_label,
        "oracle_delta": float(args.oracle_delta),
        "oracle_metric": metric,
        "mean_oracle_value": float(oracle_values.mean()),
        "mean_primary_value": float(primary_values.mean()),
        "mean_oracle_delta": float(loss.mean()),
    }
    return labels, meta


def jaccard(a: Iterable[int], b: Iterable[int]) -> float:
    sa = set(map(int, a))
    sb = set(map(int, b))
    return len(sa & sb) / max(1, len(sa | sb))


def score_entropy(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= np.maximum(probs.sum(axis=1, keepdims=True), EPS)
    return -(probs * np.log(np.maximum(probs, EPS))).sum(axis=1)


def build_features(
    data: SyntheticRetrievalData,
    primary: SearchResult,
    alt_results: Dict[str, SearchResult],
    sparse_result: SearchResult,
    mapped_queries: np.ndarray,
    new_anchor: np.ndarray,
    old_anchor: np.ndarray,
    anchor_residual: np.ndarray,
    adapter: Adapter,
    k: int,
) -> Tuple[np.ndarray, List[str]]:
    n_queries = data.new_queries.shape[0]
    margin = primary.scores[:, 0] - primary.scores[:, min(k - 1, primary.scores.shape[1] - 1)]
    entropy = score_entropy(primary.scores[:, :k])

    disagreements = []
    for qi in range(n_queries):
        sims = []
        for result in alt_results.values():
            sims.append(jaccard(primary.indices[qi, :k], result.indices[qi, :k]))
        disagreements.append(1.0 - float(np.mean(sims)))

    sparse_overlap = []
    for qi in range(n_queries):
        sparse_overlap.append(jaccard(primary.indices[qi, :k], sparse_result.indices[qi, :k]))

    anchor_sim = data.new_queries @ new_anchor.T
    nearest_anchor = np.argmax(anchor_sim, axis=1)
    local_anchor_residual = anchor_residual[nearest_anchor]
    query_ood = 1.0 - np.max(anchor_sim, axis=1)

    hit_topics = data.doc_topics[primary.indices[:, :k]]
    cell_drift = data.topic_drift[hit_topics].mean(axis=1)

    perturb = l2_normalize(data.new_queries + 0.01 * np.sign(data.new_queries))
    perturb_result = cosine_search(adapter.transform(perturb), data.old_docs, k)
    instability = np.array(
        [1.0 - jaccard(primary.indices[qi, :k], perturb_result.indices[qi, :k]) for qi in range(n_queries)]
    )

    feature_names = [
        "neg_margin",
        "entropy",
        "adapter_disagreement",
        "neg_sparse_overlap",
        "local_anchor_residual",
        "query_ood",
        "cell_drift",
        "candidate_instability",
    ]
    features = np.column_stack(
        [
            -margin,
            entropy,
            np.asarray(disagreements),
            -np.asarray(sparse_overlap),
            local_anchor_residual,
            query_ood,
            cell_drift,
            instability,
        ]
    )
    return features, feature_names


def build_pool_features(
    primary: SearchResult,
    pool: SearchResult,
    k: int,
) -> Tuple[np.ndarray, List[str]]:
    """Label-free features derived from the bounded new-space candidate pool.

    These describe what the SAME tight pool RIC-MG probes looks like, but
    WITHOUT using qrels: pool-induced rank churn vs the adapter result, the
    strength/margin/entropy of the pool re-ranking, and how many pool top-k
    documents are new relative to the adapter top-k.  They let a learned
    regressor "see" the new-space pool while still PREDICTING gain, which is the
    control needed to isolate measure-vs-predict from has-pool-info-vs-not.
    """
    n = primary.indices.shape[0]
    kk = min(k - 1, pool.scores.shape[1] - 1)
    churn = np.array(
        [1.0 - jaccard(primary.indices[qi, :k], pool.indices[qi, :k]) for qi in range(n)]
    )
    pool_top1 = pool.scores[:, 0]
    pool_margin = pool.scores[:, 0] - pool.scores[:, kk]
    pool_entropy = score_entropy(pool.scores[:, :k])
    new_in_topk = np.array(
        [len(set(pool.indices[qi, :k]) - set(primary.indices[qi, :k])) for qi in range(n)],
        dtype=np.float64,
    )
    feature_names = [
        "pool_churn",
        "pool_top1_newscore",
        "pool_margin",
        "pool_entropy",
        "pool_new_in_topk",
    ]
    features = np.column_stack([churn, pool_top1, pool_margin, pool_entropy, new_in_topk])
    return features, feature_names


def split_indices(n: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(0.4 * n)
    n_calib = int(0.3 * n)
    train = idx[:n_train]
    calib = idx[n_train : n_train + n_calib]
    test = idx[n_train + n_calib :]
    return train, calib, test


def fit_risk_scores(
    features: np.ndarray,
    labels: np.ndarray,
    train: np.ndarray,
    score_columns: List[int] | None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    x = features[:, score_columns] if score_columns is not None else features
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x[train])
    x_all = scaler.transform(x)
    if len(np.unique(labels[train])) < 2:
        constant = float(labels[train].mean())
        return np.full(labels.shape[0], constant), {"auc": float("nan"), "ap": float("nan")}
    clf = LogisticRegression(max_iter=1000, random_state=0)
    clf.fit(x_train, labels[train])
    scores = clf.predict_proba(x_all)[:, 1]
    try:
        auc = float(roc_auc_score(labels[train], scores[train]))
    except ValueError:
        auc = float("nan")
    try:
        ap = float(average_precision_score(labels[train], scores[train]))
    except ValueError:
        ap = float("nan")
    return scores, {"auc": auc, "ap": ap}


def score_fit_stats(scores: np.ndarray, labels: np.ndarray, train: np.ndarray) -> Dict[str, float]:
    if len(np.unique(labels[train])) < 2:
        return {"auc": float("nan"), "ap": float("nan")}
    try:
        auc = float(roc_auc_score(labels[train], scores[train]))
    except ValueError:
        auc = float("nan")
    try:
        ap = float(average_precision_score(labels[train], scores[train]))
    except ValueError:
        ap = float("nan")
    return {"auc": auc, "ap": ap}


def calibrate_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    calib: np.ndarray,
    alpha: float,
    alpha_factor: float = 1.0,
) -> float:
    target_alpha = alpha * alpha_factor
    candidates = np.unique(scores[calib])
    best_tau = float(candidates.min() - 1e-6)
    best_cov = -1
    for tau in candidates:
        safe = calib[scores[calib] <= tau]
        if len(safe) == 0:
            continue
        violation = labels[safe].mean()
        coverage = len(safe)
        if violation <= target_alpha and coverage > best_cov:
            best_tau = float(tau)
            best_cov = coverage
    return best_tau


def risk_eval(scores: np.ndarray, labels: np.ndarray, test: np.ndarray, tau: float) -> Dict[str, float]:
    safe = test[scores[test] <= tau]
    if len(safe) == 0:
        return {"coverage": 0.0, "violation": 0.0, "n_safe": 0}
    return {
        "coverage": float(len(safe) / len(test)),
        "violation": float(labels[safe].mean()),
        "n_safe": int(len(safe)),
    }


def mondrian_groups(values: np.ndarray, reference: np.ndarray, n_groups: int) -> np.ndarray:
    if n_groups <= 1:
        return np.zeros(values.shape[0], dtype=int)
    cuts = np.quantile(values[reference], np.linspace(0.0, 1.0, n_groups + 1)[1:-1])
    cuts = np.unique(cuts)
    if len(cuts) == 0:
        return np.zeros(values.shape[0], dtype=int)
    return np.digitize(values, cuts).astype(int)


def calibrate_mondrian_thresholds(
    scores: np.ndarray,
    labels: np.ndarray,
    calib: np.ndarray,
    groups: np.ndarray,
    alpha: float,
    alpha_factor: float,
) -> Dict[int, float]:
    thresholds: Dict[int, float] = {}
    for group_id in sorted(set(int(x) for x in groups[calib])):
        group_calib = calib[groups[calib] == group_id]
        if len(group_calib) == 0:
            continue
        thresholds[group_id] = calibrate_threshold(
            scores,
            labels,
            group_calib,
            alpha,
            alpha_factor,
        )
    return thresholds


def risk_eval_mondrian(
    scores: np.ndarray,
    labels: np.ndarray,
    test: np.ndarray,
    groups: np.ndarray,
    thresholds: Dict[int, float],
) -> Dict[str, float]:
    safe_mask = np.zeros(labels.shape[0], dtype=bool)
    for group_id, tau in thresholds.items():
        safe_mask |= (groups == group_id) & (scores <= tau)
    safe = test[safe_mask[test]]
    if len(safe) == 0:
        return {"coverage": 0.0, "violation": 0.0, "n_safe": 0}
    return {
        "coverage": float(len(safe) / len(test)),
        "violation": float(labels[safe].mean()),
        "n_safe": int(len(safe)),
    }


def binomial_upper_bound(failures: int, total: int, z: float = 1.64) -> float:
    if total <= 0:
        return 1.0
    p_hat = failures / total
    denom = 1.0 + z * z / total
    center = p_hat + z * z / (2.0 * total)
    radius = z * math.sqrt((p_hat * (1.0 - p_hat) + z * z / (4.0 * total)) / total)
    return float((center + radius) / denom)


def mondrian_safe_mask(scores: np.ndarray, groups: np.ndarray, thresholds: Dict[int, float]) -> np.ndarray:
    safe_mask = np.zeros(scores.shape[0], dtype=bool)
    for group_id, tau in thresholds.items():
        safe_mask |= (groups == group_id) & (scores <= tau)
    return safe_mask


def append_calibration_selected_mondrian_row(
    risk_scores_by_name: Dict[str, np.ndarray],
    risk_rows: List[Dict[str, float | str]],
    features: np.ndarray,
    column_map: Dict[str, int],
    labels: np.ndarray,
    train: np.ndarray,
    calib: np.ndarray,
    test: np.ndarray,
    alpha: float,
    alpha_factor: float,
    ucb_factor: float,
    ucb_z: float,
) -> None:
    group_sources = {
        "sparse": features[:, column_map["neg_sparse_overlap"]],
        "fullrisk": risk_scores_by_name["full_ric"],
        "guardrisk": risk_scores_by_name["guarded_ric"],
        "margin": features[:, column_map["neg_margin"]],
        "entropy": features[:, column_map["entropy"]],
        "disagree": features[:, column_map["adapter_disagreement"]],
        "resid": features[:, column_map["local_anchor_residual"]],
        "qood": features[:, column_map["query_ood"]],
        "cell": features[:, column_map["cell_drift"]],
    }
    score_names = [
        "full_ric",
        "guarded_ric",
        "local_residual_only",
        "margin_only",
        "entropy_only",
        "adapter_disagreement_only",
        "sparse_overlap_only",
    ]
    group_counts = [1, 2, 3, 4, 5, 8, 10, 12]
    calibration_factors = [0.0, 0.1, 0.25, 0.5, alpha_factor, 0.75, 1.0]
    accepted: List[Tuple[float, float, str, str, int, float, Dict[int, float], np.ndarray]] = []
    target_upper = alpha * ucb_factor
    for score_name in score_names:
        scores = risk_scores_by_name[score_name]
        for group_name, group_values in group_sources.items():
            for n_groups in group_counts:
                groups = mondrian_groups(group_values, calib, n_groups)
                for factor in calibration_factors:
                    thresholds = calibrate_mondrian_thresholds(
                        scores,
                        labels,
                        calib,
                        groups,
                        alpha,
                        factor,
                    )
                    safe_mask = mondrian_safe_mask(scores, groups, thresholds)
                    safe_calib = calib[safe_mask[calib]]
                    if len(safe_calib) == 0:
                        continue
                    failures = int(labels[safe_calib].sum())
                    ucb = binomial_upper_bound(failures, len(safe_calib), ucb_z)
                    if ucb <= target_upper:
                        coverage = len(safe_calib) / len(calib)
                        accepted.append(
                            (
                                coverage,
                                -ucb,
                                score_name,
                                group_name,
                                n_groups,
                                factor,
                                thresholds,
                                groups,
                            )
                        )
    if not accepted:
        risk_rows.append(
            {
                "run_id": "R003",
                "risk_model": "calibration_selected_mondrian_ric",
                "tau": json.dumps(
                    {
                        "selected": False,
                        "reason": "no_candidate_passed_calibration_ucb",
                        "target_upper": target_upper,
                        "z": ucb_z,
                    },
                    sort_keys=True,
                ),
                "train_auc": float("nan"),
                "train_ap": float("nan"),
                "coverage": 0.0,
                "violation": 0.0,
                "n_safe": 0,
            }
        )
        return

    _, neg_ucb, score_name, group_name, n_groups, factor, thresholds, groups = max(accepted)
    scores = risk_scores_by_name[score_name]
    fit_stats = score_fit_stats(scores, labels, train)
    eval_stats = risk_eval_mondrian(scores, labels, test, groups, thresholds)
    metadata = {
        "selected": True,
        "score": score_name,
        "group": group_name,
        "n_groups": n_groups,
        "alpha_factor": factor,
        "calibration_ucb": -neg_ucb,
        "target_upper": target_upper,
        "z": ucb_z,
        "thresholds": thresholds,
    }
    risk_rows.append(
        {
            "run_id": "R003",
            "risk_model": "calibration_selected_mondrian_ric",
            "tau": json.dumps(metadata, sort_keys=True),
            "train_auc": fit_stats["auc"],
            "train_ap": fit_stats["ap"],
            **eval_stats,
        }
    )


def append_guarded_ric_row(
    risk_scores_by_name: Dict[str, np.ndarray],
    risk_rows: List[Dict[str, float | str]],
    labels: np.ndarray,
    train: np.ndarray,
    calib: np.ndarray,
    test: np.ndarray,
    alpha: float,
    alpha_factor: float,
) -> None:
    guarded_scores = np.maximum(
        unit_scale(risk_scores_by_name["full_ric"]),
        unit_scale(risk_scores_by_name["local_residual_only"]),
    )
    fit_stats = score_fit_stats(guarded_scores, labels, train)
    tau = calibrate_threshold(guarded_scores, labels, calib, alpha, alpha_factor)
    eval_stats = risk_eval(guarded_scores, labels, test, tau)
    risk_scores_by_name["guarded_ric"] = guarded_scores
    risk_rows.append(
        {
            "run_id": "R003",
            "risk_model": "guarded_ric",
            "tau": tau,
            "train_auc": fit_stats["auc"],
            "train_ap": fit_stats["ap"],
            **eval_stats,
        }
    )


def append_mondrian_ric_rows(
    risk_scores_by_name: Dict[str, np.ndarray],
    risk_rows: List[Dict[str, float | str]],
    features: np.ndarray,
    column_map: Dict[str, int],
    labels: np.ndarray,
    train: np.ndarray,
    calib: np.ndarray,
    test: np.ndarray,
    alpha: float,
    alpha_factor: float,
) -> None:
    specs = [
        (
            "full_ric_mondrian_sparse_overlap_4",
            "full_ric",
            features[:, column_map["neg_sparse_overlap"]],
            4,
            alpha_factor,
        ),
        (
            "guarded_ric_mondrian_risk_8",
            "guarded_ric",
            risk_scores_by_name["full_ric"],
            8,
            alpha_factor,
        ),
    ]
    for name, score_name, group_values, n_groups, factor in specs:
        scores = risk_scores_by_name[score_name]
        groups = mondrian_groups(group_values, calib, n_groups)
        thresholds = calibrate_mondrian_thresholds(scores, labels, calib, groups, alpha, factor)
        eval_stats = risk_eval_mondrian(scores, labels, test, groups, thresholds)
        fit_stats = score_fit_stats(scores, labels, train)
        risk_rows.append(
            {
                "run_id": "R003",
                "risk_model": name,
                "tau": json.dumps(thresholds, sort_keys=True),
                "train_auc": fit_stats["auc"],
                "train_ap": fit_stats["ap"],
                **eval_stats,
            }
        )


def fit_gain_scores(features: np.ndarray, gains: np.ndarray, fit_idx: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    x_fit = scaler.fit_transform(features[fit_idx])
    x_all = scaler.transform(features)
    if len(fit_idx) < 20 or np.allclose(gains[fit_idx], gains[fit_idx][0]):
        return np.full(features.shape[0], float(np.mean(gains[fit_idx]) if len(fit_idx) else 0.0))
    model = GradientBoostingRegressor(random_state=0, max_depth=2, n_estimators=80, learning_rate=0.05)
    model.fit(x_fit, gains[fit_idx])
    return np.maximum(0.0, model.predict(x_all))


def mixed_refresh_search(
    data: SyntheticRetrievalData,
    mapped_queries: np.ndarray,
    refreshed_docs: np.ndarray,
    k: int,
    score_mix: str,
) -> SearchResult:
    old_scores = mapped_queries @ data.old_docs.T
    new_scores = data.new_queries @ data.new_docs.T
    mixed = old_scores.copy()
    if score_mix == "replace":
        mixed[:, refreshed_docs] = new_scores[:, refreshed_docs]
    elif score_mix in {"zmax", "zblend"}:
        old_mean = old_scores.mean(axis=1, keepdims=True)
        old_std = old_scores.std(axis=1, keepdims=True) + EPS
        new_mean = new_scores.mean(axis=1, keepdims=True)
        new_std = new_scores.std(axis=1, keepdims=True) + EPS
        calibrated_new = (new_scores - new_mean) / new_std * old_std + old_mean
        if score_mix == "zmax":
            mixed[:, refreshed_docs] = np.maximum(
                old_scores[:, refreshed_docs],
                calibrated_new[:, refreshed_docs],
            )
        else:
            mixed[:, refreshed_docs] = (
                0.5 * old_scores[:, refreshed_docs]
                + 0.5 * calibrated_new[:, refreshed_docs]
            )
    else:
        raise ValueError(f"Unsupported refresh score mix: {score_mix}")
    return topk(mixed, k)


def lazy_candidate_shadow_search(
    data: SyntheticRetrievalData,
    primary: SearchResult,
    sparse: SearchResult | None,
    doc_residual_scores: np.ndarray | None,
    seed: int,
    k: int,
    candidate_topk: int = 10,
    residual_ratio: float = 0.05,
    random_ratio: float = 0.05,
) -> Tuple[SearchResult, Dict[str, float]]:
    """Approximate full-new candidates from a lazily encoded document pool.

    This avoids using the full-new top-k list for document attribution. The
    scheduler may score new-space similarities only for documents selected by
    adapter hits, sparse hits, high adapter residual, and a small random sample.
    """
    rng = np.random.default_rng(seed)
    n_docs = data.old_docs.shape[0]
    topk_primary = min(candidate_topk, primary.indices.shape[1])
    topk_sparse = min(candidate_topk, sparse.indices.shape[1]) if sparse is not None else 0
    residual_docs = np.array([], dtype=int)
    if doc_residual_scores is not None and residual_ratio > 0.0:
        n_residual = max(1, int(round(n_docs * residual_ratio)))
        residual_docs = np.argsort(-doc_residual_scores)[:n_residual]
    random_docs = np.array([], dtype=int)
    if random_ratio > 0.0:
        n_random = max(1, int(round(n_docs * random_ratio)))
        random_docs = rng.choice(n_docs, size=min(n_random, n_docs), replace=False)

    out_indices = np.zeros((data.new_queries.shape[0], k), dtype=int)
    out_scores = np.zeros((data.new_queries.shape[0], k), dtype=np.float64)
    candidate_counts: List[int] = []
    unique_docs: set[int] = set()
    for qi in range(data.new_queries.shape[0]):
        pieces = [primary.indices[qi, :topk_primary], residual_docs, random_docs]
        if sparse is not None:
            pieces.append(sparse.indices[qi, :topk_sparse])
        candidates = np.unique(np.concatenate([piece for piece in pieces if len(piece)]))
        if len(candidates) == 0:
            candidates = np.arange(n_docs)
        candidate_counts.append(int(len(candidates)))
        unique_docs.update(int(x) for x in candidates)
        scores = data.new_queries[qi : qi + 1] @ data.new_docs[candidates].T
        local = topk(scores, k)
        out_indices[qi] = candidates[local.indices[0]]
        out_scores[qi] = local.scores[0]
    result = SearchResult(indices=out_indices, scores=out_scores)
    meta = {
        "lazy_candidate_docs": int(len(unique_docs)),
        "lazy_candidate_ratio": float(len(unique_docs) / n_docs),
        "lazy_mean_candidates_per_query": float(np.mean(candidate_counts)),
        "lazy_mean_candidate_ratio_per_query": float(np.mean(candidate_counts) / n_docs),
        "lazy_candidate_topk": int(candidate_topk),
        "lazy_residual_ratio": float(residual_ratio),
        "lazy_random_ratio": float(random_ratio),
    }
    return result, meta


def refresh_scores_by_policy(
    data: SyntheticRetrievalData,
    primary: SearchResult,
    shadow: SearchResult,
    risk_scores: np.ndarray,
    labels: np.ndarray,
    margin_risk: np.ndarray,
    entropy_risk: np.ndarray,
    disagreement_risk: np.ndarray,
    sparse_risk: np.ndarray,
    gain_scores: np.ndarray,
    oracle_gain: np.ndarray,
    n_clusters: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    doc_features = data.old_docs
    clusters = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit_predict(doc_features)
    policy_scores: Dict[str, np.ndarray] = {}

    random_scores = rng.random(n_clusters)
    policy_scores["random"] = random_scores

    popularity = np.zeros(n_clusters)
    margin_score = np.zeros(n_clusters)
    entropy_score = np.zeros(n_clusters)
    disagreement_score = np.zeros(n_clusters)
    sparse_score = np.zeros(n_clusters)
    residual_score = np.zeros(n_clusters)
    shadow_gain_score = np.zeros(n_clusters)
    learned_gain_score = np.zeros(n_clusters)
    fastfill_uncertainty_score = np.zeros(n_clusters)
    for qi in range(primary.indices.shape[0]):
        hit_clusters = clusters[primary.indices[qi, :10]]
        for c in hit_clusters:
            popularity[c] += data.query_frequency[qi]
            margin_score[c] += margin_risk[qi]
            entropy_score[c] += entropy_risk[qi]
            disagreement_score[c] += disagreement_risk[qi]
            sparse_score[c] += sparse_risk[qi]
            fastfill_uncertainty_score[c] += data.query_frequency[qi] * (
                0.40 * margin_risk[qi]
                + 0.40 * entropy_risk[qi]
                + 0.20 * disagreement_risk[qi]
            )
        for c in clusters[shadow.indices[qi, :10]]:
            shadow_gain_score[c] += data.query_frequency[qi] * oracle_gain[qi]
            learned_gain_score[c] += data.query_frequency[qi] * gain_scores[qi]
    for c in range(n_clusters):
        topic_mix = data.doc_topics[clusters == c]
        if len(topic_mix):
            residual_score[c] = data.topic_drift[topic_mix].mean()

    ric_score = (
        1.00 * unit_scale(shadow_gain_score)
        + 0.80 * unit_scale(margin_score)
        + 0.50 * unit_scale(entropy_score)
        + 0.50 * unit_scale(popularity)
    )

    policy_scores["popularity"] = popularity
    policy_scores["margin"] = margin_score
    policy_scores["entropy"] = entropy_score
    policy_scores["adapter_disagreement"] = disagreement_score
    policy_scores["sparse_overlap"] = sparse_score
    policy_scores["residual"] = residual_score
    policy_scores["fastfill_uncertainty"] = (
        unit_scale(fastfill_uncertainty_score)
        + 0.25 * unit_scale(residual_score)
        + 0.10 * unit_scale(popularity)
    )
    policy_scores["learned_gain"] = learned_gain_score
    policy_scores["ric"] = ric_score
    policy_scores["clusters"] = clusters
    return policy_scores


def observable_drift_utility(features: np.ndarray, column_map: Dict[str, int]) -> np.ndarray:
    """Training-free, no-shadow-index per-query refresh utility (C1).

    Combines only signals observable from the OLD index and LIVE new-model
    queries: rank instability under query perturbation, adapter disagreement
    across alternative maps into the old space, top-k margin collapse, answer
    entropy, query out-of-distribution vs anchors, and adapter residual. It
    never reads the full-new index or any re-encoded document, so it carries no
    oracle information. Weights are fixed (no fitting on oracle labels), which
    is what makes the resulting RIC-Gain variant training-free.
    """
    def col(name: str) -> np.ndarray:
        return unit_scale(features[:, column_map[name]])

    return (
        0.35 * col("candidate_instability")
        + 0.25 * col("adapter_disagreement")
        + 0.15 * col("neg_margin")
        + 0.10 * col("entropy")
        + 0.10 * col("query_ood")
        + 0.05 * col("cell_drift")
    )


def refresh_scores_by_document_policy(
    data: SyntheticRetrievalData,
    primary: SearchResult,
    shadow: SearchResult,
    sparse: SearchResult | None,
    risk_scores: np.ndarray | None,
    margin_risk: np.ndarray,
    entropy_risk: np.ndarray,
    disagreement_risk: np.ndarray,
    sparse_risk: np.ndarray,
    gain_scores: np.ndarray,
    oracle_gain: np.ndarray,
    seed: int,
    doc_residual_scores: np.ndarray | None = None,
    doc_gain_mode: str = "oracle_delta",
    drift_utility: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    if doc_gain_mode not in {"oracle_delta", "lazy_candidate", "measured_lazy", "query_drift", "none"}:
        raise ValueError(f"Unsupported document gain mode: {doc_gain_mode}")
    if doc_gain_mode == "query_drift" and drift_utility is None:
        raise ValueError("query_drift mode requires an observable drift_utility vector")
    rng = np.random.default_rng(seed)
    n_docs = data.old_docs.shape[0]
    policy_scores: Dict[str, np.ndarray] = {
        "random": rng.random(n_docs),
        "popularity": np.zeros(n_docs),
        "margin": np.zeros(n_docs),
        "entropy": np.zeros(n_docs),
        "adapter_disagreement": np.zeros(n_docs),
        "sparse_overlap": np.zeros(n_docs),
        "residual": np.zeros(n_docs),
        "fastfill_uncertainty": np.zeros(n_docs),
        "learned_gain": np.zeros(n_docs),
        "ric": np.zeros(n_docs),
    }
    if doc_residual_scores is None:
        policy_scores["residual"] = data.topic_drift[data.doc_topics].astype(np.float64)
    else:
        policy_scores["residual"] = doc_residual_scores.astype(np.float64)

    attribution = shadow
    use_oracle_gain = doc_gain_mode == "oracle_delta"
    measured_gain: np.ndarray | None = None
    if doc_gain_mode == "lazy_candidate":
        attribution, _ = lazy_candidate_shadow_search(
            data=data,
            primary=primary,
            sparse=sparse,
            doc_residual_scores=doc_residual_scores,
            seed=seed,
            k=10,
        )
    elif doc_gain_mode == "measured_lazy":
        # Encode only a TIGHT adapter+sparse candidate pool in the new space (no
        # full corpus, no random corpus probes, not circular), re-rank within it,
        # and use the REAL measured gain of that partial re-ranking over
        # adapter-only retrieval as the per-query refresh utility. Real signal,
        # unlike the learned-gain baseline which only predicts gain from
        # telemetry. residual/random probes are disabled so the candidate-pool
        # union stays small and the cost story is honest.
        attribution, _lazy_meta = lazy_candidate_shadow_search(
            data=data,
            primary=primary,
            sparse=sparse,
            doc_residual_scores=doc_residual_scores,
            seed=seed,
            k=10,
            residual_ratio=0.0,
            random_ratio=0.0,
        )
        print(
            f"measured_lazy candidate cost: union_ratio={_lazy_meta['lazy_candidate_ratio']:.4f} "
            f"mean_per_query_ratio={_lazy_meta['lazy_mean_candidate_ratio_per_query']:.4f}"
        )
        lazy_recall = per_query_recall(attribution, data.qrels, 10)
        adapter_recall = per_query_recall(primary, data.qrels, 10)
        measured_gain = np.maximum(0.0, lazy_recall - adapter_recall)
    elif doc_gain_mode == "query_drift":
        # Attribute gain to the documents the LIVE new-model query actually
        # retrieves from the OLD index. No full-new index, no re-encoded docs.
        attribution = primary

    for qi in range(primary.indices.shape[0]):
        for doc_id in primary.indices[qi, :10]:
            policy_scores["popularity"][doc_id] += data.query_frequency[qi]
            policy_scores["margin"][doc_id] += margin_risk[qi]
            policy_scores["entropy"][doc_id] += entropy_risk[qi]
            policy_scores["adapter_disagreement"][doc_id] += disagreement_risk[qi]
            policy_scores["sparse_overlap"][doc_id] += sparse_risk[qi]
            policy_scores["fastfill_uncertainty"][doc_id] += data.query_frequency[qi] * (
                0.40 * margin_risk[qi]
                + 0.40 * entropy_risk[qi]
                + 0.20 * disagreement_risk[qi]
            )
        if doc_gain_mode in {"oracle_delta", "lazy_candidate", "measured_lazy", "query_drift"}:
            risk_weight = 1.0
            # query_drift stays training-free: do not fold in the learned risk model.
            if risk_scores is not None and doc_gain_mode != "query_drift":
                risk_weight += 0.25 * float(risk_scores[qi])
            for rank_pos, doc_id in enumerate(attribution.indices[qi, :10]):
                policy_scores["learned_gain"][doc_id] += data.query_frequency[qi] * gain_scores[qi]
                rank_weight = 1.0 + 0.20 * (10 - rank_pos) / 10.0
                if use_oracle_gain:
                    utility = oracle_gain[qi]
                elif doc_gain_mode == "measured_lazy":
                    utility = float(measured_gain[qi]) * risk_weight * rank_weight
                elif doc_gain_mode == "query_drift":
                    utility = float(drift_utility[qi]) * rank_weight
                else:
                    utility = gain_scores[qi] * risk_weight * rank_weight
                policy_scores["ric"][doc_id] += data.query_frequency[qi] * utility

    policy_scores["fastfill_uncertainty"] = (
        unit_scale(policy_scores["fastfill_uncertainty"])
        + 0.25 * unit_scale(policy_scores["residual"])
        + 0.10 * unit_scale(policy_scores["popularity"])
    )
    if doc_gain_mode == "lazy_candidate":
        policy_scores["ric"] = (
            unit_scale(policy_scores["learned_gain"])
            + 0.12 * unit_scale(policy_scores["ric"])
        )
    else:
        policy_scores["ric"] = (
            unit_scale(policy_scores["ric"])
            + 0.20 * unit_scale(policy_scores["margin"])
            + 0.10 * unit_scale(policy_scores["popularity"])
        )
    return policy_scores


def evaluate_refresh_policies(
    data: SyntheticRetrievalData,
    mapped_queries: np.ndarray,
    primary: SearchResult,
    shadow: SearchResult,
    risk_scores: np.ndarray,
    labels: np.ndarray,
    margin_risk: np.ndarray,
    entropy_risk: np.ndarray,
    disagreement_risk: np.ndarray,
    sparse_risk: np.ndarray,
    gain_scores: np.ndarray,
    oracle_gain: np.ndarray,
    budgets: List[float],
    k: int,
    seed: int,
    score_mix: str,
    granularity: str,
    doc_residual_scores: np.ndarray | None = None,
    doc_gain_mode: str = "oracle_delta",
    sparse_result: SearchResult | None = None,
    drift_utility: np.ndarray | None = None,
) -> List[Dict[str, float | str]]:
    if doc_gain_mode == "query_drift" and granularity != "doc":
        raise ValueError("query_drift gain is only defined for document-level refresh")
    if granularity == "cluster":
        n_clusters = 24
        scores = refresh_scores_by_policy(
            data,
            primary,
            shadow,
            risk_scores,
            labels,
            margin_risk,
            entropy_risk,
            disagreement_risk,
            sparse_risk,
            gain_scores,
            oracle_gain,
            n_clusters=n_clusters,
            seed=seed,
        )
        groups = scores.pop("clusters").astype(int)
    elif granularity == "doc":
        scores = refresh_scores_by_document_policy(
            data,
            primary,
            shadow,
            sparse_result,
            risk_scores,
            margin_risk,
            entropy_risk,
            disagreement_risk,
            sparse_risk,
            gain_scores,
            oracle_gain,
            seed=seed,
            doc_residual_scores=doc_residual_scores,
            doc_gain_mode=doc_gain_mode,
            drift_utility=drift_utility,
        )
        groups = np.arange(data.old_docs.shape[0], dtype=int)
    else:
        raise ValueError(f"Unsupported refresh granularity: {granularity}")

    rows: List[Dict[str, float | str]] = []
    for policy, item_scores in scores.items():
        order = np.argsort(-item_scores)
        for budget in budgets:
            n_docs_target = max(1, int(round(data.old_docs.shape[0] * budget)))
            if granularity == "doc":
                selected_docs = order[:n_docs_target]
            else:
                selected_groups: List[int] = []
                selected_docs = np.array([], dtype=int)
                for group_id in order:
                    selected_groups.append(int(group_id))
                    selected_docs = np.flatnonzero(np.isin(groups, selected_groups))
                    if len(selected_docs) >= n_docs_target:
                        break
            result = mixed_refresh_search(data, mapped_queries, selected_docs, k, score_mix)
            metrics = metric_values(result, data.qrels, k)
            rows.append(
                {
                    "policy": policy,
                    "budget": float(budget),
                    "refreshed_docs": int(len(selected_docs)),
                    **metrics,
                }
            )
    return rows


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_synthetic_suite(args: argparse.Namespace) -> Dict[str, object]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.output_dir)
    data = generate_synthetic_data(
        seed=args.seed,
        n_docs=args.n_docs,
        n_queries=args.n_queries,
        dim=args.dim,
        n_topics=args.n_topics,
        relevant_per_query=args.relevant_per_query,
        drift_strength=args.drift_strength,
        hard_topic_fraction=args.hard_topic_fraction,
    )
    k = args.k
    rng = np.random.default_rng(args.seed)
    anchor_idx = rng.choice(args.n_docs, size=min(args.n_anchors, args.n_docs), replace=False)
    new_anchor = data.new_docs[anchor_idx]
    old_anchor = data.old_docs[anchor_idx]

    procrustes = fit_procrustes(new_anchor, old_anchor)
    affine = fit_affine(new_anchor, old_anchor)
    adapters = {"procrustes": procrustes, "affine": affine}

    old_result = cosine_search(data.old_queries, data.old_docs, k)
    full_new_result = cosine_search(data.new_queries, data.new_docs, k)
    mismatch_result = mismatched_space_search(data.new_queries, data.old_docs, k)
    sparse_result = topk(data.sparse_scores, k)
    adapter_results = {
        name: cosine_search(adapter.transform(data.new_queries), data.old_docs, k)
        for name, adapter in adapters.items()
    }

    r001 = {
        "run_id": "R001",
        "dataset": "synthetic",
        "systems": {
            "old_index_old_query": metric_values(old_result, data.qrels, k),
            "full_new_index_oracle": metric_values(full_new_result, data.qrels, k),
            "mismatched_new_query_old_index": metric_values(mismatch_result, data.qrels, k),
        },
    }
    write_json(out / "r001_sanity.json", r001)

    r002 = {
        "run_id": "R002",
        "dataset": "synthetic",
        "systems": {
            name: metric_values(result, data.qrels, k) for name, result in adapter_results.items()
        },
    }
    write_json(out / "r002_adapters.json", r002)

    primary_adapter = adapters[args.primary_adapter]
    primary_result = adapter_results[args.primary_adapter]
    mapped_queries = primary_adapter.transform(data.new_queries)
    alt_results = {name: result for name, result in adapter_results.items() if name != args.primary_adapter}
    alt_results["mismatch"] = mismatch_result
    anchor_pred = primary_adapter.transform(new_anchor)
    anchor_residual = np.linalg.norm(anchor_pred - old_anchor, axis=1)
    features, feature_names = build_features(
        data,
        primary_result,
        alt_results,
        sparse_result,
        mapped_queries,
        new_anchor,
        old_anchor,
        anchor_residual,
        primary_adapter,
        k,
    )
    labels, failure_meta = make_failure_labels(primary_result, full_new_result, data.qrels, k, args)
    train, calib, test = split_indices(args.n_queries, args.seed)

    column_map = {name: i for i, name in enumerate(feature_names)}
    risk_specs = {
        "margin_only": [column_map["neg_margin"]],
        "entropy_only": [column_map["entropy"]],
        "adapter_disagreement_only": [column_map["adapter_disagreement"]],
        "sparse_overlap_only": [column_map["neg_sparse_overlap"]],
        "local_residual_only": [column_map["local_anchor_residual"]],
        "instability_only": [column_map["candidate_instability"]],
        "full_ric": None,
    }
    risk_rows = []
    risk_scores_by_name = {}
    for name, cols in risk_specs.items():
        scores, fit_stats = fit_risk_scores(features, labels, train, cols)
        tau = calibrate_threshold(scores, labels, calib, args.alpha, args.calibration_alpha_factor)
        eval_stats = risk_eval(scores, labels, test, tau)
        risk_scores_by_name[name] = scores
        risk_rows.append(
            {
                "run_id": "R003",
                "risk_model": name,
                "tau": tau,
                "train_auc": fit_stats["auc"],
                "train_ap": fit_stats["ap"],
                **eval_stats,
            }
        )
    append_guarded_ric_row(
        risk_scores_by_name,
        risk_rows,
        labels,
        train,
        calib,
        test,
        args.alpha,
        args.calibration_alpha_factor,
    )
    append_mondrian_ric_rows(
        risk_scores_by_name,
        risk_rows,
        features,
        column_map,
        labels,
        train,
        calib,
        test,
        args.alpha,
        args.calibration_alpha_factor,
    )
    append_calibration_selected_mondrian_row(
        risk_scores_by_name,
        risk_rows,
        features,
        column_map,
        labels,
        train,
        calib,
        test,
        args.alpha,
        args.calibration_alpha_factor,
        args.mondrian_selection_ucb_factor,
        args.mondrian_selection_z,
    )
    r003 = {
        "run_id": "R003",
        "dataset": "synthetic",
        "alpha": args.alpha,
        **failure_meta,
        "failure_rate_all_queries": float(labels.mean()),
        "feature_names": feature_names,
        "splits": {"train": len(train), "calibration": len(calib), "test": len(test)},
        "risk_models": risk_rows,
    }
    write_json(out / "r003_risk_calibration.json", r003)
    write_csv(out / "r003_risk_calibration.csv", risk_rows)

    oracle_recall = per_query_recall(full_new_result, data.qrels, k)
    primary_recall = per_query_recall(primary_result, data.qrels, k)
    oracle_gain = np.maximum(0.0, oracle_recall - primary_recall)
    gain_scores = fit_gain_scores(features, oracle_gain, np.concatenate([train, calib]))
    drift_utility = observable_drift_utility(features, column_map)
    margin_risk = features[:, column_map["neg_margin"]]
    entropy_risk = features[:, column_map["entropy"]]
    disagreement_risk = features[:, column_map["adapter_disagreement"]]
    sparse_risk = features[:, column_map["neg_sparse_overlap"]]
    refresh_rows = evaluate_refresh_policies(
        data=data,
        mapped_queries=mapped_queries,
        primary=primary_result,
        shadow=full_new_result,
        risk_scores=risk_scores_by_name["guarded_ric"],
        labels=labels,
        margin_risk=margin_risk,
        entropy_risk=entropy_risk,
        disagreement_risk=disagreement_risk,
        sparse_risk=sparse_risk,
        gain_scores=gain_scores,
        oracle_gain=oracle_gain,
        budgets=args.refresh_budgets,
        k=k,
        seed=args.seed,
        score_mix=args.refresh_score_mix,
        granularity=args.refresh_granularity,
        doc_gain_mode=args.doc_gain_mode,
        sparse_result=sparse_result,
        drift_utility=drift_utility,
    )
    write_json(out / "r004_refresh_scheduler.json", {"run_id": "R004", "rows": refresh_rows})
    write_csv(out / "r004_refresh_scheduler.csv", refresh_rows)

    summary = {
        "suite": "synthetic_smoke",
        "seed": args.seed,
        "output_dir": str(out),
        "r001": r001,
        "r002": r002,
        "r003": r003,
        "r004": {"run_id": "R004", "rows": refresh_rows},
    }
    write_json(out / "summary.json", summary)
    return summary


BEIR_URLS = {
    "fiqa": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip",
    "nfcorpus": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip",
    "scifact": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
    "trec-covid": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/trec-covid.zip",
    "scidocs": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scidocs.zip",
    "arguana": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/arguana.zip",
    "quora": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/quora.zip",
}


def download_and_extract_beir(dataset: str, data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = data_dir / dataset
    if (dataset_dir / "corpus.jsonl").exists():
        return dataset_dir
    if dataset not in BEIR_URLS:
        raise ValueError(f"Unsupported BEIR dataset: {dataset}")
    zip_path = data_dir / f"{dataset}.zip"
    if not zip_path.exists():
        print(f"Downloading {dataset} from {BEIR_URLS[dataset]}")
        urllib.request.urlretrieve(BEIR_URLS[dataset], zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(data_dir)
    if not (dataset_dir / "corpus.jsonl").exists():
        raise FileNotFoundError(f"Could not find BEIR files after extraction: {dataset_dir}")
    return dataset_dir


def read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def find_qrels_file(dataset_dir: Path) -> Path:
    candidates = [
        dataset_dir / "qrels" / "test.tsv",
        dataset_dir / "qrels" / "dev.tsv",
        dataset_dir / "qrels" / "train.tsv",
    ]
    for path in candidates:
        if path.exists():
            return path
    found = sorted((dataset_dir / "qrels").glob("*.tsv"))
    if not found:
        raise FileNotFoundError(f"No qrels TSV found under {dataset_dir / 'qrels'}")
    return found[0]


def read_qrels(path: Path) -> Dict[str, set[str]]:
    qrels: Dict[str, set[str]] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        for row in reader:
            if len(row) < 3:
                continue
            qid, docid, score = row[0], row[1], row[2]
            try:
                rel = float(score)
            except ValueError:
                continue
            if rel <= 0:
                continue
            qrels.setdefault(qid, set()).add(docid)
    return qrels


def fit_text_embeddings(
    corpus_texts: List[str],
    query_texts: List[str],
    dim: int,
    old_max_features: int,
    new_max_features: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    old_vec = TfidfVectorizer(
        lowercase=True,
        analyzer="char_wb",
        ngram_range=(3, 4),
        min_df=1,
        max_features=old_max_features,
        sublinear_tf=True,
    )
    new_vec = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_features=new_max_features,
        sublinear_tf=True,
    )
    sparse_vec = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 1),
        min_df=1,
        max_features=new_max_features,
        sublinear_tf=True,
    )
    old_doc_sparse = old_vec.fit_transform(corpus_texts)
    old_query_sparse = old_vec.transform(query_texts)
    new_doc_sparse = new_vec.fit_transform(corpus_texts)
    new_query_sparse = new_vec.transform(query_texts)
    sparse_doc = sparse_vec.fit_transform(corpus_texts)
    sparse_query = sparse_vec.transform(query_texts)

    old_components = max(2, min(dim, old_doc_sparse.shape[1] - 1, old_doc_sparse.shape[0] - 1))
    new_components = max(2, min(dim, new_doc_sparse.shape[1] - 1, new_doc_sparse.shape[0] - 1))
    old_svd = TruncatedSVD(n_components=old_components, random_state=seed)
    new_svd = TruncatedSVD(n_components=new_components, random_state=seed + 1)
    old_docs = old_svd.fit_transform(old_doc_sparse)
    old_queries = old_svd.transform(old_query_sparse)
    new_docs = new_svd.fit_transform(new_doc_sparse)
    new_queries = new_svd.transform(new_query_sparse)

    if old_components < dim:
        old_docs = np.pad(old_docs, ((0, 0), (0, dim - old_components)))
        old_queries = np.pad(old_queries, ((0, 0), (0, dim - old_components)))
    if new_components < dim:
        new_docs = np.pad(new_docs, ((0, 0), (0, dim - new_components)))
        new_queries = np.pad(new_queries, ((0, 0), (0, dim - new_components)))

    sparse_scores = (sparse_query @ sparse_doc.T).toarray()
    return (
        l2_normalize(old_docs),
        l2_normalize(new_docs),
        l2_normalize(old_queries),
        l2_normalize(new_queries),
        sparse_scores,
    )


def fit_sparse_scores(corpus_texts: List[str], query_texts: List[str], max_features: int) -> np.ndarray:
    sparse_vec = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 1),
        min_df=1,
        max_features=max_features,
        sublinear_tf=True,
    )
    doc_sparse = sparse_vec.fit_transform(corpus_texts)
    query_sparse = sparse_vec.transform(query_texts)
    return (query_sparse @ doc_sparse.T).toarray()


def model_slug(model_name: str) -> str:
    return model_name.replace("/", "__").replace(":", "_")


def text_hash(texts: List[str]) -> str:
    h = hashlib.sha1()
    for text in texts[:5] + texts[-5:]:
        h.update(text.encode("utf-8", errors="ignore"))
        h.update(b"\0")
    h.update(str(len(texts)).encode("ascii"))
    return h.hexdigest()[:12]


def apply_model_prefix(texts: List[str], model_name: str, role: str) -> List[str]:
    lower = model_name.lower()
    if "intfloat/e5" in lower:
        prefix = "query: " if role == "query" else "passage: "
        return [prefix + text for text in texts]
    if "bge" in lower and role == "query":
        return ["Represent this sentence for searching relevant passages: " + text for text in texts]
    return texts


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def encode_with_transformers(
    texts: List[str],
    model_name: str,
    role: str,
    cache_dir: Path,
    batch_size: int,
    max_length: int,
    device_arg: str,
    local_files_only: bool,
) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{model_slug(model_name)}_{role}_{len(texts)}_{text_hash(texts)}.npz"
    if cache_path.exists() and cache_path.stat().st_size > 0:
        try:
            return np.load(cache_path)["embeddings"]
        except Exception:
            pass

    import torch
    from transformers import AutoModel, AutoTokenizer

    device = resolve_device(device_arg)
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    model = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
    model.to(device)
    model.eval()

    encoded: List[np.ndarray] = []
    prepared = apply_model_prefix(texts, model_name, role)
    with torch.no_grad():
        for start in range(0, len(prepared), batch_size):
            batch = prepared[start : start + batch_size]
            toks = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            toks = {key: value.to(device) for key, value in toks.items()}
            outputs = model(**toks)
            hidden = outputs.last_hidden_state
            mask = toks["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            encoded.append(pooled.cpu().numpy().astype(np.float64))
    embeddings = np.vstack(encoded)
    np.savez_compressed(cache_path, embeddings=embeddings)
    return embeddings


def load_beir_sentence_data(args: argparse.Namespace) -> SyntheticRetrievalData:
    records = load_beir_text_records(args)
    cache_dir = Path(args.embedding_cache_dir) / args.dataset_name
    old_docs = encode_with_transformers(
        records.corpus_texts,
        args.old_model,
        "doc",
        cache_dir,
        args.encode_batch_size,
        args.max_length,
        args.device,
        args.local_files_only,
    )
    old_queries = encode_with_transformers(
        records.query_texts,
        args.old_model,
        "query",
        cache_dir,
        args.encode_batch_size,
        args.max_length,
        args.device,
        args.local_files_only,
    )
    new_docs = encode_with_transformers(
        records.corpus_texts,
        args.new_model,
        "doc",
        cache_dir,
        args.encode_batch_size,
        args.max_length,
        args.device,
        args.local_files_only,
    )
    new_queries = encode_with_transformers(
        records.query_texts,
        args.new_model,
        "query",
        cache_dir,
        args.encode_batch_size,
        args.max_length,
        args.device,
        args.local_files_only,
    )
    sparse_scores = fit_sparse_scores(records.corpus_texts, records.query_texts, args.new_max_features)
    rng = np.random.default_rng(args.seed)
    query_frequency = rng.zipf(a=1.4, size=records.n_queries).astype(np.float64)
    query_frequency /= query_frequency.mean()
    return SyntheticRetrievalData(
        old_docs=l2_normalize(old_docs),
        new_docs=l2_normalize(new_docs),
        old_queries=l2_normalize(old_queries),
        new_queries=l2_normalize(new_queries),
        doc_topics=np.zeros(records.n_docs, dtype=int),
        query_topics=np.zeros(records.n_queries, dtype=int),
        qrels=records.qrels,
        sparse_scores=sparse_scores,
        query_frequency=query_frequency,
        topic_drift=np.ones(1, dtype=np.float64),
    )


def load_beir_text_records(args: argparse.Namespace) -> BeirTextRecords:
    dataset_dir = download_and_extract_beir(args.dataset_name, Path(args.data_dir))
    corpus_rows = read_jsonl(dataset_dir / "corpus.jsonl")
    query_rows = read_jsonl(dataset_dir / "queries.jsonl")
    qrels_by_id = read_qrels(find_qrels_file(dataset_dir))

    query_rows = [row for row in query_rows if row["_id"] in qrels_by_id]
    query_rows = [
        row for row in query_rows if any(doc_id in qrels_by_id[row["_id"]] for doc_id in qrels_by_id[row["_id"]])
    ]
    if args.max_queries and len(query_rows) > args.max_queries:
        rng = np.random.default_rng(args.seed)
        query_rows = list(rng.choice(query_rows, size=args.max_queries, replace=False))

    needed_doc_ids = set()
    for row in query_rows:
        needed_doc_ids.update(qrels_by_id[row["_id"]])
    if args.max_docs and len(corpus_rows) > args.max_docs:
        rng = np.random.default_rng(args.seed)
        relevant_rows = [row for row in corpus_rows if row["_id"] in needed_doc_ids]
        remaining = [row for row in corpus_rows if row["_id"] not in needed_doc_ids]
        keep_remaining = max(0, args.max_docs - len(relevant_rows))
        if keep_remaining < len(remaining):
            remaining = list(rng.choice(remaining, size=keep_remaining, replace=False))
        corpus_rows = relevant_rows + remaining

    doc_ids = [row["_id"] for row in corpus_rows]
    doc_id_to_idx = {doc_id: i for i, doc_id in enumerate(doc_ids)}
    filtered_queries = []
    qrels: List[set[int]] = []
    for row in query_rows:
        rel = {doc_id_to_idx[d] for d in qrels_by_id[row["_id"]] if d in doc_id_to_idx}
        if rel:
            filtered_queries.append(row)
            qrels.append(rel)
    query_rows = filtered_queries
    if not query_rows:
        raise ValueError("No queries with qrels survived filtering")

    corpus_texts = [f"{row.get('title', '')} {row.get('text', '')}".strip() for row in corpus_rows]
    query_texts = [row.get("text", "").strip() for row in query_rows]
    return BeirTextRecords(
        corpus_texts=corpus_texts,
        query_texts=query_texts,
        qrels=qrels,
        n_docs=len(corpus_rows),
        n_queries=len(query_rows),
    )


def load_beir_text_svd_data(args: argparse.Namespace) -> SyntheticRetrievalData:
    records = load_beir_text_records(args)
    old_docs, new_docs, old_queries, new_queries, sparse_scores = fit_text_embeddings(
        corpus_texts=records.corpus_texts,
        query_texts=records.query_texts,
        dim=args.dim,
        old_max_features=args.old_max_features,
        new_max_features=args.new_max_features,
        seed=args.seed,
    )
    rng = np.random.default_rng(args.seed)
    query_frequency = rng.zipf(a=1.4, size=records.n_queries).astype(np.float64)
    query_frequency /= query_frequency.mean()
    placeholder_topics = np.zeros(records.n_docs, dtype=int)
    placeholder_drift = np.ones(1, dtype=np.float64)
    return SyntheticRetrievalData(
        old_docs=old_docs,
        new_docs=new_docs,
        old_queries=old_queries,
        new_queries=new_queries,
        doc_topics=placeholder_topics,
        query_topics=np.zeros(records.n_queries, dtype=int),
        qrels=records.qrels,
        sparse_scores=sparse_scores,
        query_frequency=query_frequency,
        topic_drift=placeholder_drift,
    )


def run_real_suite(args: argparse.Namespace) -> Dict[str, object]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.output_dir)
    if args.suite == "beir_text_svd":
        data = load_beir_text_svd_data(args)
    elif args.suite == "beir_sentence":
        data = load_beir_sentence_data(args)
    else:
        raise ValueError(f"Unsupported real suite: {args.suite}")
    k = args.k
    rng = np.random.default_rng(args.seed)
    anchor_idx = rng.choice(data.old_docs.shape[0], size=min(args.n_anchors, data.old_docs.shape[0]), replace=False)
    new_anchor = data.new_docs[anchor_idx]
    old_anchor = data.old_docs[anchor_idx]

    procrustes = fit_procrustes(new_anchor, old_anchor)
    affine = fit_affine(new_anchor, old_anchor)
    adapters = {"procrustes": procrustes, "affine": affine}

    old_result = cosine_search(data.old_queries, data.old_docs, k)
    full_new_result = cosine_search(data.new_queries, data.new_docs, k)
    mismatch_result = mismatched_space_search(data.new_queries, data.old_docs, k)
    sparse_result = topk(data.sparse_scores, k)
    adapter_results = {
        name: cosine_search(adapter.transform(data.new_queries), data.old_docs, k)
        for name, adapter in adapters.items()
    }

    primary_adapter = adapters[args.primary_adapter]
    mapped_queries = primary_adapter.transform(data.new_queries)
    doc_pred = primary_adapter.transform(data.new_docs)
    doc_residual = np.linalg.norm(doc_pred - data.old_docs, axis=1)
    n_clusters = min(args.real_clusters, max(2, data.old_docs.shape[0] // 25))
    doc_clusters = KMeans(n_clusters=n_clusters, random_state=args.seed, n_init=10).fit_predict(data.old_docs)
    cluster_drift = np.zeros(n_clusters, dtype=np.float64)
    for cluster_id in range(n_clusters):
        mask = doc_clusters == cluster_id
        cluster_drift[cluster_id] = float(doc_residual[mask].mean()) if mask.any() else 0.0
    data.doc_topics = doc_clusters
    data.topic_drift = cluster_drift

    r001 = {
        "run_id": "R001",
        "dataset": args.dataset_name,
        "suite": args.suite,
        "old_model": args.old_model if args.suite == "beir_sentence" else "tfidf_char_svd",
        "new_model": args.new_model if args.suite == "beir_sentence" else "tfidf_word_bigram_svd",
        "systems": {
            "old_index_old_query": metric_values(old_result, data.qrels, k),
            "full_new_index_oracle": metric_values(full_new_result, data.qrels, k),
            "mismatched_new_query_old_index": metric_values(mismatch_result, data.qrels, k),
            "sparse_tfidf": metric_values(sparse_result, data.qrels, k),
        },
        "n_docs": int(data.old_docs.shape[0]),
        "n_queries": int(data.old_queries.shape[0]),
    }
    write_json(out / "r001_sanity.json", r001)

    r002 = {
        "run_id": "R002",
        "dataset": args.dataset_name,
        "suite": args.suite,
        "systems": {
            name: metric_values(result, data.qrels, k) for name, result in adapter_results.items()
        },
    }
    write_json(out / "r002_adapters.json", r002)

    primary_result = adapter_results[args.primary_adapter]
    alt_results = {name: result for name, result in adapter_results.items() if name != args.primary_adapter}
    alt_results["mismatch"] = mismatch_result
    anchor_pred = primary_adapter.transform(new_anchor)
    anchor_residual = np.linalg.norm(anchor_pred - old_anchor, axis=1)
    features, feature_names = build_features(
        data,
        primary_result,
        alt_results,
        sparse_result,
        mapped_queries,
        new_anchor,
        old_anchor,
        anchor_residual,
        primary_adapter,
        k,
    )
    labels, failure_meta = make_failure_labels(primary_result, full_new_result, data.qrels, k, args)
    train, calib, test = split_indices(data.old_queries.shape[0], args.seed)
    column_map = {name: i for i, name in enumerate(feature_names)}
    risk_specs = {
        "margin_only": [column_map["neg_margin"]],
        "entropy_only": [column_map["entropy"]],
        "adapter_disagreement_only": [column_map["adapter_disagreement"]],
        "sparse_overlap_only": [column_map["neg_sparse_overlap"]],
        "local_residual_only": [column_map["local_anchor_residual"]],
        "instability_only": [column_map["candidate_instability"]],
        "full_ric": None,
    }
    risk_rows = []
    risk_scores_by_name = {}
    for name, cols in risk_specs.items():
        scores, fit_stats = fit_risk_scores(features, labels, train, cols)
        tau = calibrate_threshold(scores, labels, calib, args.alpha, args.calibration_alpha_factor)
        eval_stats = risk_eval(scores, labels, test, tau)
        risk_scores_by_name[name] = scores
        risk_rows.append(
            {
                "run_id": "R003",
                "risk_model": name,
                "tau": tau,
                "train_auc": fit_stats["auc"],
                "train_ap": fit_stats["ap"],
                **eval_stats,
            }
        )
    append_guarded_ric_row(
        risk_scores_by_name,
        risk_rows,
        labels,
        train,
        calib,
        test,
        args.alpha,
        args.calibration_alpha_factor,
    )
    append_mondrian_ric_rows(
        risk_scores_by_name,
        risk_rows,
        features,
        column_map,
        labels,
        train,
        calib,
        test,
        args.alpha,
        args.calibration_alpha_factor,
    )
    append_calibration_selected_mondrian_row(
        risk_scores_by_name,
        risk_rows,
        features,
        column_map,
        labels,
        train,
        calib,
        test,
        args.alpha,
        args.calibration_alpha_factor,
        args.mondrian_selection_ucb_factor,
        args.mondrian_selection_z,
    )
    r003 = {
        "run_id": "R003",
        "dataset": args.dataset_name,
        "alpha": args.alpha,
        **failure_meta,
        "failure_rate_all_queries": float(labels.mean()),
        "feature_names": feature_names,
        "splits": {"train": len(train), "calibration": len(calib), "test": len(test)},
        "risk_models": risk_rows,
    }
    write_json(out / "r003_risk_calibration.json", r003)
    write_csv(out / "r003_risk_calibration.csv", risk_rows)

    oracle_recall = per_query_recall(full_new_result, data.qrels, k)
    primary_recall = per_query_recall(primary_result, data.qrels, k)
    oracle_gain = np.maximum(0.0, oracle_recall - primary_recall)
    gain_scores = fit_gain_scores(features, oracle_gain, np.concatenate([train, calib]))
    drift_utility = observable_drift_utility(features, column_map)
    margin_risk = features[:, column_map["neg_margin"]]
    entropy_risk = features[:, column_map["entropy"]]
    disagreement_risk = features[:, column_map["adapter_disagreement"]]
    sparse_risk = features[:, column_map["neg_sparse_overlap"]]
    refresh_rows = evaluate_refresh_policies(
        data=data,
        mapped_queries=mapped_queries,
        primary=primary_result,
        shadow=full_new_result,
        risk_scores=risk_scores_by_name["guarded_ric"],
        labels=labels,
        margin_risk=margin_risk,
        entropy_risk=entropy_risk,
        disagreement_risk=disagreement_risk,
        sparse_risk=sparse_risk,
        gain_scores=gain_scores,
        oracle_gain=oracle_gain,
        budgets=args.refresh_budgets,
        k=k,
        seed=args.seed,
        score_mix=args.refresh_score_mix,
        granularity=args.refresh_granularity,
        doc_residual_scores=doc_residual,
        doc_gain_mode=args.doc_gain_mode,
        sparse_result=sparse_result,
        drift_utility=drift_utility,
    )
    write_json(out / "r004_refresh_scheduler.json", {"run_id": "R004", "rows": refresh_rows})
    write_csv(out / "r004_refresh_scheduler.csv", refresh_rows)

    summary = {
        "suite": args.suite,
        "dataset": args.dataset_name,
        "seed": args.seed,
        "output_dir": str(out),
        "r001": r001,
        "r002": r002,
        "r003": r003,
        "r004": {"run_id": "R004", "rows": refresh_rows},
    }
    write_json(out / "summary.json", summary)
    return summary


def print_brief(summary: Dict[str, object]) -> None:
    r001 = summary["r001"]["systems"]  # type: ignore[index]
    r002 = summary["r002"]["systems"]  # type: ignore[index]
    r003 = summary["r003"]["risk_models"]  # type: ignore[index]
    print(f"RIC bridge completed: {summary.get('suite', 'unknown')}")
    print("R001 full_new recall:", round(r001["full_new_index_oracle"]["recall@10"], 4))
    print("R001 mismatch recall:", round(r001["mismatched_new_query_old_index"]["recall@10"], 4))
    for name, metrics in r002.items():
        print(f"R002 {name} recall:", round(metrics["recall@10"], 4))
    for row in r003:
        print(
            "R003",
            row["risk_model"],
            "coverage",
            round(row["coverage"], 4),
            "violation",
            round(row["violation"], 4),
        )
    print("Outputs:", summary["output_dir"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RIC experiment bridge pilots.")
    parser.add_argument(
        "--suite",
        default="synthetic_smoke",
        choices=["synthetic_smoke", "beir_text_svd", "beir_sentence"],
    )
    parser.add_argument("--output-dir", default="results/bridge/synthetic_smoke")
    parser.add_argument("--dataset-name", default="scifact", choices=sorted(BEIR_URLS))
    parser.add_argument("--data-dir", default="data/beir")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-docs", type=int, default=2400)
    parser.add_argument("--n-queries", type=int, default=360)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--n-topics", type=int, default=12)
    parser.add_argument("--n-anchors", type=int, default=420)
    parser.add_argument("--max-docs", type=int, default=0)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--old-max-features", type=int, default=20000)
    parser.add_argument("--new-max-features", type=int, default=30000)
    parser.add_argument("--old-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--new-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--embedding-cache-dir", default="data/embeddings")
    parser.add_argument("--encode-batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--real-clusters", type=int, default=24)
    parser.add_argument("--relevant-per-query", type=int, default=12)
    parser.add_argument("--drift-strength", type=float, default=0.85)
    parser.add_argument("--hard-topic-fraction", type=float, default=0.33)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--primary-adapter", default="procrustes", choices=["procrustes", "affine"])
    parser.add_argument(
        "--failure-label",
        default="oracle_delta_recall",
        choices=["oracle_delta_recall", "oracle_delta_ndcg", "absolute_recall"],
    )
    parser.add_argument("--oracle-delta", type=float, default=0.05)
    parser.add_argument("--failure-recall-threshold", type=float, default=0.10)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--calibration-alpha-factor", type=float, default=0.50)
    parser.add_argument(
        "--mondrian-selection-ucb-factor",
        type=float,
        default=0.25,
        help="Calibration-only Mondrian selection accepts candidates whose Wilson upper bound is below alpha times this factor.",
    )
    parser.add_argument(
        "--mondrian-selection-z",
        type=float,
        default=1.64,
        help="z value for the one-sided Wilson upper bound used in calibration-only Mondrian selection.",
    )
    parser.add_argument(
        "--refresh-score-mix",
        default="zblend",
        choices=["replace", "zmax", "zblend"],
    )
    parser.add_argument(
        "--refresh-granularity",
        default="doc",
        choices=["cluster", "doc"],
    )
    parser.add_argument(
        "--doc-gain-mode",
        default="oracle_delta",
        choices=["oracle_delta", "lazy_candidate", "measured_lazy", "query_drift", "none"],
        help=(
            "Document-level gain signal for the doc refresh scheduler. "
            "oracle_delta = full-new shadow oracle (upper-bound reference); "
            "lazy_candidate = lazily re-encoded candidate pool with predicted gain; "
            "measured_lazy = bounded candidate-pool re-encoding with REAL measured "
            "partial-pool gain (deployable headline method); "
            "query_drift = training-free, no-encoding estimator from observable "
            "query drift; none = ablate the gain term."
        ),
    )
    parser.add_argument(
        "--refresh-budgets",
        type=float,
        nargs="+",
        default=[0.01, 0.05, 0.10, 0.20, 0.50],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.suite == "synthetic_smoke":
        summary = run_synthetic_suite(args)
    else:
        summary = run_real_suite(args)
    print_brief(summary)


if __name__ == "__main__":
    main()
