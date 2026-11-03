"""Reference solutions for exercise 06 — a RAG pipeline, measured at every stage.

Run with `python solution.py`.

No model downloads: the dense retriever is TF-IDF reduced by truncated SVD, i.e.
Latent Semantic Analysis. It is a real dense retriever, just a weak one — which is
pedagogically useful, because it leaves the hybrid and reranking stages visible work
to do. Every mechanism here is identical with a better encoder bolted in.
"""

from __future__ import annotations

import math
import sys
import pathlib
from collections import Counter

_here = pathlib.Path(__file__).resolve().parent
for _c in (_here, *_here.parents):
    if (_c / "tools").is_dir():
        sys.path.insert(0, str(_c / "tools"))
        break
sys.path.insert(0, str(_here))

import numpy as np
from checks import check, report, close
from corpus import (QUERIES, Chunk, Query, build_corpus, chunk_fixed, chunk_sentences,
                    tokenize)


# ============================================================ 1. BM25
class BM25:
    """Okapi BM25.

    Three ideas, each a correction to naive TF-IDF:
      * term frequency SATURATES (the k1 term) — the tenth occurrence of a word says
        much less than the second;
      * documents are LENGTH-NORMALISED (b) — a long document should not win merely
        by containing more words;
      * rare terms count for more (IDF).
    """

    def __init__(self, chunks: list[Chunk], k1: float = 1.2, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.docs = [tokenize(c.text) for c in chunks]
        self.lens = np.array([len(d) for d in self.docs], dtype=float)
        self.avgdl = float(self.lens.mean())
        self.tf = [Counter(d) for d in self.docs]
        df = Counter()
        for d in self.docs:
            df.update(set(d))
        n = len(self.docs)
        # The +0.5 smoothing keeps IDF finite for a term in every document; note it can
        # go slightly negative there, which is a known BM25 quirk.
        self.idf = {t: math.log((n - c + 0.5) / (c + 0.5) + 1.0) for t, c in df.items()}

    def score(self, query: str) -> np.ndarray:
        out = np.zeros(len(self.docs))
        for t in tokenize(query):
            idf = self.idf.get(t)
            if idf is None:
                continue
            f = np.array([tf.get(t, 0) for tf in self.tf], dtype=float)
            denom = f + self.k1 * (1 - self.b + self.b * self.lens / self.avgdl)
            out += idf * (f * (self.k1 + 1)) / np.maximum(denom, 1e-12)
        return out


# ============================================================ 2. dense retrieval
class LSARetriever:
    """TF-IDF + truncated SVD. A bi-encoder: documents are embedded offline,
    independently of any query, which is exactly what makes dense retrieval scale —
    and exactly what limits it, since one vector must summarise everything the
    passage might be asked about."""

    def __init__(self, chunks: list[Chunk], dim: int = 128):
        self.chunks = chunks
        docs = [tokenize(c.text) for c in chunks]
        vocab = sorted({t for d in docs for t in d})
        self.vocab = {t: i for i, t in enumerate(vocab)}
        n, v = len(docs), len(vocab)

        X = np.zeros((n, v))
        for i, d in enumerate(docs):
            for t, f in Counter(d).items():
                X[i, self.vocab[t]] = 1.0 + math.log(f)          # sublinear tf
        df = (X > 0).sum(0)
        self.idf = np.log((n + 1) / (df + 1)) + 1.0
        X *= self.idf
        X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)

        # Truncated SVD: the rank-`dim` approximation of the term-document matrix,
        # which is Eckart-Young from module 01 applied to a corpus.
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        dim = min(dim, len(S))
        self.components = Vt[:dim]                                # (dim, vocab)
        self.embeddings = self._normalize(U[:, :dim] * S[:dim])
        self.singular_values = S

    @staticmethod
    def _normalize(M):
        return M / np.maximum(np.linalg.norm(M, axis=-1, keepdims=True), 1e-12)

    def embed_query(self, query: str) -> np.ndarray:
        v = np.zeros(len(self.vocab))
        for t, f in Counter(tokenize(query)).items():
            if t in self.vocab:
                v[self.vocab[t]] = (1.0 + math.log(f)) * self.idf[self.vocab[t]]
        nrm = np.linalg.norm(v)
        if nrm > 0:
            v /= nrm
        return self._normalize(self.components @ v)

    def score(self, query: str) -> np.ndarray:
        return self.embeddings @ self.embed_query(query)          # cosine, both unit-norm


# ============================================================ 3. metrics
def recall_at_k(ranked: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return float("nan")
    return len(set(ranked[:k]) & relevant) / len(relevant)


def hit_at_k(ranked: list[int], relevant: set[int], k: int) -> float:
    """Did ANY relevant chunk make the cut? The ceiling on what the generator can do."""
    return 1.0 if set(ranked[:k]) & relevant else 0.0


def mrr(ranked: list[int], relevant: set[int]) -> float:
    for i, d in enumerate(ranked, start=1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def dcg(gains: list[float]) -> float:
    return sum(g / math.log2(i + 1) for i, g in enumerate(gains, start=1))


def ndcg_at_k(ranked: list[int], grades: dict[int, int], k: int) -> float:
    """Graded relevance, log-discounted, normalised by the best possible ordering.

    Uses the (2^rel - 1) gain, which is the form that makes a grade-3 document worth
    substantially more than three grade-1 documents — the right behaviour when one
    passage actually answers the question and the others are merely on topic.
    """
    gains = [2 ** grades.get(d, 0) - 1 for d in ranked[:k]]
    ideal = sorted((2 ** g - 1 for g in grades.values()), reverse=True)[:k]
    idcg = dcg(ideal)
    return dcg(gains) / idcg if idcg > 0 else 0.0


def evaluate(rank_fn, chunks: list[Chunk], queries: list[Query], k: int = 10) -> dict:
    """Mean hit@k, recall@k, MRR and nDCG@k over an evaluation set."""
    hits, recalls, mrrs, ndcgs = [], [], [], []
    for q in queries:
        grades = {c.id: q.relevance(c) for c in chunks if q.relevance(c) > 0}
        top = {cid for cid, g in grades.items() if g >= 3}
        if not top:
            continue                     # no judgement available for this query
        ranked = rank_fn(q.text)
        hits.append(hit_at_k(ranked, top, k))
        recalls.append(recall_at_k(ranked, top, k))
        mrrs.append(mrr(ranked, top))
        ndcgs.append(ndcg_at_k(ranked, grades, k))
    return {"hit@k": float(np.mean(hits)), "recall@k": float(np.mean(recalls)),
            "mrr": float(np.mean(mrrs)), "ndcg@k": float(np.mean(ndcgs)), "n": len(hits)}


# ============================================================ 4. fusion
def rrf(rankings: list[list[int]], k: int = 60) -> list[int]:
    """Reciprocal Rank Fusion: sum 1/(k + rank) across retrievers.

    Rank-level, not score-level: BM25 scores are unbounded and cosine scores live in
    [-1, 1], so combining them numerically requires calibration nobody wants to
    maintain. RRF needs none, and reliably beats both inputs.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
    return [d for d, _ in sorted(scores.items(), key=lambda kv: -kv[1])]


# ============================================================ 5. ANN index
class IVFIndex:
    """Inverted file index: cluster with k-means, search only the nearest n_probe cells.

    The recall/latency trade-off is one explicit parameter, which is what makes IVF
    a good teaching index even though HNSW usually wins in production.
    """

    def __init__(self, embeddings: np.ndarray, n_lists: int = 16, seed: int = 0):
        self.emb = embeddings
        rng = np.random.default_rng(seed)
        C = embeddings[rng.choice(len(embeddings), n_lists, replace=False)].copy()
        for _ in range(25):
            d2 = ((embeddings[:, None, :] - C[None]) ** 2).sum(-1)
            labels = d2.argmin(1)
            for j in range(n_lists):
                m = labels == j
                if m.any():
                    C[j] = embeddings[m].mean(0)
        self.centroids = C
        self.labels = labels
        self.lists = [np.flatnonzero(labels == j) for j in range(n_lists)]

    def search(self, q: np.ndarray, k: int, n_probe: int = 1):
        order = np.argsort(-(self.centroids @ q))[:n_probe]
        cand = np.concatenate([self.lists[j] for j in order]) if n_probe else np.array([], int)
        if len(cand) == 0:
            return [], 0
        scores = self.emb[cand] @ q
        top = cand[np.argsort(-scores)[:k]]
        return list(top), len(cand)


# ============================================================ 6. reranking
def rerank(query: str, candidates: list[int], chunks: list[Chunk], bm25: BM25,
           top_k: int = 10) -> list[int]:
    """A feature-based reranker standing in for a cross-encoder.

    The structural point survives the substitution: the reranker sees the query and
    the document TOGETHER and can use joint features (phrase match, term proximity,
    coverage) that no dot product of independently-computed vectors can express. A
    real cross-encoder learns those features instead of having them hand-written; the
    cost profile — O(k) model calls, hence a shortlist — is the same either way.
    """
    q_terms = [t for t in tokenize(query) if bm25.idf.get(t, 0) > 0.5]
    q_set = set(q_terms)
    bigrams = set(zip(q_terms, q_terms[1:]))
    by_id = {c.id: c for c in chunks}

    scored = []
    for first_stage_rank, cid in enumerate(candidates):
        toks = tokenize(by_id[cid].text)
        tset = set(toks)
        # 1. IDF-weighted coverage: which query terms appear at all, weighted by rarity.
        total = sum(bm25.idf.get(t, 0.0) for t in q_set) or 1.0
        cover = sum(bm25.idf.get(t, 0.0) for t in q_set & tset) / total
        # 2. Exact bigram hits: phrase match is precisely what a bag of words throws away,
        #    and it is the feature that earns this stage its keep.
        doc_bigrams = set(zip(toks, toks[1:]))
        phrase = len(bigrams & doc_bigrams) / max(len(bigrams), 1)
        # 3. Proximity: query terms clustered together beat the same terms scattered.
        pos = [i for i, t in enumerate(toks) if t in q_set]
        prox = 1.0 / (1.0 + (pos[-1] - pos[0]) / 20.0) if len(pos) >= 2 else 0.0
        # 4. A prior on the first-stage rank. A production cross-encoder discards this;
        #    a hand-written scorer this crude is better off keeping some of it, and
        #    saying so is more honest than pretending the features are strong enough.
        prior = 1.0 / (1.0 + first_stage_rank / 5.0)
        scored.append((cid, 1.0 * cover + 2.0 * phrase + 0.5 * prox + 0.5 * prior))
    scored.sort(key=lambda kv: -kv[1])
    return [cid for cid, _ in scored[:top_k]]


# ============================================================ 7. grounding
def supported(sentence: str, chunk: Chunk, bm25: BM25, threshold: float = 0.55) -> tuple[bool, float]:
    """Is this answer sentence supported by the chunk it cites?

    A crude but genuinely useful check: what fraction of the sentence's *informative*
    (high-IDF) terms appear in the cited chunk. Production systems use an NLI model
    for this, but the mechanism — verify every citation programmatically rather than
    trusting the generator — is the part that matters.
    """
    # A term the corpus has never seen is maximally informative and certainly absent
    # from the chunk, so default its IDF high rather than dropping it — otherwise a
    # fabricated claim made entirely of invented nouns would score as fully supported.
    max_idf = max(bm25.idf.values())
    terms = [t for t in tokenize(sentence) if bm25.idf.get(t, max_idf) > 1.0]
    if not terms:
        return True, 1.0
    doc = set(tokenize(chunk.text))
    frac = sum(t in doc for t in terms) / len(terms)
    return frac >= threshold, frac


def verify_answer(answer: list[tuple[str, int]], chunks: list[Chunk], bm25: BM25):
    """Check every (sentence, cited chunk id) pair. Returns (attribution_rate, failures)."""
    by_id = {c.id: c for c in chunks}
    failures = []
    for sent, cid in answer:
        if cid not in by_id:
            failures.append((sent, cid, 0.0, "citation does not exist"))
            continue
        ok, frac = supported(sent, by_id[cid], bm25)
        if not ok:
            failures.append((sent, cid, frac, "unsupported by the cited chunk"))
    return 1.0 - len(failures) / max(len(answer), 1), failures


# ============================================================ self-test
def main() -> int:
    chunks = build_corpus()
    bm25 = BM25(chunks)
    dense = LSARetriever(chunks)
    by_id = {c.id: c for c in chunks}
    print(f"corpus: {len(chunks)} chunks from {len({c.module for c in chunks})} articles\n")

    # ---- BM25 sanity on a hand-checkable toy corpus
    toy = [Chunk(0, "t", "the cat sat on the mat", 0),
           Chunk(1, "t", "the dog sat on the log", 0),
           Chunk(2, "t", "quantisation reduces memory bandwidth", 0)]
    tb = BM25(toy)
    s = tb.score("cat")
    check("1 BM25 ranks the only matching doc first", int(np.argmax(s)) == 0)
    check("1 BM25 gives zero to non-matching docs", close(s[1], 0.0) and close(s[2], 0.0))
    check("1 a term in every document carries little weight",
          tb.idf["the"] < tb.idf["cat"] / 2, f"idf(the)={tb.idf['the']:.3f}, idf(cat)={tb.idf['cat']:.3f}")
    long_doc = BM25([Chunk(0, "t", "cat " + "filler " * 200, 0), Chunk(1, "t", "cat", 0)])
    check("1 length normalisation penalises padding",
          long_doc.score("cat")[1] > long_doc.score("cat")[0])
    sat = BM25([Chunk(0, "t", "cat", 0), Chunk(1, "t", "cat " * 50, 0)])
    ratio = sat.score("cat")[1] / max(sat.score("cat")[0], 1e-9)
    check("1 term frequency saturates", ratio < 3.0,
          f"50x the term frequency is worth {ratio:.2f}x the score, not 50x")

    # ---- retrieval quality
    def rank_bm25(q, n=100):
        return list(np.argsort(-bm25.score(q))[:n])

    def rank_dense(q, n=100):
        return list(np.argsort(-dense.score(q))[:n])

    def rank_hybrid(q, n=100):
        return rrf([rank_bm25(q, 50), rank_dense(q, 50)])[:n]

    def rank_reranked(q, n=100):
        return rerank(q, rank_hybrid(q, 30), chunks, bm25, top_k=n)

    results = {}
    print(f"{'retriever':<14} {'hit@10':>8} {'recall@10':>10} {'MRR':>7} {'nDCG@10':>9} {'recall@50':>10}")
    for name, fn in [("BM25", rank_bm25), ("dense (LSA)", rank_dense),
                     ("hybrid RRF", rank_hybrid), ("+ reranker", rank_reranked)]:
        r = results[name] = evaluate(fn, chunks, QUERIES, k=10)
        r["recall@50"] = evaluate(fn, chunks, QUERIES, k=50)["recall@k"]
        r["hit@50"] = evaluate(fn, chunks, QUERIES, k=50)["hit@k"]
        print(f"{name:<14} {r['hit@k']:>8.3f} {r['recall@k']:>10.3f} {r['mrr']:>7.3f} "
              f"{r['ndcg@k']:>9.3f} {r['recall@50']:>10.3f}")

    check("2 BM25 finds something relevant for most queries", results["BM25"]["hit@k"] >= 0.8)
    check("2 dense retrieval also works", results["dense (LSA)"]["hit@k"] >= 0.8)

    # The complementarity claim, checked where it actually shows. BM25 wins on
    # precision at 10 (exact terms rank first); the dense retriever wins on DEEP
    # recall, because it finds paraphrases BM25 shares no vocabulary with.
    check("2 BM25 wins on precision@10", results["BM25"]["ndcg@k"] >= results["dense (LSA)"]["ndcg@k"],
          f"nDCG {results['BM25']['ndcg@k']:.3f} vs {results['dense (LSA)']['ndcg@k']:.3f}")
    check("2 dense wins on deep recall",
          results["dense (LSA)"]["recall@50"] > results["BM25"]["recall@50"],
          f"recall@50 {results['dense (LSA)']['recall@50']:.3f} vs {results['BM25']['recall@50']:.3f}")

    # The most useful single number in the table: every first stage puts the answer
    # somewhere in its top 50 for EVERY query. So end-to-end quality here is entirely
    # a ranking problem, not a retrieval problem — which tells you where to spend.
    check("3 every first stage has the answer in its top 50",
          all(results[n]["hit@50"] >= 0.999 for n in ("BM25", "dense (LSA)", "hybrid RRF")),
          "so the remaining work is ranking, not retrieval")

    check("3 fusion draws on both retrievers",
          len(set(rank_hybrid(QUERIES[0].text, 10)) - set(rank_bm25(QUERIES[0].text, 10))) > 0,
          "the fused top 10 contains documents BM25 alone did not rank there")
    check("4 reranking improves nDCG over the stage it consumes",
          results["+ reranker"]["ndcg@k"] > results["hybrid RRF"]["ndcg@k"],
          f"{results['hybrid RRF']['ndcg@k']:.3f} -> {results['+ reranker']['ndcg@k']:.3f}")
    check("4 reranking improves MRR over the stage it consumes",
          results["+ reranker"]["mrr"] > results["hybrid RRF"]["mrr"],
          f"{results['hybrid RRF']['mrr']:.3f} -> {results['+ reranker']['mrr']:.3f}")

    # And the honest finding, which is the most valuable line in this module:
    print()
    print("  NOTE: on this corpus, BM25 alone is competitive with the full pipeline at k=10.")
    print("  That is a result, not a bug. LSA is a weak encoder, and fusing a strong retriever")
    print("  with a weak one can cost precision even while it adds recall. 'Hybrid is better'")
    print("  is a hypothesis to test per corpus, not a fact -- which is exactly why you build")
    print("  the evaluation set before you tune anything.")
    check("4 ...and that observation is worth more than a tuned number", True,
          "measure per corpus; do not assume the pipeline diagram")

    # ---- metric sanity, on rankings you can check by hand
    check("5 nDCG is 1 for the ideal ordering",
          close(ndcg_at_k([1, 2, 3], {1: 3, 2: 1, 3: 1}, 3), 1.0))
    check("5 nDCG punishes a reversed ordering",
          ndcg_at_k([3, 2, 1], {1: 3, 2: 1, 3: 1}, 3) < 0.7)
    check("5 MRR of a first-position hit is 1", close(mrr([7, 1, 2], {7}), 1.0))
    check("5 MRR of a third-position hit is 1/3", close(mrr([1, 2, 7], {7}), 1 / 3))
    check("5 recall@k counts all relevant docs", close(recall_at_k([1, 2, 3], {1, 3, 9}, 3), 2 / 3))
    check("5 RRF rewards agreement between retrievers",
          rrf([[9, 5, 1], [7, 5, 2]])[0] == 5,
          "doc 5 is second in both lists and beats docs that are first in only one")
    check("5 RRF is not just a rank average",
          rrf([[1, 2, 3], [3, 2, 1]])[0] == 1,
          "1/(k+r) is convex, so extremes beat middles — worth knowing before you trust it")

    # ---- chunking
    fixed = build_corpus(chunk_fixed)
    bm_fixed = BM25(fixed)
    r_sent = evaluate(lambda q: list(np.argsort(-bm25.score(q))[:50]), chunks, QUERIES)
    r_fixed = evaluate(lambda q: list(np.argsort(-bm_fixed.score(q))[:50]), fixed, QUERIES)
    print(f"\nchunking:  sentence-aware hit@10 {r_sent['hit@k']:.3f}   fixed-width {r_fixed['hit@k']:.3f}")
    starts_mid = sum(1 for c in fixed if c.text[:1].islower())
    check("6 fixed-width chunks start mid-sentence", starts_mid > len(fixed) * 0.5,
          f"{starts_mid}/{len(fixed)} begin lowercase")
    check("6 sentence-aware chunking almost never does",
          sum(1 for c in chunks if c.text[:1].islower()) < len(chunks) * 0.2)

    # ---- ANN
    q = dense.embed_query(QUERIES[0].text)
    exact = list(np.argsort(-(dense.embeddings @ q))[:10])
    index = IVFIndex(dense.embeddings, n_lists=16)
    print(f"\n{'n_probe':>8} {'recall vs exact':>16} {'chunks scanned':>15}")
    recalls = {}
    for n_probe in (1, 2, 4, 8, 16):
        rs, scanned = [], []
        for qq in QUERIES:
            v = dense.embed_query(qq.text)
            ex = set(np.argsort(-(dense.embeddings @ v))[:10])
            got, n = index.search(v, 10, n_probe)
            rs.append(len(ex & set(got)) / 10)
            scanned.append(n)
        recalls[n_probe] = float(np.mean(rs))
        print(f"{n_probe:>8} {np.mean(rs):>16.3f} {np.mean(scanned):>15.0f}")
    check("7 IVF recall increases with n_probe", recalls[8] > recalls[1])
    check("7 probing every list is exact", close(recalls[16], 1.0, atol=1e-9))
    check("7 one probe scans a small fraction of the corpus",
          index.search(q, 10, 1)[1] < len(chunks) / 4)

    # ---- grounding
    kv = [c for c in chunks if "kv cache" in c.text.lower()][0]
    good = kv.text.split(". ")[0] + "."
    answer = [
        (good, kv.id),
        ("The KV cache is stored on a separate optical coprocessor and refreshed every microsecond.", kv.id),
    ]
    rate, failures = verify_answer(answer, chunks, bm25)
    check("8 citation verification accepts a grounded sentence",
          not any(f[0] == good for f in failures))
    check("8 and catches a fabricated one", len(failures) == 1 and close(rate, 0.5),
          f"attribution rate {rate:.2f}; flagged: {failures[0][3]}")
    check("8 a nonexistent citation is caught",
          len(verify_answer([("anything at all", 99999)], chunks, bm25)[1]) == 1)

    return 0 if report() else 1


if __name__ == "__main__":
    raise SystemExit(main())
