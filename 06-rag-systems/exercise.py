# %% [markdown]
# # Exercise 06 — A RAG Pipeline, Measured at Every Stage
#
# **The corpus is this course.** You will retrieve over the ten articles in this repository,
# which has a useful property: you already know the right answers, so a bad retrieval is obvious
# on sight rather than requiring a labelling effort.
#
# Nothing here downloads a model. The dense retriever is TF-IDF reduced by truncated SVD — Latent
# Semantic Analysis, a real bi-encoder, just a weak one. That weakness is deliberate: it leaves the
# hybrid and reranking stages visible work to do, and every mechanism is identical with a stronger
# encoder dropped in.
#
# Eight parts, in pipeline order. The discipline being taught is the one that matters most in a
# RAG interview: **measure each stage separately**, because "the answers are bad" is not a
# diagnosis.

# %%
import sys
import pathlib

_here = pathlib.Path.cwd()
for _c in (_here, *_here.parents):
    if (_c / "tools").is_dir():
        sys.path.insert(0, str(_c / "tools"))
        break
sys.path.insert(0, str(_here if (_here / "corpus.py").exists() else _here / "06-rag-systems"))

import math
from collections import Counter

import numpy as np
from checks import check, report, close
from corpus import QUERIES, Chunk, build_corpus, chunk_fixed, chunk_sentences, tokenize

chunks = build_corpus()
by_id = {c.id: c for c in chunks}
print(f"{len(chunks)} chunks from {len({c.module for c in chunks})} articles, "
      f"mean {sum(len(c.text) for c in chunks) / len(chunks):.0f} chars")
print(f"{len(QUERIES)} evaluation queries with graded relevance judgements\n")
for q in QUERIES[:4]:
    print(f"  {q.text}\n     -> {q.module}, needs one of {q.phrases}")

# %% [markdown]
# ---
# ## Part 1 — BM25
#
# $$\mathrm{BM25}(q,d) = \sum_{t \in q}\mathrm{IDF}(t)\cdot\frac{f(t,d)(k_1+1)}{f(t,d)+k_1\left(1-b+b\frac{|d|}{\text{avgdl}}\right)}$$
#
# with $\mathrm{IDF}(t) = \ln\!\left(\frac{N-n_t+0.5}{n_t+0.5}+1\right)$, $k_1=1.2$, $b=0.75$.
#
# Before implementing, answer: what does $k_1$ control, what does $b$ control, and what happens at
# $b=0$? The checks below test each of those behaviours on corpora small enough to verify by hand.

# %%
class BM25:
    def __init__(self, chunks: list[Chunk], k1: float = 1.2, b: float = 0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        self.docs = [tokenize(c.text) for c in chunks]
        # TODO: document lengths, average length, per-document term counts, and self.idf
        raise NotImplementedError

    def score(self, query: str) -> np.ndarray:
        """Score every chunk against `query`. Returns an array of len(chunks)."""
        # TODO
        raise NotImplementedError


try:
    toy = [Chunk(0, "t", "the cat sat on the mat", 0),
           Chunk(1, "t", "the dog sat on the log", 0),
           Chunk(2, "t", "quantisation reduces memory bandwidth", 0)]
    tb = BM25(toy)
    s = tb.score("cat")
    check("1 the matching document wins", int(np.argmax(s)) == 0)
    check("1 non-matching documents score zero", close(s[1], 0.0) and close(s[2], 0.0))
    check("1 a term in every document carries little weight", tb.idf["the"] < tb.idf["cat"] / 2,
          f"idf(the)={tb.idf['the']:.3f} vs idf(cat)={tb.idf['cat']:.3f}")

    padded = BM25([Chunk(0, "t", "cat " + "filler " * 200, 0), Chunk(1, "t", "cat", 0)])
    check("1 length normalisation penalises padding", padded.score("cat")[1] > padded.score("cat")[0])

    sat = BM25([Chunk(0, "t", "cat", 0), Chunk(1, "t", "cat " * 50, 0)])
    ratio = sat.score("cat")[1] / max(sat.score("cat")[0], 1e-9)
    check("1 term frequency saturates", ratio < 3.0,
          f"50x the frequency is worth {ratio:.2f}x the score")

    nolen = BM25([Chunk(0, "t", "cat " + "filler " * 200, 0), Chunk(1, "t", "cat", 0)], b=0.0)
    check("1 b=0 turns length normalisation off",
          close(nolen.score("cat")[0], nolen.score("cat")[1], rtol=1e-9))
except Exception as _e:
    check("part 1", False, f"{type(_e).__name__}: {_e}")
report()

# %%
bm25 = None
try:
    bm25 = BM25(chunks)
    print("BM25 top 3 for:", QUERIES[0].text)
    for i in np.argsort(-bm25.score(QUERIES[0].text))[:3]:
        print(f"  [{by_id[i].module}] {by_id[i].preview}")
except Exception as _e:
    print(f"BM25 not ready yet ({type(_e).__name__}) - implement part 1 first.")

# %% [markdown]
# ---
# ## Part 2 — Dense retrieval
#
# Build a TF-IDF matrix over the chunks, then reduce it with a truncated SVD. That is LSA, and it
# is the same Eckart–Young truncation from module 01 applied to a term–document matrix.
#
# The architectural point to internalise: document embeddings are computed **without seeing any
# query**. That independence is what lets you index them offline, and it is also the ceiling on
# what a bi-encoder can do — one vector has to summarise everything the passage might be asked
# about. Part 7's reranker exists precisely because of this limitation.

# %%
class LSARetriever:
    def __init__(self, chunks: list[Chunk], dim: int = 128):
        self.chunks = chunks
        # TODO:
        #   1. build the vocabulary and a (n_chunks, vocab) matrix of (1 + log tf) * idf
        #   2. L2-normalise the rows
        #   3. np.linalg.svd(..., full_matrices=False); keep the top `dim` components
        #   4. store self.components = Vt[:dim] and self.embeddings = normalised U[:, :dim] * S[:dim]
        raise NotImplementedError

    def embed_query(self, query: str) -> np.ndarray:
        """Same weighting as a document, then project with self.components, then normalise."""
        # TODO
        raise NotImplementedError

    def score(self, query: str) -> np.ndarray:
        # TODO — both sides are unit-norm, so a dot product is the cosine.
        raise NotImplementedError


dense = None
try:
    dense = LSARetriever(chunks)
    check("2 embeddings have the right shape", dense.embeddings.shape == (len(chunks), 128))
    check("2 embeddings are unit-norm",
          close(np.linalg.norm(dense.embeddings, axis=1), np.ones(len(chunks)), atol=1e-8))
    s = dense.score(QUERIES[0].text)
    check("2 cosine scores are in [-1, 1]", float(np.abs(s).max()) <= 1.0 + 1e-9)
    check("2 a chunk is most similar to itself",
          int(np.argmax(dense.embeddings @ dense.embeddings[10])) == 10)
    print("\nDense top 3 for:", QUERIES[0].text)
    for i in np.argsort(-s)[:3]:
        print(f"  [{by_id[i].module}] {by_id[i].preview}")
except Exception as _e:
    check("part 2", False, f"{type(_e).__name__}: {_e}")
report()

# %% [markdown]
# ---
# ## Part 3 — Metrics first
#
# Do not tune anything until you can measure it. Implement four, and check each against a ranking
# you can verify in your head.
#
# - **hit@k** — is *any* relevant chunk in the top $k$? The hard ceiling on the generator.
# - **recall@k** — what fraction of relevant chunks made it?
# - **MRR** — $1/\text{rank of the first relevant document}$.
# - **nDCG@k** — $\sum_i \frac{2^{rel_i}-1}{\log_2(i+1)}$, normalised by the ideal ordering.
#
# Use the $2^{rel}-1$ gain: it makes one genuinely-answering passage worth more than three merely
# on-topic ones, which is the behaviour you want when comparing rerankers.

# %%
def hit_at_k(ranked, relevant, k): ...
def recall_at_k(ranked, relevant, k): ...
def mrr(ranked, relevant): ...
def dcg(gains): ...
def ndcg_at_k(ranked, grades, k): ...


# TODO: replace the five stubs above with real implementations.
# (They are written as `...` bodies so the notebook still imports; each returns None
#  until you implement it, and the checks below will fail loudly.)

def evaluate(rank_fn, chunks, queries, k: int = 10) -> dict:
    """Mean hit@k, recall@k, MRR and nDCG@k over the evaluation set. Given to you."""
    hits, recalls, mrrs, ndcgs = [], [], [], []
    for q in queries:
        grades = {c.id: q.relevance(c) for c in chunks if q.relevance(c) > 0}
        top = {cid for cid, g in grades.items() if g >= 3}
        if not top:
            continue
        ranked = rank_fn(q.text)
        hits.append(hit_at_k(ranked, top, k))
        recalls.append(recall_at_k(ranked, top, k))
        mrrs.append(mrr(ranked, top))
        ndcgs.append(ndcg_at_k(ranked, grades, k))
    return {"hit@k": float(np.mean(hits)), "recall@k": float(np.mean(recalls)),
            "mrr": float(np.mean(mrrs)), "ndcg@k": float(np.mean(ndcgs)), "n": len(hits)}


try:
    check("3 hit@k is binary", hit_at_k([5, 6, 7], {7, 9}, 3) == 1.0 and hit_at_k([5, 6], {7}, 2) == 0.0)
    check("3 recall@k counts all relevant", close(recall_at_k([1, 2, 3], {1, 3, 9}, 3), 2 / 3))
    check("3 MRR at rank 1", close(mrr([7, 1, 2], {7}), 1.0))
    check("3 MRR at rank 3", close(mrr([1, 2, 7], {7}), 1 / 3))
    check("3 MRR is 0 when nothing is found", close(mrr([1, 2, 3], {7}), 0.0))
    check("3 nDCG is 1 for the ideal ordering", close(ndcg_at_k([1, 2, 3], {1: 3, 2: 1, 3: 1}, 3), 1.0))
    check("3 nDCG punishes a reversal", ndcg_at_k([3, 2, 1], {1: 3, 2: 1, 3: 1}, 3) < 0.7)
    check("3 one grade-3 beats three grade-1s",
          ndcg_at_k([1], {1: 3, 2: 1, 3: 1, 4: 1}, 1) > ndcg_at_k([2, 3, 4], {1: 3, 2: 1, 3: 1, 4: 1}, 3))
except Exception as _e:
    check("part 3", False, f"{type(_e).__name__}: {_e}")
report()

# %% [markdown]
# ---
# ## Part 4 — Hybrid retrieval with RRF
#
# $$\mathrm{RRF}(d) = \sum_r \frac{1}{k + \mathrm{rank}_r(d)},\qquad k \approx 60.$$
#
# Rank-level, not score-level: BM25 scores are unbounded, cosine scores live in $[-1,1]$, and
# nobody wants to maintain a calibration between them.

# %%
def rrf(rankings: list[list[int]], k: int = 60) -> list[int]:
    # TODO
    raise NotImplementedError


# Defined outside the try so later cells can still reference them if this one fails.
rank_bm25 = lambda q, n=100: list(np.argsort(-bm25.score(q))[:n])
rank_dense = lambda q, n=100: list(np.argsort(-dense.score(q))[:n])
rank_hybrid = lambda q, n=100: rrf([rank_bm25(q, 50), rank_dense(q, 50)])[:n]

try:
    check("4 RRF rewards agreement", rrf([[9, 5, 1], [7, 5, 2]])[0] == 5,
          "doc 5 is second in both and beats docs that are first in only one list")
    check("4 RRF is convex, not a rank average", rrf([[1, 2, 3], [3, 2, 1]])[0] == 1,
          "1/(k+r) means extremes beat middles — know this before you trust it")

    print(f"{'retriever':<14} {'hit@10':>8} {'recall@10':>10} {'MRR':>7} {'nDCG@10':>9} {'recall@50':>10}")
    res = {}
    for name, fn in [("BM25", rank_bm25), ("dense (LSA)", rank_dense), ("hybrid RRF", rank_hybrid)]:
        r = res[name] = evaluate(fn, chunks, QUERIES, k=10)
        deep = evaluate(fn, chunks, QUERIES, k=50)
        r["recall@50"], r["hit@50"] = deep["recall@k"], deep["hit@k"]
        print(f"{name:<14} {r['hit@k']:>8.3f} {r['recall@k']:>10.3f} {r['mrr']:>7.3f} "
              f"{r['ndcg@k']:>9.3f} {r['recall@50']:>10.3f}")

    check("4 both retrievers work at all", res["BM25"]["hit@k"] >= 0.8 and res["dense (LSA)"]["hit@k"] >= 0.8)
    # Complementarity, checked where it actually shows: BM25 wins precision@10 (exact
    # terms rank first), dense wins DEEP recall (it finds paraphrases BM25 shares no
    # vocabulary with). Look at the two columns, not at one summary number.
    check("4 BM25 wins on precision@10", res["BM25"]["ndcg@k"] >= res["dense (LSA)"]["ndcg@k"])
    check("4 dense wins on deep recall", res["dense (LSA)"]["recall@50"] > res["BM25"]["recall@50"])
    check("4 every first stage has the answer in its top 50",
          all(res[n]["hit@50"] >= 0.999 for n in res),
          "so what remains is a RANKING problem, not a retrieval problem")
    check("4 fusion draws on both retrievers",
          len(set(rank_hybrid(QUERIES[0].text, 10)) - set(rank_bm25(QUERIES[0].text, 10))) > 0)
except Exception as _e:
    check("part 4", False, f"{type(_e).__name__}: {_e}")
report()

# %% [markdown]
# **4a-note.** You may find that the fused ranking does **not** beat BM25 at $k=10$ on this corpus.
# That is a result, not a bug, and it is the most useful thing in this module: LSA is a weak
# encoder, and fusing a strong retriever with a weak one can cost precision even while it adds
# recall. "Hybrid is better" is a hypothesis to test per corpus, not a fact — which is precisely
# why you build the evaluation set *before* tuning anything.
#
# **4b.** Find a query where BM25 wins and one where dense wins, and explain each in one sentence.
# The expected shapes: BM25 wins on rare exact tokens (`d_k`, `zero_grad`, `bf16`); dense wins on
# paraphrase, where the query and the passage share no vocabulary at all.

# %%
try:
    for q in QUERIES:
        top = {c.id for c in chunks if q.relevance(c) >= 3}
        if not top:
            continue
        rb, rd = mrr(rank_bm25(q.text), top), mrr(rank_dense(q.text), top)
        if abs(rb - rd) > 0.2:
            winner = "BM25 " if rb > rd else "dense"
            print(f"{winner} wins ({rb:.2f} vs {rd:.2f}): {q.text}")
except Exception as _e:
    pass

# %% [markdown]
# ---
# ## Part 5 — Chunking
#
# Compare fixed-width against sentence-aware chunking. Look at the *text* of a few chunks from
# each, not just the metric — a chunk that starts mid-sentence embeds badly and reads terribly when
# shown to a user as a citation.

# %%
try:
    fixed = build_corpus(chunk_fixed)
    bm_fixed = BM25(fixed)
    r_sent = evaluate(lambda q: list(np.argsort(-bm25.score(q))[:50]), chunks, QUERIES)
    r_fix = evaluate(lambda q: list(np.argsort(-bm_fixed.score(q))[:50]), fixed, QUERIES)
    print(f"sentence-aware: hit@10 {r_sent['hit@k']:.3f}  MRR {r_sent['mrr']:.3f}")
    print(f"fixed-width:    hit@10 {r_fix['hit@k']:.3f}  MRR {r_fix['mrr']:.3f}\n")
    print("a fixed-width chunk:\n  ", fixed[7].preview)
    print("a sentence-aware chunk:\n  ", chunks[7].preview)
    lower_fixed = sum(1 for c in fixed if c.text[:1].islower()) / len(fixed)
    lower_sent = sum(1 for c in chunks if c.text[:1].islower()) / len(chunks)
    check("5 fixed-width chunks usually start mid-sentence", lower_fixed > 0.5, f"{lower_fixed:.0%}")
    check("5 sentence-aware chunks rarely do", lower_sent < 0.2, f"{lower_sent:.0%}")
except Exception as _e:
    check("part 5", False, f"{type(_e).__name__}: {_e}")
report()

# %% [markdown]
# **5b.** Sweep the chunk size (200, 400, 700, 1200, 2000 characters) and plot hit@10. You should
# see a hump: too small and a chunk lacks the context to be understood, too large and its
# embedding becomes an average of several topics. Where is the optimum here, and what would move it?

# %%
try:
    print(f"{'size':>6} {'chunks':>8} {'hit@10':>8} {'MRR':>7}")
    for size in (200, 400, 700, 1200, 2000):
        cs = build_corpus(chunk_sentences, size=size, overlap=size // 7)
        b = BM25(cs)
        r = evaluate(lambda q: list(np.argsort(-b.score(q))[:50]), cs, QUERIES)
        print(f"{size:>6} {len(cs):>8} {r['hit@k']:>8.3f} {r['mrr']:>7.3f}")
except Exception as _e:
    pass

# %% [markdown]
# ---
# ## Part 6 — An approximate index
#
# Exact search is $O(Nd)$ per query and perfectly fine up to a few hundred thousand vectors. Past
# that, IVF: cluster with k-means, search only the $n_{\text{probe}}$ nearest cells.
#
# Report **recall against exact search** alongside the speedup. A system that is 50× faster at 70%
# recall may be much worse than one 10× faster at 98%, and quoting QPS alone hides that.

# %%
class IVFIndex:
    def __init__(self, embeddings: np.ndarray, n_lists: int = 16, seed: int = 0):
        # TODO: k-means over the embeddings; store centroids and the member list of each cell.
        raise NotImplementedError

    def search(self, q: np.ndarray, k: int, n_probe: int = 1):
        """Return (top-k ids, number of vectors actually scored)."""
        # TODO
        raise NotImplementedError


try:
    index = IVFIndex(dense.embeddings, n_lists=16)
    print(f"{'n_probe':>8} {'recall@10 vs exact':>20} {'scanned':>9} {'speedup':>9}")
    rec = {}
    for n_probe in (1, 2, 4, 8, 16):
        rs, scanned = [], []
        for q in QUERIES:
            v = dense.embed_query(q.text)
            exact = set(np.argsort(-(dense.embeddings @ v))[:10])
            got, n = index.search(v, 10, n_probe)
            rs.append(len(exact & set(got)) / 10)
            scanned.append(n)
        rec[n_probe] = float(np.mean(rs))
        print(f"{n_probe:>8} {np.mean(rs):>20.3f} {np.mean(scanned):>9.0f} "
              f"{len(chunks) / max(np.mean(scanned), 1):>8.1f}x")
    check("6 recall rises with n_probe", rec[8] > rec[1])
    check("6 probing every list is exact", close(rec[16], 1.0, atol=1e-9))
except Exception as _e:
    check("part 6", False, f"{type(_e).__name__}: {_e}")
report()

# %% [markdown]
# ---
# ## Part 7 — Reranking
#
# The first stage optimised recall at $k{=}50$. Now optimise precision at $k{=}5$.
#
# A real reranker is a cross-encoder: query and document go through the model **together**, so
# every query token attends to every document token. We cannot download one, so write a
# feature-based stand-in — but keep the structural property that matters, which is that it uses
# **joint** features no dot product of independently-computed vectors could express:
#
# 1. IDF-weighted coverage of the query's terms;
# 2. exact **bigram** hits — phrase match is exactly what a bag of words discards;
# 3. proximity — matched terms clustered together beat the same terms scattered;
# 4. a mild prior on the first-stage rank (a production cross-encoder discards this; a
#    hand-written scorer this crude is better off keeping some of it, and saying so is more honest
#    than pretending otherwise).
#
# Target: beat the hybrid first stage on nDCG@10 *and* MRR.

# %%
def rerank(query: str, candidates: list[int], chunks, bm25, top_k: int = 10) -> list[int]:
    # TODO
    raise NotImplementedError


try:
    rank_reranked = lambda q, n=100: rerank(q, rank_hybrid(q, 30), chunks, bm25, top_k=n)
    r_h = evaluate(rank_hybrid, chunks, QUERIES, k=10)
    r_r = evaluate(rank_reranked, chunks, QUERIES, k=10)
    print(f"{'stage':<14} {'recall@10':>10} {'MRR':>7} {'nDCG@10':>9}")
    print(f"{'hybrid':<14} {r_h['recall@k']:>10.3f} {r_h['mrr']:>7.3f} {r_h['ndcg@k']:>9.3f}")
    print(f"{'+ reranker':<14} {r_r['recall@k']:>10.3f} {r_r['mrr']:>7.3f} {r_r['ndcg@k']:>9.3f}")
    check("7 reranking improves nDCG@10", r_r["ndcg@k"] > r_h["ndcg@k"])
    check("7 reranking improves MRR", r_r["mrr"] > r_h["mrr"])
except Exception as _e:
    check("part 7", False, f"{type(_e).__name__}: {_e}")
report()

# %% [markdown]
# **7b. (discussion)** Why can a reranker only be applied to a shortlist? Give the complexity of
# both stages, and say what happens to end-to-end quality if the first stage's recall@50 is 0.6.

# %% [markdown]
# ---
# ## Part 8 — Grounding and citation verification
#
# The retrieval half is measurable; the generation half is where trust is won or lost. The
# mechanism that matters is not "ask the model to cite its sources" — it is **verifying the
# citations programmatically**, so an unsupported claim is a detectable event rather than a
# customer complaint.
#
# Implement a support check: what fraction of an answer sentence's *informative* terms appear in
# the chunk it cites. Note the trap the check below is designed around — a term the corpus has
# never seen must count as **unsupported**, not be silently dropped, or a claim made entirely of
# invented nouns scores as perfectly grounded.

# %%
def supported(sentence: str, chunk: Chunk, bm25, threshold: float = 0.55):
    """Return (is_supported, fraction_of_informative_terms_present)."""
    # TODO
    raise NotImplementedError


def verify_answer(answer: list[tuple[str, int]], chunks, bm25):
    """Check every (sentence, cited chunk id). Return (attribution_rate, failures)."""
    # TODO
    raise NotImplementedError


try:
    kv = [c for c in chunks if "kv cache" in c.text.lower()][0]
    grounded = kv.text.split(". ")[0] + "."
    fabricated = ("The KV cache is stored on a separate optical coprocessor and "
                  "refreshed every microsecond.")
    rate, failures = verify_answer([(grounded, kv.id), (fabricated, kv.id)], chunks, bm25)
    print(f"attribution rate: {rate:.2f}")
    for sent, cid, frac, why in failures:
        print(f"  FLAGGED ({frac:.2f} support): {sent[:70]}...  [{why}]")
    check("8 a grounded sentence passes", not any(f[0] == grounded for f in failures))
    check("8 a fabricated one is caught", any(f[0] == fabricated for f in failures))
    check("8 a nonexistent citation is caught",
          len(verify_answer([("anything at all", 99999)], chunks, bm25)[1]) == 1)
except Exception as _e:
    check("part 8", False, f"{type(_e).__name__}: {_e}")
report()

# %% [markdown]
# **8b. (discussion)** Name a false claim this lexical checker would *not* catch, and say what you
# would use instead. (Hint: reuse the chunk's own vocabulary and reverse the meaning. The
# production answer is a natural-language-inference model, and the reason is precisely this
# failure.)
#
# **8c.** Assemble a full pipeline function `answer(query)` that retrieves, reranks, takes the top
# 3, and returns the assembled context with chunk ids. Order the context strongest-first-and-last
# ("lost in the middle"), and deduplicate near-identical chunks — overlapping chunks from the same
# article will otherwise give a false impression of corroboration.

# %% [markdown]
# ---
# ## Revisiting the spine
#
# Attention is retrieval. $\mathrm{softmax}(QK^\top/\sqrt{d})V$ is a soft differentiable lookup
# over $T$ items in the model's context; RAG is the same operation with a hard top-$k$ over a
# billion items outside the model's parameters. The differences are worth stating precisely,
# because it is a strong thing to say unprompted in an interview:
#
# | | attention | RAG |
# |---|---|---|
# | memory | the context window, in activations | an external index |
# | selection | softmax over all keys | top-$k$ over an ANN index |
# | similarity | learned $W_QW_K^\top$ (module 01) | frozen embeddings + BM25 |
# | differentiable | yes | no (usually) |
# | updates | retraining | reindexing |
#
# **Stretch goals:**
#
# 1. Make retrieval differentiable: replace the top-$k$ with a softmax over the whole corpus and
#    backpropagate into the query embedding. That is what RAG's original end-to-end training did.
# 2. Add query expansion: generate three paraphrases of the query, retrieve for each, fuse with
#    RRF. Measure the lift — this is usually one of the biggest cheap wins available.
# 3. Implement contextual retrieval: prepend a one-sentence description of each chunk's place in
#    its article before indexing. Measure the change in hit@10.
# 4. Implement HNSW's search procedure (greedy descent through a layered proximity graph) and
#    compare recall/latency against your IVF index at equal memory.
