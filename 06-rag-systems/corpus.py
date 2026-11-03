"""The corpus for exercise 06: this course's own articles, plus an evaluation set.

Using the repository as the corpus has two advantages. The text is real prose rather
than synthetic filler, so lexical and semantic retrieval behave the way they do in
practice; and you already know the answers, so you can judge a bad retrieval instantly.

Relevance judgements are defined by a *predicate* (module + required phrase) rather
than by chunk id, so they survive re-chunking and survive edits to the articles.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Chunk:
    id: int
    module: str
    text: str
    start_char: int

    @property
    def preview(self) -> str:
        t = " ".join(self.text.split())
        return t[:110] + ("..." if len(t) > 110 else "")


@dataclass
class Query:
    text: str
    module: str
    phrases: tuple[str, ...]
    grade: int = 3           # graded relevance, for nDCG
    notes: str = ""
    _cache: dict = field(default_factory=dict, repr=False)

    def relevance(self, chunk: Chunk) -> int:
        """Graded relevance of a chunk for this query.

        3 — right module and contains a required phrase (the passage that answers it)
        1 — right module, no required phrase (topically related, not the answer)
        0 — wrong module
        """
        if chunk.module != self.module:
            return 0
        low = chunk.text.lower()
        return self.grade if any(p.lower() in low for p in self.phrases) else 1


def load_documents() -> list[tuple[str, str]]:
    """(module_name, markdown_text) for every article in the course."""
    out = []
    for path in sorted(ROOT.glob("*/README.md")):
        out.append((path.parent.name, path.read_text(encoding="utf-8")))
    if not out:
        raise RuntimeError(f"no module READMEs found under {ROOT}")
    return out


# --------------------------------------------------------------- evaluation set
# Twelve questions with a known home. Each is answerable from exactly one article,
# which makes both the "was it retrieved at all" and the "was it ranked first"
# questions meaningful.
QUERIES: list[Query] = [
    Query("why is there a square root of d_k in the attention score?",
          "01-math-foundations", ("saturat", "1/\\sqrt{d_k}", "sqrt{d_k}"),
          notes="the variance argument"),
    Query("what does a KV cache store and why is it correct?",
          "05-inference-optimization", ("kv cache", "keys and values"),
          notes="causal mask makes it an identity"),
    Query("how much memory does Adam need per parameter?",
          "04-ml-system-design", ("16 bytes", "bytes per parameter", "bytes/param"),
          notes="the 16 bytes/param table"),
    Query("what is the pipeline bubble and how do I shrink it?",
          "04-ml-system-design", ("bubble", "micro-batches"),
          notes="GPipe formula"),
    Query("how do I check that my backward pass is correct?",
          "02-ml-coding-from-scratch", ("finite difference", "central difference", "gradcheck"),
          notes="numerical gradient"),
    Query("the model runs but the loss does not go down, what do I check first?",
          "03-ml-debugging", ("overfit a single batch", "single batch"),
          notes="rung 1"),
    Query("why does layer normalisation help optimisation?",
          "02-ml-coding-from-scratch", ("projects the gradient", "orthogonally to the directions"),
          notes="gradient projection"),
    Query("what is the difference between a bi-encoder and a cross-encoder?",
          "06-rag-systems", ("cross-encoder", "bi-encoder"),
          notes="reranking"),
    Query("does speculative decoding change the output distribution?",
          "05-inference-optimization", ("distribution-preserving", "exactly as $p$", "rejection sampler"),
          notes="Leviathan theorem 1"),
    Query("what does the singular value decomposition tell me about a weight matrix?",
          "01-math-foundations", ("eckart", "singular value", "low-rank"),
          notes="Eckart-Young"),
    Query("what is the difference between a view and a copy in pytorch?",
          "00-pytorch-basics", ("views alias", "stride trick", "free"),
          notes="strided memory model"),
    Query("how should I think about constitutional AI versus RLHF?",
          "08-ai-safety-alignment", ("constitutional ai replaces", "written set of principles"),
          notes="alignment methods"),
]


# --------------------------------------------------------------- chunking
def strip_markdown(text: str) -> str:
    """Remove the markup that would otherwise dominate the term statistics."""
    text = re.sub(r"```.*?```", " ", text, flags=re.S)         # code fences
    text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.S)       # display maths
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)       # links -> link text
    text = re.sub(r"^\s*[-*+]\s+", " ", text, flags=re.M)   # list bullets
    text = re.sub(r"[#*_`>|]+", " ", text)                    # keep hyphens: "cross-encoder"
    text = re.sub(r"\s-{2,}\s", " ", text)                    # ...but not em-dash rules
    return re.sub(r"\s+", " ", text).strip()


def chunk_fixed(text: str, module: str, size: int = 700, overlap: int = 100,
                start_id: int = 0) -> list[Chunk]:
    """Fixed-width character chunks with overlap. The naive baseline."""
    clean = strip_markdown(text)
    out, i, cid = [], 0, start_id
    step = max(1, size - overlap)
    while i < len(clean):
        out.append(Chunk(cid, module, clean[i:i + size], i))
        cid += 1
        i += step
    return out


def chunk_sentences(text: str, module: str, size: int = 700, overlap: int = 100,
                    start_id: int = 0) -> list[Chunk]:
    """Pack whole sentences up to `size`, with a sentence or two of overlap.

    The point of comparison with `chunk_fixed`: a chunk that begins mid-sentence
    embeds badly and reads badly when shown to a user as a citation.
    """
    clean = strip_markdown(text)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean) if s.strip()]
    out, cur, cur_len, cid, pos = [], [], 0, start_id, 0
    for s in sentences:
        if cur_len + len(s) > size and cur:
            out.append(Chunk(cid, module, " ".join(cur), pos))
            cid += 1
            # carry the tail sentences forward as overlap
            keep, klen = [], 0
            for t in reversed(cur):
                if klen + len(t) > overlap:
                    break
                keep.insert(0, t)
                klen += len(t)
            pos += max(1, cur_len - klen)
            cur, cur_len = keep, klen
        cur.append(s)
        cur_len += len(s) + 1
    if cur:
        out.append(Chunk(cid, module, " ".join(cur), pos))
    return out


def build_corpus(strategy=chunk_sentences, **kw) -> list[Chunk]:
    chunks, next_id = [], 0
    for module, text in load_documents():
        part = strategy(text, module, start_id=next_id, **kw)
        chunks.extend(part)
        next_id += len(part)
    return chunks


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens of length >= 2. No stemming, no stopword list —
    BM25's IDF term already discounts common words, which is most of the benefit."""
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) >= 2]


if __name__ == "__main__":
    chunks = build_corpus()
    print(f"{len(chunks)} chunks from {len(load_documents())} articles")
    by_mod: dict[str, int] = {}
    for c in chunks:
        by_mod[c.module] = by_mod.get(c.module, 0) + 1
    for m, n in sorted(by_mod.items()):
        print(f"  {m:<32} {n:>3} chunks")
    print(f"\nmean chunk length: {sum(len(c.text) for c in chunks) / len(chunks):.0f} chars")
    print(f"\n{len(QUERIES)} evaluation queries")
    for q in QUERIES:
        rel = [c for c in chunks if q.relevance(c) >= 3]
        print(f"  {len(rel):>2} highly-relevant chunks  |  {q.text[:60]}")
