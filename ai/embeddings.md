---
area: ai
confidence: 1
status: seed
related: [transformers, vector-databases, rag, feature-engineering, model-evaluation-metrics]
updated: 2026-10-14
last_verified: 2026-10-14
---

# Embeddings

> **TL;DR:** An embedding is a learned dense vector representation of an object (word, document, image, user, item) in which geometric proximity encodes semantic similarity.

## Technical definition

An embedding is a mapping f: X → ℝᵈ, learned from data, from a discrete or high-dimensional input space X (tokens, documents, images, user/item IDs) to a dense, low-dimensional vector space (typically d ≈ 128–4096). The training objective is chosen so that distances or inner products in ℝᵈ reflect a task-relevant notion of similarity.

Common training paradigms:
- **Lookup tables trained end to end**: an embedding matrix E ∈ ℝ^{|V|×d}, indexed by ID. Used for tokens, categorical features, and user/item IDs in recommenders.
- **Contrastive learning**: pull positive pairs together and push negatives apart, e.g. InfoNCE: L = −log( exp(sim(q,k⁺)/τ) / Σⱼ exp(sim(q,kⱼ)/τ) ). Used by CLIP, SimCSE, and modern text-embedding models, with in-batch or hard negatives.
- **Encoder models** (Transformers): contextual embeddings from a pooled hidden state (mean pooling or [CLS]), usually fine-tuned for retrieval with a contrastive objective.
- **Two-tower models**: separate query and item encoders trained jointly. Standard for large-scale retrieval in search, ads, and recommendations.

Similarity measures:
- Cosine: cos(u,v) = u·v / (‖u‖‖v‖). With L2-normalised vectors this equals the dot product.
- Dot product: used when vector magnitude carries signal (some recommenders).
- Euclidean (L2): equivalent ranking to cosine on normalised vectors.

Retrieval at scale uses **approximate nearest neighbour (ANN)** search, trading a little recall for large speedups over exact O(N·d) scan. Common indexes: HNSW (graph-based, high recall, memory heavy), IVF (cluster-based), and IVF-PQ (adds product quantisation to compress vectors). Typical libraries: FAISS, ScaNN, hnswlib.

Properties and limits:
- Embedding models are not interchangeable. Vectors from different models or versions live in different spaces and cannot be compared.
- Quality is task-dependent. Benchmark on your own retrieval data (recall@k, nDCG, MRR), not only on public leaderboards such as MTEB.
- Some models support Matryoshka-style truncation, which allows smaller dimensions with modest quality loss. Check the model card.

## Intuition (beginner)

Think of a map where every item gets GPS coordinates. Things that mean similar things get coordinates close together: "cat" lands near "kitten", far from "invoice". Computers can't compare meaning directly, but they can compare distance between coordinates. An embedding model is the thing that assigns the coordinates, with many more than two dimensions.

## Mini example

Toy 2D coordinates (real ones have hundreds of dimensions):

| Item | x | y |
|---|---|---|
| "refund my order" | 0.9 | 0.1 |
| "I want my money back" | 0.8 | 0.2 |
| "best pasta recipe" | 0.1 | 0.9 |

The first two are close, so a search for "refund my order" finds "I want my money back" even though no words overlap.

Real code (Python, `sentence-transformers`):

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim
docs = ["refund my order", "I want my money back", "best pasta recipe"]
vecs = model.encode(docs, normalize_embeddings=True)

query = model.encode(["how do I get a refund?"], normalize_embeddings=True)
scores = vecs @ query.T          # cosine similarity (vectors are normalised)
print(docs[int(np.argmax(scores))])
```

## In production

- **Typical pipeline**: offline batch job embeds the corpus, vectors go into a vector index (FAISS, pgvector, Milvus, Elasticsearch/OpenSearch kNN, or a managed vector DB). At query time, embed the query online and run ANN search, then often **rerank** the top-k with a cross-encoder or business-rule scoring.
- **Latency and cost**: query embedding adds one model forward pass (~ms to tens of ms). Index memory scales as N × d × bytes per float (e.g. 100M × 768 × 4 B ≈ 307 GB before compression), which is why quantisation (PQ, int8) and sharding are common.
- **Versioning**: changing the embedding model requires re-embedding the whole corpus and rebuilding the index. Run old and new indexes side by side, shadow-test, then cut over.
- **Freshness**: new or updated documents need incremental index updates or periodic rebuilds. Streaming ingestion is more complex than batch.
- **Failure modes**: domain mismatch (a general model on jargon-heavy text), multilingual gaps, long documents truncated at the model's max length (so chunking strategy matters), and embedding drift when the upstream model changes silently.
- **Monitoring**: offline recall@k and nDCG on a labelled set, online click-through or resolution rate, and latency/p99 of the embed + ANN path.

## Business context

- **Semantic search / RAG**: find relevant knowledge-base articles or documents to ground LLM answers. Metric moved: ticket deflection rate, answer accuracy.
- **Recommendations**: two-tower retrieval of candidate items for a user, before a heavier ranking model. Metric moved: CTR, conversion, GMV.
- **Dedup and clustering**: detect near-duplicate listings or group similar support tickets.
- **Scenario**: an e-commerce support bot embeds 50k help articles, retrieves the top 5 for each customer question, and passes them to an LLM. Keyword search missed paraphrases ("money back" vs "refund"); embeddings fixed most of those misses.

## Pitfalls and interview angles

- Mixing embeddings from different models or versions in one index.
- Forgetting to normalise when the index assumes cosine or inner product.
- Chunking too large (diluted meaning) or too small (lost context).
- Assuming high cosine similarity means "relevant". Similar is not the same as answers the question. Hence reranking.
- Interview prompts: "Design semantic search for 100M documents." "How would you evaluate an embedding model?" "HNSW vs IVF-PQ trade-offs?" "Why two-tower for retrieval and cross-encoder for ranking?"

## Self-test
1. Why does cosine similarity equal the dot product after L2 normalisation?
2. What is the main trade-off an ANN index makes, and how would you measure it?
3. You switch to a better embedding model. What must you do to the existing index, and why?

## Related
- [[transformers]] · [[vector-databases]] · [[rag]] · [[feature-engineering]] · [[model-evaluation-metrics]]

## References
- Reimers & Gurevych, *Sentence-BERT* (2019)
- Radford et al., *CLIP* (2021)
- Malkov & Yashunin, *HNSW* (2016)
- Johnson et al., *Billion-scale similarity search with GPUs* (FAISS, 2017)
- MTEB benchmark (verify the current leaderboard and model versions yourself)
