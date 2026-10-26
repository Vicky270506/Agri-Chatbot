import re
import streamlit as st
from rank_bm25 import BM25Okapi
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

with open("agri_data.txt", encoding="utf-8") as f:
    facts = [line.strip() for line in f.read().splitlines() if line.strip()]

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.from_texts(facts, embeddings)

STOPWORDS = {
    "a", "an", "the", "is", "are", "of", "in", "on", "for", "to", "and", "or",
    "with", "what", "which", "how", "do", "does", "can", "i", "my", "me",
    "should", "be", "it", "its", "that", "this", "at", "by", "as", "well",
}

def tokenize(text):
    """Lowercase, split into words, drop stopwords, crude plural stripping
    so 'crops' matches 'crop' and 'fungicides' matches 'fungicide'."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    words = [w for w in words if w not in STOPWORDS]
    return [w[:-1] if len(w) > 3 and w.endswith("s") else w for w in words]

bm25 = BM25Okapi([tokenize(fact) for fact in facts])

RRF_K = 60

DEFAULT_CUTOFF = 0.80

def hybrid_search(query, top_n=3):
    """Combine semantic and keyword rankings with Reciprocal Rank Fusion.
    Returns (results, best_distance) where results is a list of
    (fact, semantic_rank, semantic_distance, keyword_rank) tuples and
    best_distance is the smallest semantic distance over ALL facts."""
    dense = db.similarity_search_with_score(query, k=len(facts))
    dense_rank = {doc.page_content: r for r, (doc, _) in enumerate(dense, start=1)}
    dense_dist = {doc.page_content: float(dist) for doc, dist in dense}
    best_distance = min(dense_dist.values())

    scores = bm25.get_scores(tokenize(query))
    order = sorted(range(len(facts)), key=lambda i: scores[i], reverse=True)
    keyword_rank = {
        facts[i]: r
        for r, i in enumerate([i for i in order if scores[i] > 0], start=1)
    }

    fused = []
    for fact in facts:
        score = 1 / (RRF_K + dense_rank[fact])
        if fact in keyword_rank:
            score += 1 / (RRF_K + keyword_rank[fact])
        fused.append((fact, score))

    fused.sort(key=lambda x: x[1], reverse=True)
    results = [
        (fact, dense_rank[fact], dense_dist[fact], keyword_rank.get(fact))
        for fact, _ in fused[:top_n]
    ]
    return results, best_distance

def show_results(results):
    for fact, sem_rank, sem_dist, kw_rank in results:
        st.success(fact)
        kw_text = f"#{kw_rank}" if kw_rank else "no keyword match"
        st.caption(
            f"Semantic rank: #{sem_rank} (distance {sem_dist:.3f})  |  "
            f"Keyword rank: {kw_text}"
        )

st.title("🌾 Agri Chatbot (Offline Hybrid RAG)")

cutoff = st.sidebar.slider(
    "Out-of-scope cutoff (semantic distance)",
    min_value=0.30,
    max_value=1.60,
    value=DEFAULT_CUTOFF,
    step=0.01,
    help="If even the closest fact is farther than this, the app says it "
         "doesn't know instead of returning unrelated facts.",
)

query = st.text_input("Ask your question:")

if query:
    results, best_distance = hybrid_search(query, top_n=3)

    if best_distance > cutoff:
        st.warning(
            "I couldn't find that in my knowledge base. I only know about "
            "crops, soils, rainfall, fertilizers, diseases and irrigation."
        )
        st.caption(
            f"Closest fact distance: {best_distance:.3f} (cutoff {cutoff:.2f})"
        )
        with st.expander("Show the closest facts anyway"):
            show_results(results)
    else:
        show_results(results)
