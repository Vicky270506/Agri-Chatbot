# 🌾 Agri Hybrid Retrieval Engine

A hybrid information-retrieval system for agriculture questions. It combines **BM25 keyword search** with **sentence-embedding semantic search** (MiniLM + FAISS), merges the two rankings with **Reciprocal Rank Fusion (RRF)**, and **declines questions its knowledge base cannot answer**. It runs locally in a Streamlit app, with no API key and no internet connection after the first model download.

> **Note:** this is a retrieval system, not full RAG. It returns the most relevant facts from the knowledge base; no language model writes an answer.

---

## Table of contents

- [Why this project](#why-this-project)
- [Features](#features)
- [How it works](#how-it-works)
- [Demo queries and observed results](#demo-queries-and-observed-results)
- [Project structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Tech stack](#tech-stack)
- [Evaluation status](#evaluation-status)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [Acknowledgements](#acknowledgements)
- [Team](#team)
- [License](#license)

---

## Why this project

Questions are short and informal ("my groundnut leaves have spots"), while the knowledge that answers them is written formally ("Groundnut leaf spot disease reduces crop quality and needs proper spraying"). Each retrieval method handles this differently:

| Method | Good at | Misses |
|---|---|---|
| **BM25 (keyword)** | Exact crop, disease and fertilizer names | Paraphrases with different words |
| **Embeddings (semantic)** | Related meaning without shared words | Fine distinctions (high vs. moderate vs. low rainfall) |
| **Hybrid (this project)** | Both, merged by rank | Still needs evaluation on a larger query set |

A search system should also know when it *doesn't* know. Nearest-neighbour search always returns something, even for "Who won the cricket world cup?", so this project adds a distance cutoff that refuses such questions.

## Features

- **One fact per chunk.** Each line of `agri_data.txt` is one retrievable fact, so results are single sentences, not paragraph blobs.
- **Dual index.** The same facts are indexed as 384-dimensional embeddings (FAISS) and as BM25 tokens.
- **Rank fusion.** Dense and keyword rankings are merged with RRF (k = 60). No score normalization is needed.
- **Explainable results.** Every result shows its semantic rank, semantic distance and keyword rank.
- **Out-of-scope detection.** If the best semantic distance is above a cutoff (default `0.80`), the app shows a warning instead of unrelated facts. A sidebar slider adjusts the cutoff live.
- **Runs offline.** After the embedding model has been downloaded once, nothing needs the internet.

## How it works

```
                      ┌────────────────────┐
 agri_data.txt ─────► │ one fact per line  │
 (41 facts)           └─────────┬──────────┘
                      ┌─────────┴──────────┐
                      ▼                    ▼
              BM25 index            MiniLM embeddings
              (keyword)             + FAISS (semantic)
                      │                    │
 user question ───────┴──────┬─────────────┘
                            ▼
                  keyword rank + semantic rank
                            ▼
              Reciprocal Rank Fusion  (k = 60)
                            ▼
        best semantic distance > cutoff (0.80)?
             │ no                         │ yes
             ▼                            ▼
      top-3 facts + ranks        "not found in knowledge base"
```

**Reciprocal Rank Fusion.** Each ranked list gives a fact `1 / (60 + rank)` points, and the points are added:

```
RRF(d) = 1 / (60 + rank_semantic(d)) + 1 / (60 + rank_bm25(d))
```

A fact with no keyword match simply gets no points from the BM25 list.

**Distance.** FAISS returns squared Euclidean distance, so lower means closer. With unit-length embeddings this is about `2 − 2 × cosine similarity`.

**Tokenizer (keyword side).** Lowercase → regex `[a-z0-9]+` → remove a 32-word stopword list → trim a trailing "s" so "crops" matches "crop".

## Demo queries and observed results

Results from an informal test of eight hand-written queries (best semantic distance of the closest fact; lower is closer):

| Query | In knowledge base? | Best distance | Top result |
|---|---|---|---|
| What does DAP fertilizer do? | ✅ | 0.508 | DAP fertilizer provides phosphorus for root development. |
| Which soil is best for rice? | ✅ | 0.566 | Clay soil retains water and is suitable for rice cultivation. |
| My groundnut leaves have spots, what should I do? | ✅ | 0.605 | Groundnut leaf spot disease reduces crop quality and needs proper spraying. |
| What helps plants resist disease? | ✅ | 0.683 | Potassium fertilizers improve disease resistance in plants. |
| What is the best crop to grow in Chennai during monsoon? | ❌ | 0.910 | *(declined at cutoff 0.80)* |
| How much urea should I apply per acre? | ❌ | 1.246 | *(declined)* |
| How do I treat powdery mildew on grapes? | ❌ | 1.302 | *(declined)* |
| Who won the cricket world cup? | ❌ | 1.742 | *(declined)* |

**Where hybrid search helped.** For *"Which crop does well in dry areas with little water?"*, semantic-only search did not show the Ragi fact (*"Ragi grows in dry regions with low rainfall and sandy soil."*) in its top three. With BM25 added, the word "dry" lifted it to rank 2.

> These are qualitative observations from a very small query set, and the cutoff was chosen using the same queries. See [Evaluation status](#evaluation-status).

<!-- Add a screenshot of the running app here, for example:
![Agri Hybrid Retrieval screenshot](docs/screenshot.png)
-->

## Project structure

```
├── app.py             # Streamlit app: indexing, hybrid search, cutoff, UI
├── agri_data.txt      # Knowledge base: one agricultural fact per line
├── requirements.txt   # Python dependencies
└── README.md
```

## Installation

**Prerequisites:** Git and Python 3.10–3.12 (newer or older versions may lack pre-built packages for some dependencies).

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows (PowerShell)
# source venv/bin/activate     # macOS / Linux

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` must include `rank-bm25` (the BM25 library) in addition to Streamlit, LangChain, FAISS and sentence-transformers.

**Windows note.** If installation fails while building `greenlet` ("Microsoft Visual C++ 14.0 or greater is required"), use Python 3.11, or install a pre-built wheel with `pip install --only-binary=:all: greenlet` before installing the requirements. The installation downloads PyTorch, so use a good connection.

## Usage

```bash
streamlit run app.py
```

Open <http://localhost:8501> and type a question about crops, soils, rainfall, diseases, fertilizers or irrigation.

The **first launch** downloads the `all-MiniLM-L6-v2` model (about 90 MB), so it takes longer. Later launches use the local copy.

Try:

- *Which soil is best for rice?*
- *What can I spray instead of chemical pesticides?*
- *Which crop does well in dry areas with little water?*
- *How do I treat powdery mildew on grapes?* (expected: not found)

## Configuration

| Setting | Where | Default | Meaning |
|---|---|---|---|
| Out-of-scope cutoff | Sidebar slider (0.30–1.60) | `0.80` | If even the closest fact is farther than this, the app declines. Lower = stricter. |
| `DEFAULT_CUTOFF` | `app.py` | `0.80` | Starting value of the slider. |
| `RRF_K` | `app.py` | `60` | Reciprocal Rank Fusion constant. |
| `top_n` | `hybrid_search(...)` in `app.py` | `3` | Number of facts shown. |

**Adding knowledge.** Add new facts to `agri_data.txt`, one self-contained sentence per line, and restart the app. Avoid duplicate lines.

## Tech stack

- **Python**
- **Streamlit**: web interface
- **sentence-transformers**: `all-MiniLM-L6-v2` embeddings (384 dimensions)
- **FAISS** (`faiss-cpu`) via **LangChain** (`langchain-community`): vector index
- **rank_bm25**: `BM25Okapi` keyword scoring

## Evaluation status

The project has only been tested informally:

- Eight manually written queries (four in scope, four out of scope).
- In-scope best distances were 0.508–0.683; out-of-scope best distances were 0.910–1.742. The 0.80 cutoff sits between them.
- **The cutoff was chosen on the same queries, so the clean separation is optimistic.**
- No human relevance judgments exist, and **MRR, nDCG and precision have not been computed.**

A proper evaluation would use 40+ judged queries, a held-out test set, and a comparison of BM25-only, semantic-only and hybrid ranking. See the project report for the full plan.

## Limitations

- Tiny knowledge base (41 facts), hand-written and **not verified by agricultural experts**. Do not use it for field decisions.
- The cutoff applies only to the *best* result, so weaker second and third results can still appear for accepted questions.
- The tokenizer is crude (plural trimming, not stemming), and the stopword "well" also affects phrases like "well-drained".
- English only.
- The model and indexes are rebuilt on every Streamlit rerun (no caching), and latency has not been measured.
- No language model is used, so there is no generated answer.

## Roadmap

- [ ] Cache the model and indexes (`st.cache_resource`)
- [ ] Build a judged benchmark and report P@3, MRR and nDCG for BM25-only, semantic-only and hybrid
- [ ] Per-result distance threshold
- [ ] Proper stemmer and a domain stopword list
- [ ] Optional LLM step that answers only from the retrieved facts and cites them (full RAG)
- [ ] Larger, expert-verified, multilingual (e.g. Tamil) knowledge base

## Acknowledgements

This project extends the open-source [Agri-Chatbot](https://github.com/kaviyadharshini2805/Agri-Chatbot) by **Kaviyadharshini M**, which provided the original Streamlit + LangChain + FAISS prototype and the knowledge base file `agri_data.txt`.

**Changes made in this project:**

1. One-fact-per-line chunking instead of character splitting
2. Added a BM25 keyword index
3. Added Reciprocal Rank Fusion and rank/distance display
4. Added distance-based out-of-scope detection with an adjustable cutoff

Also built on [Sentence-Transformers](https://www.sbert.net/), [FAISS](https://github.com/facebookresearch/faiss), [rank_bm25](https://github.com/dorianbrown/rank_bm25), [LangChain](https://www.langchain.com/) and [Streamlit](https://streamlit.io/).

## Team

B.Tech. Artificial Intelligence and Data Science, Department of Information Technology, Madras Institute of Technology, Anna University.

- Vignesh B (2023510043)
- Jagadeesh M (2023510052)
- Shyam Sundar V (2023510059)

## License

No license has been chosen yet. The baseline repository this work builds on does not include a license file, so check with its author before choosing one and publishing under it. Once decided, add a `LICENSE` file and update this section.
