# Kokoro AI

Kokoro AI is a recruiter assistant that searches a local library of PDF resumes. It combines semantic and keyword search, ranks candidate matches, and lets recruiters review the original resume.

## Start the app

1. Install the packages from `requirements.txt` in your Python environment.
2. Add `GOOGLE_API_KEY` and `PINECONE_API_KEY` to the project `.env` file. Other settings can also be overridden there.
3. Start Streamlit from the project root:

   ```bash
   streamlit run app.py
   ```

The root `app.py` loads the recruiter interface in `streamlit/app.py`. `main.py` provides separate configuration and startup checks; it is not the Streamlit entry point.

## How resume indexing works

When the Streamlit app starts, it creates the application services and starts a background indexer. The indexer scans `data/resumes/` and `data/jds/` for PDFs. Existing files are checked on each app start; matching file hashes in `data/index/document_registry.json` let ingestion skip documents that were already indexed.

For a new or changed PDF in `data/resumes/`, the folder watcher waits for the file to stop changing, then indexes it. It checks the folder about every two seconds and retries failed files with increasing delays. The watcher handles new and changed files; removing a PDF from the folder does not automatically delete its existing Pinecone records.

Each PDF passes through this pipeline:

1. Validate the file and calculate its SHA-256 hash.
2. Extract and normalize the text, identify resume sections, and read candidate details where possible.
3. Split the document into overlapping text chunks while keeping document and candidate metadata.
4. Generate dense and sparse vectors with Pinecone-hosted embedding models and upsert them into the same Pinecone index. The index maps its `text` field to the stored chunk text (`text` → `text`).
5. Add the chunks to the local BM25 corpus and save the corpus under `data/index/bm25/`.
6. Save the document hash and indexing metadata in `data/index/document_registry.json`.

The index is created when the first vectors are written if it does not already exist. The default index name is `hireflow`. If the index is missing at startup, the application clears stale registry records and rebuilds from the local PDF folders.

## How a candidate search works

1. The chat UI cleans the recruiter’s input and keeps the conversation and any pasted job description in the active session.
2. Input guardrails check the request. Relevant conversation memory and a cached answer may be reused.
3. The application builds a resume search request and asks the hybrid indexer for matches.
4. The vector store embeds the query and searches Pinecone for semantic matches. It separately creates a sparse query and searches the same index for keyword matches. If the configured semantic retriever has no Pinecone sparse search method, the hybrid indexer can use its local BM25 search instead.
5. The hybrid indexer normalizes each retrieval source’s scores, combines them using the configured weights, and returns the top results. The defaults are 60% semantic and 40% keyword relevance.
6. Gemini reranks the retrieved candidates against the recruiter’s requirements and returns fit dimensions, strengths, gaps, and a recommendation. The final fit score weights role relevance 30%, skills match 35%, experience match 20%, domain relevance 10%, and evidence strength 5%.
7. When Gemini reranking is unavailable or its response cannot be validated, the application keeps suitable keyword-supported results and labels their score as search relevance. It does not present that fallback score as a Gemini fit assessment.
8. For successful Gemini rankings, the prompt builder prepares a grounded answer. Gemini generates the response using the retrieved resume information and conversation context.
9. Output guardrails validate the answer. The completed response is saved in the session cache and memory, then the chat displays the answer and candidate cards.

The current chat answer path uses a shallow resume search plan. `core/retrieval/search_router.py` contains query analysis and planning utilities, but `KokoroApplication.answer()` currently constructs its search plan directly.

## Where code lives

| File or folder | Responsibility |
| --- | --- |
| `app.py` | Root Streamlit launcher. |
| `streamlit/app.py` | Chat interface, session controls, resume watcher, candidate cards, and evaluation UI. |
| `core/application.py` | Coordinates indexing, retrieval, ranking, generation, memory, and guardrails. |
| `core/ingestion/parsing.py` | PDF text extraction, cleanup, section detection, and candidate metadata. |
| `core/ingestion/ingestion.py` | File validation, hashing, chunking, indexing, and batch results. |
| `core/ingestion/registry.py` | Local hash registry used to skip previously indexed documents. |
| `core/retrieval/vector_store.py` | Pinecone index creation, hosted dense/sparse embeddings, vector writes, and queries. |
| `core/retrieval/hybrid_indexer.py` | Local BM25 storage, retrieval-score normalization, fusion, and hybrid ranking. |
| `core/retrieval/re_ranker.py` | Gemini fit scoring, response validation, and search-only fallback behavior. |
| `core/retrieval/search_router.py` | Query intent, filters, decomposition, and search-plan utilities. |
| `core/generation/prompt_builder.py` | Builds grounded prompts; it does not call Gemini. |
| `core/guardrails/guardrails.py` | Checks recruiter input, retrieved context, and generated output. |
| `core/memory/memory_rag.py` | Session memory, relevant-turn selection, response cache, and cache expiry. |
| `core/evaluation/evaluator.py` | Retrieval metrics and optional RAGAS generation evaluation. |
| `utils/config.py` | Environment and `.env` settings, defaults, and configuration validation. |
| `utils/schemas.py` | Shared Pydantic models for documents, queries, retrieval results, and responses. |
| `tests/` | Unit tests for parsing, ingestion, retrieval, schemas, memory, guardrails, and reranking. |

## Configuration defaults

The defaults are defined in `utils/config.py` and can be overridden by environment variables or `.env` values.

- Pinecone index: `hireflow`
- Dense embedding model: `llama-text-embed-v2`
- Sparse embedding model: `pinecone-sparse-english-v0`
- Vector dimension: `768`
- Pinecone distance metric: dot product
- Resume chunks: 1,000 characters with 150 characters of overlap
- Semantic and keyword fusion weights: `0.60` and `0.40`
- Gemini model: `gemini-3.6-flash`

The Pinecone index dimension and metric must match these settings. Changing the embedding model, dimension, or index type may require creating a compatible index and reindexing the local PDFs.

## Evaluation and tests

In the app’s **Evaluation** workspace, enter a search question and, optionally, known relevant candidate IDs. The app can report precision@k, recall@k, reciprocal rank, and nDCG. Optional answer evaluation uses RAGAS and Gemini.

Run the test suite from the project root after installing the test dependency:

```bash
python -m pip install pytest
python -m pytest -q
```

Useful static checks for the Python files are:

```bash
black --check $(git ls-files '*.py')
isort --profile black --check-only $(git ls-files '*.py')
pyflakes $(git ls-files '*.py')
```
