# VIGO Team - Huawei TechArena 2025

Our submission for the Huawei TechArena 2025 competition, where we finished in the **top 7**.

The challenge theme was **"spend less, do more"**. In practice that meant building an LLM inference pipeline that answers exam-style questions as accurately as possible while keeping latency as low as possible. We couldn't just throw a big model at it, so we used a small one (**Qwen3 1.7B**) and put most of our effort into everything around it: retrieval, prompting, and a solver for the maths questions.

Everything is written in Python.

## What the pipeline does

The entry point is `load.py`. It exposes `loadPipeline()`, which returns a `MyModel` object. You call that object with a list of questions and it returns a list of answers. Each question looks like this:

```python
{
    "questionID": "c5c79dfc-...",
    "subject": "history",          # algebra, math, geography, history or chinese
    "question": "What were the underlying causes of the Peloponnesian War?"
}
```

and each answer comes back as `{"questionID": ..., "answer": ...}`.

Here's what happens to a batch of questions:

1. **Algebra / math questions go to a calculator tool first.** `AlgebraTool` is a small solver built on SymPy. It pulls the expression out of the question with regex (plain text, or LaTeX in `$...$`, `$$...$$` or `\(...\)`), converts it to something SymPy can parse, and tries four strategies in order: simple arithmetic, linear equations / systems ("Solve for x: ..."), function evaluation ("f(x) = ..., x = 3"), and trig / special values. If any of them produces an answer, we use it directly and the LLM never sees that question. If none of them work, the question falls through to the model.
2. **Everything else is grouped by subject** so each group can use its own generation settings.
3. **History and geography questions get retrieval (RAG).** The question is embedded with our fine-tuned bi-encoder, the top 10 chunks are pulled from a FAISS index, and a cross-encoder reranks them. The reranked chunks are pasted into the prompt as context before the question.
4. **Qwen3-1.7B generates the answer** in batches of up to 64 questions. The model is loaded once for the whole run and unloaded when it's done, to free GPU memory.
5. **The output is cleaned up.** Any `<think>...</think>` blocks are stripped and we keep only the first line of the response, since the task wants short answers.

Per-subject generation settings (all in `generation_params` in `load.py`):

| Subject   | temperature | max new tokens |
|-----------|-------------|----------------|
| algebra   | 0.1         | 130            |
| math      | 0.1         | 120            |
| geography | 0.2         | 80             |
| history   | 0.2         | 80             |
| chinese   | 0.3         | 120            |

### Keeping it fast and small

A few things we did specifically for the "spend less" part:

- **4-bit quantisation** (bitsandbytes NF4 with double quantisation) when a GPU is available, with fp16 compute. It falls back to fp32 on CPU.
- **Batching** questions instead of running them one at a time.
- **Short prompts and short outputs**: inputs are truncated to 412 tokens and the system prompt just asks for "a short, factual answer only".
- **Offline mode**: the model is loaded from a local cache (`/app/models`) with `HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE` set, so there are no network calls at inference time.
- **Solving maths with SymPy** instead of asking the model, which is both faster and more reliable for simple problems.

## Repo structure

| Folder / file | What it's for |
|---|---|
| `load.py` | The final pipeline that ties everything together (described above). |
| `finetuner/` | Fine-tuning code for the model. |
| `train_rag/` | Building the retrieval side: chunking the documents, building the FAISS index, and the extraction script. |
| `biencoder/` | The bi-encoder used for first-stage retrieval (loaded with `SentenceTransformer("biencoder")`). |
| `reranker/` | The cross-encoder that reranks the retrieved chunks (loaded with `CrossEncoder("reranker")`). |
| `difficulty_classifier/` | A classifier for estimating how hard a question is. |
| `history_prompt_engineering/` | Prompt experiments for the history questions. |

Each folder has its own README with more detail.

## Results

Scored by the LLM judge the organisers deployed:

| Metric | Score |
|---|---|
| Overall accuracy | 68.77% |
| Latency score | 98.35% |
| Algebra | 51.15% |
| Geography | 71.92% |
| History | 83.43% |
| Chinese | 43.61% |

History and geography did best, which makes sense because they're the two subjects where retrieval feeds the model real context. Chinese was our weakest, and algebra was held back by the limits of a 1.7B model and by our regex-based parsing only covering fairly simple question formats.

## Running it

`load.py` has an example at the bottom (`if __name__ == "__main__"`) that runs a set of history and geography questions and prints the answers plus total latency:

```bash
python -u load.py
```

To run it you'll need:

- Python 3 with `torch`, `transformers`, `bitsandbytes` (for the GPU/4-bit path), `sentence-transformers`, `faiss`, and `sympy`
- `Qwen/Qwen3-1.7B` already downloaded into `/app/models` (the code loads with `local_files_only=True`)
- The retrieval files in the working directory: the `biencoder` and `reranker` model folders, plus `faiss_index.idx`, `id_map.json` and `chunked_docs.jsonl`

If the retrieval files are missing, the pipeline prints a warning and carries on without RAG, so history and geography just run on the base model.

## Known limitations

Being honest about what we'd fix with more time:

- The maths tool is regex and pattern based, so it only handles fairly standard question phrasings. Anything unusual goes to the LLM, which is much less reliable at algebra.
- Taking only the first line of the output is a simple trick that works for short answers but would cut off anything that needs a longer response.
- Chinese questions don't use retrieval, and this is where we scored lowest.

## Team

Built by the VIGO team for Huawei TechArena 2025.
