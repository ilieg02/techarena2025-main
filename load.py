# -*- coding: utf-8 -*-
"""
Cleaned pipeline (load2.py)

What I changed:
- Replaced all non-UTF-8 / Windows-1252 smart characters in the questions payload with safe ASCII equivalents
  (e.g. replaced smart dashes/quotes with '-' and removed stray invalid bytes).
- Fixed indentation for the example-usage block so the questions list runs under the __main__ guard.
- Kept the utf-8 file encoding declaration at the top so Python reads the file correctly.
- No functional changes to logic beyond sanitizing literal strings and indentation.

Save this file as UTF-8 and run: python -u load2.py
"""

import os
import time
import re
import json
import hashlib
from typing import Optional, List, Dict, Any

import torch
import sympy
from sympy import symbols, Eq, solve, sympify

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss


# ==============================
# UTILITY FUNCTIONS
# ==============================


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def remove_thinking(text: str) -> str:
    # remove any <think>...</think> fragments and stray tags
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.replace("<think>", "").replace("</think>", "").strip()


# ==============================
# ALGEBRA TOOL
# ==============================


class AlgebraTool:
    def _format(self, value: Any) -> str:
        try:
            v = float(value)
            if v.is_integer():
                return str(int(v))
            return f"{v:.4f}"
        except Exception:
            return str(value)

    def _extract_expr(self, q: str) -> Optional[str]:
        # try common latex/equation delimiters
        m = re.search(r"\$\$(.*?)\$\$|\$(.*?)\$|\\\((.*?)\\\)", q, flags=re.DOTALL)
        if m:
            return next((g for g in m.groups() if g), None)
        return None

    def latex_to_math(self, expr: str) -> str:
        # Very small latex -> python replacements; not exhaustive
        expr = expr.replace("\\", "")
        expr = re.sub(r"frac\{(.*?)\}\{(.*?)\}", r"(\1)/(\2)", expr)
        expr = re.sub(r"sqrt\{(.*?)\}", r"sqrt(\1)", expr)
        replacements = [
            ("times", "*"),
            ("div", "/"),
            ("cdot", "*"),
            ("pi", "pi"),
            ("sin", "sin"),
            ("cos", "cos"),
            ("tan", "tan"),
            ("alpha", "alpha"),
            ("beta", "beta"),
            ("theta", "theta"),
            ("infty", "oo"),
        ]
        for old, new in replacements:
            expr = expr.replace(old, new)
        expr = expr.replace("{", "(").replace("}", ")")
        return expr

    def simple_arithmetic(self, q: str) -> Optional[str]:
        expr = self._extract_expr(q)
        if not expr:
            match = re.search(r"(?:Calculate|What is|Evaluate|Compute|Find|Simplify)\s+(.*?)(?:[\.\?]|$)", q, re.IGNORECASE)
            if not match:
                return None
            expr = match.group(1).strip()

        # Use unicode escape sequences so the source file contains only safe characters
        replacements = [
            ("\u2212", "-"),  # minus sign
            ("\u00D7", "*"),  # multiplication sign
            ("\u00F7", "/"),  # division sign
            ("^", "**"),
            ("\u221A", "sqrt"),  # square root symbol
            ("\u03C0", "pi"),  # pi
            ("\u00B2", "**2"),  # superscript 2
            ("\u00B3", "**3"),  # superscript 3
            ("\u2211", "Sum"),  # summation
            ("\u220F", "Product"),  # product
        ]
        for old, new in replacements:
            expr = expr.replace(old, new)

        try:
            expr = self.latex_to_math(expr)
            parsed = sympify(expr, evaluate=True, locals={"sqrt": sympy.sqrt, "pi": sympy.pi, "E": sympy.E})
            result = parsed.evalf(n=10)
            return self._format(result)
        except Exception:
            return None

    def linear_equation(self, q: str) -> Optional[str]:
        x = symbols("x")
        try:
            # Try to capture "Solve ..." style or "Solve for x: ..."
            m = re.search(r"Solve(?: for [^\:]+)?\:?\s*(.+)", q, re.IGNORECASE)
            if m:
                expr = m.group(1).replace("^", "**")
                equations = re.split(r",| and ", expr)
                eqs = []
                for eq in equations:
                    if "=" in eq:
                        left, right = eq.split("=", 1)
                        eqs.append(Eq(sympify(left.strip()), sympify(right.strip())))
                if len(eqs) > 1:
                    sol = solve(eqs, list({s for s in eqs[0].free_symbols}))
                    # If solve returns dict-like or list, format sensibly
                    if isinstance(sol, dict):
                        if x in sol:
                            return self._format(sol[x])
                        return str(sol)
                    return self._format(sol)
                elif len(eqs) == 1:
                    sol = solve(eqs[0], x)
                    if sol:
                        return self._format(sol[0])
            else:
                # Try generic "Solve for x: 2*x+3=7" style
                m2 = re.search(r"for\s+([a-zA-Z]+)\s*[:\-]?\s*(.+)", q, re.IGNORECASE)
                if m2:
                    varname, expr = m2.groups()
                    var = symbols(varname)
                    if "=" in expr:
                        left, right = expr.split("=", 1)
                        eq = Eq(sympify(left.strip()), sympify(right.strip()))
                        sol = solve(eq, var)
                        if sol:
                            return self._format(sol[0])
        except Exception:
            return None
        return None

    def evaluate_function(self, q: str) -> Optional[str]:
        m = re.search(r"f\s*\(\s*([a-zA-Z]+)\s*\)\s*=\s*([^\n;]+).*?([a-zA-Z]+)\s*=\s*([^\n;]+)", q, re.IGNORECASE | re.DOTALL)
        if m:
            try:
                varname = m.group(1)
                fx_expr = m.group(2)
                val_name = m.group(3)
                x_val = m.group(4)
                x = symbols(varname)
                fx = sympify(fx_expr)
                return self._format(fx.subs(x, sympify(x_val)).evalf())
            except Exception:
                return None
        # fallback simpler form: "f(x) = x**2, x = 3"
        m2 = re.search(r"f\s*\(\s*x\s*\)\s*=\s*([^\n;,]+).*x\s*=\s*([^\n;]+)", q, re.IGNORECASE | re.DOTALL)
        if m2:
            try:
                fx_expr, x_val = m2.groups()
                x = symbols("x")
                fx = sympify(fx_expr)
                return self._format(fx.subs(x, sympify(x_val)).evalf())
            except Exception:
                return None
        return None

    def trig_or_special(self, q: str) -> Optional[str]:
        try:
            m = re.search(r"(?:Evaluate|Compute|What is|Find|Simplify|Calculate)\s+(.*?)(?:[\.?\n]|$)", q, re.IGNORECASE | re.DOTALL)
            if not m:
                return None
            expr = m.group(1).strip()
            # use escape sequences for special glyphs
            expr = expr.replace("^", "**").replace("\u03C0", "pi").replace("\u221A", "sqrt")
            # avoid blind replace of single-letter 'e' -> E if it's part of a variable name
            expr = re.sub(r"(?<![A-Za-z0-9_])e(?![A-Za-z0-9_])", "E", expr)
            val = sympify(expr, evaluate=True).evalf()
            return self._format(val)
        except Exception:
            return None

    def solve(self, q: str) -> Optional[str]:
        for fn in [self.simple_arithmetic, self.linear_equation, self.evaluate_function, self.trig_or_special]:
            try:
                out = fn(q)
                if out is not None:
                    return out
            except Exception:
                continue
        return None


# ==============================
# MAIN PIPELINE
# ==============================


class MyModel:
    def __init__(self):
        self.tool = AlgebraTool()
        self.cache_dir = "/app/models"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.use_4bit = torch.cuda.is_available()
        self.bnb_config = (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            if self.use_4bit
            else None
        )

        self.model = None
        self.tokenizer = None
        self.max_batch_size_17b = 64
        self.generation_params = {
            "algebra": {"temperature": 0.1, "top_p": 0.9, "max_new_tokens": 130},
            "math": {"temperature": 0.1, "top_p": 0.9, "max_new_tokens": 120},
            "geography": {"temperature": 0.2, "top_p": 0.9, "max_new_tokens": 80},
            "history": {"temperature": 0.2, "top_p": 0.9, "max_new_tokens": 80},
            "chinese": {"temperature": 0.3, "top_p": 0.9, "max_new_tokens": 120},
        }

        # ==============================
        # LOAD BI-ENCODER + CROSS-ENCODER + FAISS
        # ==============================
        try:
            self.biencoder = SentenceTransformer("biencoder")
            self.reranker = CrossEncoder("reranker")
            self.faiss_index = faiss.read_index("faiss_index.idx")
            with open("id_map.json", "r") as f:
                self.corpus_ids = json.load(f)
            with open("chunked_docs.jsonl", "r") as f:
                self.corpus = [json.loads(line) for line in f]
            print("[RAG] Bi-encoder, cross-encoder, FAISS index, and corpus loaded successfully.")
        except Exception as e:
            print("[RAG] Warning: failed to load RAG models or FAISS index:", e)
            self.biencoder = None
            self.reranker = None
            self.faiss_index = None
            self.corpus_ids = []
            self.corpus = []

    # -----------------------------
    # UTILITY FUNCTIONS
    # -----------------------------
    def clear_torch_cache(self):
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # best-effort
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass

    def chunk(self, items: List[Dict[str, Any]], size: int):
        for i in range(0, len(items), size):
            yield items[i : i + size]

    # -----------------------------
    # RAG RETRIEVAL
    # -----------------------------
    @torch.inference_mode()
    def _retrieve(self, question: str, topk: int = 10):
        if self.faiss_index is None or self.biencoder is None or self.reranker is None:
            return []

        q_emb = self.biencoder.encode([question], convert_to_numpy=True)
        distances, indices = self.faiss_index.search(q_emb, topk)
        # indices shape: (1, topk) ; some indices can be -1 if not enough vectors
        retrieved_texts = []
        retrieved_ids = []
        for idx in indices[0]:
            if idx < 0 or idx >= len(self.corpus):
                continue
            retrieved_texts.append(self.corpus[idx]["text"])
            # safely handle id map length
            retrieved_ids.append(self.corpus_ids[idx] if idx < len(self.corpus_ids) else None)

        if not retrieved_texts:
            return []

        pairs = [[question, text] for text in retrieved_texts]
        scores = self.reranker.predict(pairs)
        reranked = sorted(zip(retrieved_ids, retrieved_texts, scores), key=lambda x: x[2], reverse=True)
        return [text for _, text, _ in reranked]

    # -----------------------------
    # LOAD QWEN3-1.7B (or configured model)
    # -----------------------------
    def load_17b(self):
        if self.model is not None:
            return
        self.clear_torch_cache()
        name = "Qwen/Qwen3-1.7B"
        self.tokenizer = AutoTokenizer.from_pretrained(
            name, cache_dir=self.cache_dir, local_files_only=True, trust_remote_code=True
        )
        if getattr(self.tokenizer, "pad_token", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if torch.cuda.is_available():
            # when using 4-bit, pass quantization_config
            if self.bnb_config is not None:
                self.model = AutoModelForCausalLM.from_pretrained(
                    name,
                    cache_dir=self.cache_dir,
                    local_files_only=True,
                    trust_remote_code=True,
                    quantization_config=self.bnb_config,
                    torch_dtype=torch.float16,
                    device_map="auto",
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    name,
                    cache_dir=self.cache_dir,
                    local_files_only=True,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    device_map="auto",
                )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                name, cache_dir=self.cache_dir, local_files_only=True, trust_remote_code=True, torch_dtype=torch.float32
            ).to("cpu")
        self.model.eval()

    def unload(self):
        if self.model is not None:
            try:
                del self.model
            except Exception:
                pass
        if self.tokenizer is not None:
            try:
                del self.tokenizer
            except Exception:
                pass
        self.model = None
        self.tokenizer = None
        self.clear_torch_cache()
        time.sleep(0.1)

    # -----------------------------
    # BATCH GENERATE
    # -----------------------------
    @torch.inference_mode()
    def batch_generate(self, batch: List[Dict[str, Any]]):
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model/tokenizer not loaded. Call load_17b() before generation.")

        prompts = []
        for q in batch:
            user_q = q["question"]
            subj = (q.get("subject") or "").lower()
            if subj in ("history", "geography"):
                ctx_docs = self._retrieve(user_q)
                if ctx_docs:
                    context_block = "\n\n".join([f"[DOC]\n{c}" for c in ctx_docs])
                    user_q = f"Context:\n{context_block}\n\nQuestion: {q['question']}\nAnswer concisely:"

            messages = [
                {"role": "system", "content": "You are a precise QA model. Provide a short, factual answer only."},
                {"role": "user", "content": user_q},
            ]
            # Qwen-style tokenizers often have apply_chat_template
            if hasattr(self.tokenizer, "apply_chat_template"):
                text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                # fallback to simple concatenation
                text = "\n".join([m["content"] for m in messages]) + "\n"
            prompts.append(text)

        enc = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=412)
        enc = {k: v.to(self.device) for k, v in enc.items()}

        subj = (batch[0].get("subject") or "").lower()
        params = self.generation_params.get(subj, {"temperature": 0.2, "top_p": 0.9, "max_new_tokens": 100})

        out = self.model.generate(
            **enc,
            max_new_tokens=params["max_new_tokens"],
            temperature=params["temperature"],
            top_p=params["top_p"],
            pad_token_id=self.tokenizer.eos_token_id,
        )

        answers = []
        input_len = enc["input_ids"].shape[1]
        for i, q in enumerate(batch):
            # out is shape (batch, seq_len_out). We take tokens after the input portion.
            seq = out[i].tolist()
            gen_ids = seq[input_len:] if len(seq) > input_len else []
            decoded = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            cleaned = remove_thinking(decoded).split("\n")[0].strip()
            answers.append({"questionID": q["questionID"], "answer": cleaned})

        return answers

    # -----------------------------
    # MAIN CALL
    # -----------------------------
    def __call__(self, questions: List[Dict[str, Any]]):
        algebra_solved: Dict[str, str] = {}
        buckets: Dict[str, List[Dict[str, Any]]] = {}
        for q in questions:
            subj = (q.get("subject") or "").lower()
            if subj in ("algebra", "math"):
                solved = self.tool.solve(q.get("question", ""))
                if solved is not None:
                    algebra_solved[q["questionID"]] = solved
                    continue
            buckets.setdefault(subj, []).append(q)

        final: Dict[str, str] = {}
        if buckets:
            self.load_17b()
            for subj, q_list in buckets.items():
                for chunk in self.chunk(q_list, self.max_batch_size_17b):
                    out_chunk = self.batch_generate(chunk)
                    for r in out_chunk:
                        final[r["questionID"]] = r["answer"]
            self.unload()

        final_output = []
        for q in questions:
            qid = q["questionID"]
            if qid in algebra_solved:
                final_output.append({"questionID": qid, "answer": algebra_solved[qid]})
            else:
                # safe fallback if answer missing
                answer = final.get(qid, "")
                final_output.append({"questionID": qid, "answer": answer})

        return final_output


def loadPipeline() -> MyModel:
    return MyModel()


# ==============================
# EXAMPLE USAGE
# ==============================
if __name__ == "__main__":
    pipeline = loadPipeline()
    questions = [
        {
            "questionID": "c5c79dfc-8b2f-4fb3-8b4d-13b3aaf00a11",
            "subject": "history",
            "question": "What were the underlying causes of the Peloponnesian War between Athens and Sparta?"
        },
        {
            "questionID": "0f4c2ce2-40ac-44e5-bfd0-1fbf728cf4a2",
            "subject": "history",
            "question": "How did Julius Caesar assassination alter the political trajectory of the Roman Republic?"
        },
        {
            "questionID": "71afc203-8d0a-4b7b-8a62-6f53663e7558",
            "subject": "history",
            "question": "What role did monasteries play in preserving European knowledge during the Middle Ages?"
        },
        {
            "questionID": "fa6b6409-3ab4-4b00-9060-94ae964aa2cc",
            "subject": "history",
            "question": "Explain the technological innovations that enabled the Viking Age expansions."
        },
        {
            "questionID": "f70a3f77-2bd7-4c86-aaa0-f26d81b8e24f",
            "subject": "history",
            "question": "What were the socio-economic causes of the French Revolution in 1789?"
        },
        {
            "questionID": "2d2e2b22-7e2a-4758-90ec-8dd90c9d4503",
            "subject": "history",
            "question": "How did the Black Death reshape European society and labor structures?"
        },
        {
            "questionID": "eab7d2b4-721e-4d41-9e70-0e80ef96f75d",
            "subject": "history",
            "question": "What were the primary motivations behind the Crusades launched by European powers?"
        },
        {
            "questionID": "a7410900-0dfc-4b45-a9aa-7a279628f5d0",
            "subject": "history",
            "question": "Discuss the military strategies that enabled the Mongol Empire to expand so rapidly."
        },
        {
            "questionID": "6fa34971-e5e8-4e54-9fa1-851ac818b2f3",
            "subject": "history",
            "question": "Why is the Magna Carta considered a foundational document in constitutional history?"
        },
        {
            "questionID": "0d388784-4f95-44d6-9179-8d671bc313aa",
            "subject": "history",
            "question": "How did the printing press contribute to the spread of the Renaissance and Reformation?"
        },
        {
            "questionID": "243c4e58-a80e-4c8b-8cb9-bf48e4fc0f51",
            "subject": "history",
            "question": "What were the long-term consequences of the Spanish conquest of the Aztec Empire?"
        },
        {
            "questionID": "0e07ac8c-ddb4-4816-b025-d6027ddf9dd7",
            "subject": "history",
            "question": "How did the Scientific Revolution challenge traditional European worldviews?"
        },
        {
            "questionID": "e5dc6ca9-f071-4212-8161-99385fe50f4c",
            "subject": "history",
            "question": "Explain the global significance of the American Declaration of Independence."
        },
        {
            "questionID": "4ef073af-6ad1-44ad-afa4-07d069a70b38",
            "subject": "history",
            "question": "What factors accelerated the Industrial Revolution in Great Britain?"
        },
        {
            "questionID": "619e10bb-aeb6-4f93-b463-188d347adb22",
            "subject": "history",
            "question": "How did the Opium Wars alter China's relationship with Western powers?"
        },
        {
            "questionID": "f2c49cf7-9871-46a9-ab4c-8200e074293e",
            "subject": "history",
            "question": "What were the primary goals and achievements of the abolitionist movement?"
        },
        {
            "questionID": "c094fd9a-212d-4d9e-b768-1f26dc1c2c43",
            "subject": "history",
            "question": "What geopolitical tensions led to the outbreak of World War I?"
        },
        {
            "questionID": "4c5f2d29-a376-4ecc-b21e-46b648fbc34c",
            "subject": "history",
            "question": "How did trench warfare shape the outcomes of battles during World War I?"
        },
        {
            "questionID": "b970f461-4159-4638-9d6e-0a086f7bcfb8",
            "subject": "history",
            "question": "Why was the Treaty of Versailles criticized for contributing to future global conflicts?"
        },
        {
            "questionID": "6a4639d5-31f3-4e48-b97a-607abd4c20f6",
            "subject": "history",
            "question": "What were the key ideological differences between fascism and communism in the 20th century?"
        },
        {
            "questionID": "4f7a4f05-6fb5-47aa-8410-1f152faa96ff",
            "subject": "history",
            "question": "Explain how the Great Depression transformed global economic policies."
        },
        {
            "questionID": "cfa67371-142e-41e4-a16d-cd40ae25f48c",
            "subject": "history",
            "question": "What events triggered the beginning of World War II in Europe?"
        },
        {
            "questionID": "294d0f91-4cec-4bdb-8ad6-69c1480dcd28",
            "subject": "history",
            "question": "How did the Battle of Stalingrad become a turning point in World War II?"
        },
        {
            "questionID": "78b9e3ac-deb0-4d20-86c7-94184136da87",
            "subject": "history",
            "question": "What was the significance of the Nuremberg Trials in shaping international law?"
        },
        {
            "questionID": "acf4c452-122b-45d5-89e1-b5ee566a0bf2",
            "subject": "history",
            "question": "How did the Cold War reshape global alliances and political ideologies?"
        },
        {
            "questionID": "9c230980-d4f8-41cb-995e-80ce0df5e21a",
            "subject": "history",
            "question": "What were the causes and outcomes of the Cuban Missile Crisis?"
        },
        {
            "questionID": "b2e32545-f4f4-4305-a17c-20c0cfeab7fc",
            "subject": "history",
            "question": "How did the Civil Rights Movement transform American society?"
        },
        {
            "questionID": "6a94e378-0c5d-45a5-a704-8b70a7964639",
            "subject": "history",
            "question": "What historical conditions led to the formation of the United Nations in 1945?"
        },
        {
            "questionID": "2d6aac96-e1f8-4332-8078-e2b3356d44fd",
            "subject": "history",
            "question": "What factors contributed to the decolonization of Africa after World War II?"
        },
        {
            "questionID": "d056b7d0-e217-422f-ba1d-b489d58f4deb",
            "subject": "history",
            "question": "What were the political and social effects of the Iranian Revolution of 1979?"
        },
        {
            "questionID": "5d6856b5-8648-4b40-8d4c-ae2a997d7009",
            "subject": "history",
            "question": "How did the fall of the Berlin Wall signal the end of the Cold War?"
        },
        {
            "questionID": "2b4a7500-7182-48b5-96df-6a1987dc1dad",
            "subject": "history",
            "question": "What were the major consequences of the collapse of the Soviet Union in 1991?"
        },
        {
            "questionID": "0c5e6f48-3b72-4e53-b5a8-41c825a5c033",
            "subject": "history",
            "question": "How did the invention of gunpowder transform medieval warfare?"
        },
        {
            "questionID": "d0a2535e-ae76-4b1b-a559-29bb574190a1",
            "subject": "history",
            "question": "What led to the rise and fall of the Khmer Empire in Southeast Asia?"
        },
        {
            "questionID": "08cc83e8-e2b0-44af-b7b8-40cfb0dc2550",
            "subject": "history",
            "question": "What were the key achievements of the Abbasid Caliphate during the Islamic Golden Age?"
        },
        {
            "questionID": "ab42a406-64f1-4889-a996-dccc8da2d2f4",
            "subject": "history",
            "question": "How did the Bantu migrations influence cultural and linguistic development in Africa?"
        },
        {
            "questionID": "73fa609d-cc4e-4d41-b250-c8e7fbd7afc8",
            "subject": "history",
            "question": "What were the technological and maritime advancements of the Age of Exploration?"
        },
        {
            "questionID": "09d1d41f-0c7a-4ca5-a584-a58cfa816eba",
            "subject": "history",
            "question": "How did the Renaissance change artistic, scientific, and cultural thinking in Europe?"
        },
        {
            "questionID": "d3f846b1-4e02-47a7-8d2d-cde7460d6e4a",
            "subject": "history",
            "question": "Why was the Battle of Hastings in 1066 pivotal for English history?"
        },
        {
            "questionID": "c0d53c76-8fc5-465f-9f62-d94bb95f33b4",
            "subject": "history",
            "question": "How did nationalism shape the unifications of Germany and Italy in the 19th century?"
        },
        {
            "questionID": "2e38e01c-00ab-4b23-ab07-1a2f9075a655",
            "subject": "history",
            "question": "What were the causes and impacts of the Boxer Rebellion in China?"
        },
        {
            "questionID": "e7ec6e4e-fc25-46d1-bb79-7ab8fcf248ef",
            "subject": "history",
            "question": "How did the Suez Crisis of 1956 reshape Middle Eastern and global geopolitics?"
        },
        {
            "questionID": "d9f2d142-7956-4b5a-a256-299234f68e14",
            "subject": "history",
            "question": "What were the major goals and outcomes of the Non Aligned Movement during the Cold War?"
        },
        {
            "questionID": "590bf250-0fc8-469b-9c2e-3d24262f5194",
            "subject": "history",
            "question": "How did the Green Revolution transform agriculture in developing countries?"
        },
        {
            "questionID": "f2842d3d-165e-4e75-916f-e487e7a37433",
            "subject": "history",
            "question": "What were the cultural and political influences of the Byzantine Empire on Eastern Europe?"
        },
        {
            "questionID": "9a292514-53eb-4f8b-a960-1502dfb69a11",
            "subject": "geography",
            "question": "What geological processes are responsible for the formation of rift valleys?"
        },
        {
            "questionID": "9d986e30-49a9-4cf8-b957-ae01aa0f7d39",
            "subject": "geography",
            "question": "How do ocean currents influence global climate patterns?"
        },
        {
            "questionID": "2b927b12-5e5e-4310-a39c-30460d5f72b1",
            "subject": "geography",
            "question": "What factors determine the distribution of major world biomes?"
        },
        {
            "questionID": "125be522-9f25-4796-a01e-aaee801f4bb2",
            "subject": "geography",
            "question": "How do plate tectonics contribute to the formation of mountains like the Himalayas?"
        },
        {
            "questionID": "1fdcac7d-4ebe-43f8-b0f6-86444041e4fa",
            "subject": "geography",
            "question": "What environmental conditions lead to the development of Mediterranean climates?"
        },
        {
            "questionID": "c4ccda49-2869-4c52-a909-9a51491de741",
            "subject": "geography",
            "question": "How do monsoon systems affect agricultural activities in South Asia?"
        },
        {
            "questionID": "f9b8c9d5-cc3a-4009-91b8-683083429683",
            "subject": "geography",
            "question": "What are the main causes and consequences of desertification in the Sahel region?"
        },
        {
            "questionID": "cc9fc277-8c22-4eb6-b5c0-d4fd1e1c6532",
            "subject": "geography",
            "question": "Why are river deltas such as the Nile and Ganges highly productive agricultural regions?"
        },
        {
            "questionID": "97798a9c-5622-4c91-97b9-58028b35de8a",
            "subject": "geography",
            "question": "How do tectonic hotspots create island chains like Hawaii?"
        },
        {
            "questionID": "d3d8a578-f297-4b7f-baca-1562a664f71f",
            "subject": "geography",
            "question": "What physical and climatic factors influence the distribution of world population?"
        },
        {
            "questionID": "a1b5df0f-8fef-49c9-b4cc-515ce6ff34c7",
            "subject": "geography",
            "question": "What are the major characteristics of tropical rainforest ecosystems?"
        },
        {
            "questionID": "9b8faa46-cc91-4c2b-9975-7a8b6cc3dcce",
            "subject": "geography",
            "question": "How do wind patterns contribute to the creation of rain shadows?"
        },
        {
            "questionID": "8c98e61d-a65c-4919-b015-7a81a8e6d630",
            "subject": "geography",
            "question": "What environmental issues affect the sustainability of the Amazon Basin?"
        },
        {
            "questionID": "b54939f5-d3fb-464f-b271-afac57244cc9",
            "subject": "geography",
            "question": "How do river systems shape economic development in continental regions?"
        },
        {
            "questionID": "bf6ad776-a6c8-42a7-a427-7e023dc021e0",
            "subject": "geography",
            "question": "What factors explain why some countries have high volcanic activity while others do not?"
        },
        {
            "questionID": "05cb1f3b-5706-4b16-b2ce-f513726b2bb4",
            "subject": "geography",
            "question": "How has climate change affected Arctic ice coverage and global sea levels?"
        },
        {
            "questionID": "d435223c-7e7b-4cfc-80e9-c3d4e4867f7c",
            "subject": "geography",
            "question": "What geological factors determine the location of major mineral deposits?"
        },
        {
            "questionID": "b3a3cb11-0af4-4d94-a5b8-dbeb7c10f2f1",
            "subject": "geography",
            "question": "How do urbanization patterns differ between developed and developing nations?"
        },
        {
            "questionID": "5b8bb947-4e63-4f01-8698-ae41e5e518ab",
            "subject": "geography",
            "question": "What natural processes contribute to soil fertility in volcanic regions?"
        },
        {
            "questionID": "2181e671-fede-4d6c-8fac-540635b8a799",
            "subject": "geography",
            "question": "How do fjords form, and which regions of the world are most known for them?"
        },
        {
            "questionID": "30e43576-28a4-4b31-9e93-bb30f83a48f9",
            "subject": "geography",
            "question": "What is the significance of the Tropic of Cancer and Tropic of Capricorn in global climate?"
        },
        {
            "questionID": "4641fb55-97a0-4eba-b3ed-5cac724c8b06",
            "subject": "geography",
            "question": "What factors influence the creation of coral reefs and their biodiversity?"
        },
        {
            "questionID": "470c9320-8a6b-4555-9e44-a4ef4d330191",
            "subject": "geography",
            "question": "How do human activities contribute to coastal erosion?"
        },
        {
            "questionID": "aa4a1c1c-7201-4b38-9579-7ae89b17f98d",
            "subject": "geography",
            "question": "How do mountain ranges affect transportation and economic development?"
        },
        {
            "questionID": "91681bf2-78e8-4324-a0cb-962480555dab",
            "subject": "geography",
            "question": "What geographical factors influence the spread of diseases like malaria?"
        },
        {
            "questionID": "b232eae8-ddea-465d-a210-af7320fc19ae",
            "subject": "geography",
            "question": "How do river flooding cycles support agriculture in floodplain regions?"
        },
        {
            "questionID": "6c6578c0-1ae2-43af-b478-65255375e0f4",
            "subject": "geography",
            "question": "What factors determine the boundaries of Earth tectonic plates?"
        },
        {
            "questionID": "5e8a4b7a-2f74-48af-aa7f-0d082d31aaa2",
            "subject": "geography",
            "question": "How do glaciers erode and shape landscapes over long periods?"
        },
        {
            "questionID": "d8e1e667-aee9-4894-84eb-540cac1e6e53",
            "subject": "geography",
            "question": "What geographical features contribute to the arid climate of deserts like the Sahara?"
        },
        {
            "questionID": "50fe2936-d6fb-41cf-a973-6571613e85a6",
            "subject": "geography",
            "question": "How do human settlement patterns develop around major rivers?"
        },
        {
            "questionID": "f4671577-9c1e-4074-9f70-a8d99d9592bd",
            "subject": "geography",
            "question": "What is the role of the jet stream in influencing weather across continents?"
        },
        {
            "questionID": "a4883da8-7a66-4504-af9b-a130f0db4d74",
            "subject": "geography",
            "question": "How do monsoon winds shape the seasonal climate of Southeast Asia?"
        },
        {
            "questionID": "838aeae9-6d27-44d3-9130-fd12a43948d4",
            "subject": "geography",
            "question": "What environmental and geographical factors contribute to the formation of savannas?"
        },
        {
            "questionID": "8d9266cf-17bc-4549-b7b9-f108dd35f7ac",
            "subject": "geography",
            "question": "How do volcanic eruptions influence short-term and long-term climate patterns?"
        },
        {
            "questionID": "39e06a6a-e401-4b23-973a-84fdc5bcd6ff",
            "subject": "geography",
            "question": "What factors determine the salinity levels of major oceans and seas?"
        },
        {
            "questionID": "7188bcbf-bd4a-4eb6-96af-52525177ead2",
            "subject": "geography",
            "question": "How does the El Nino-Southern Oscillation alter weather patterns around the world?"
        },
        {
            "questionID": "95039038-e257-4d2b-aebd-7d89312309d0",
            "subject": "geography",
            "question": "What geographical features contribute to biodiversity hotspots like Madagascar?"
        },
        {
            "questionID": "5d5fa004-9a66-43cd-9891-e1efae925815",
            "subject": "geography",
            "question": "How do human land use changes impact natural hydrological cycles?"
        },
        {
            "questionID": "72e12f9d-a44f-44e7-9aaa-77b024d8f79d",
            "subject": "geography",
            "question": "Why are estuaries among the most productive ecosystems on Earth?"
        },
        {
            "questionID": "162adc29-63f0-4249-8669-5f4d69038e24",
            "subject": "geography",
            "question": "How do climate zones influence patterns of agriculture around the world?"
        }
    ]
    start_time = time.perf_counter()
    answers = pipeline(questions)
    elapsed = time.perf_counter() - start_time
    print(answers)
    print(f"[TOTAL LATENCY] {elapsed*1000:.2f} ms")