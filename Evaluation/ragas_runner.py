"""
ragas_runner.py — RAGAS Evaluation for the Academic RAG Knowledge Assistant

Reads pre-computed pipeline results from dataset3_fixed_results.json,
fetches the actual slide text from ChromaDB, and evaluates using the
RAGAS framework (v0.4.x) with OpenAI GPT-4o-mini as the judge LLM.

NOTE on RAGAS v0.4.x API:
  The new ragas.metrics.collections classes inherit from BaseMetric, which is
  NOT the Metric base class that ragas.evaluate() type-checks against. We
  therefore use the classic pre-instantiated singleton metrics from ragas.metrics,
  which do inherit from Metric, and inject the LLM after import.
  This is still supported (deprecated but not yet removed) in v0.4.x.

Requires: ragas, langchain-openai, eval_type_backport (for Python 3.9)

Metrics evaluated:
  - Faithfulness    : Is the answer grounded in the retrieved slides?
  - AnswerRelevancy : Does the answer address the question asked?
  - ContextPrecision: Are the retrieved slides relevant to the question?
  - ContextRecall   : Does the context contain what's needed to answer?

Usage:
    python Evaluation/ragas_runner.py
"""

import os
import sys
import json
import warnings
from typing import Optional

# Suppress noisy third-party warnings that don't affect functionality
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ── path setup so we can reach vector_db from Evaluation/ ──────────────────
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import chromadb

# ── OpenAI native client (required by RAGAS v0.4 llm_factory) ──────────────
from openai import OpenAI

# ── RAGAS — use classic pre-instantiated metrics (compatible with evaluate()) ──
# The new ragas.metrics.collections classes inherit from a different base class
# (BaseMetric) that is NOT recognised by ragas.evaluate()'s isinstance check.
# The classic singletons from ragas.metrics inherit from Metric and pass the check.
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.llms import llm_factory
from langchain_openai import OpenAIEmbeddings as LCOpenAIEmbeddings
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

# ── Config ──────────────────────────────────────────────────────────────────
EVAL_DIR        = os.path.dirname(os.path.abspath(__file__))
# Accept optional results filename argument — defaults to the Phase 4 baseline
_DEFAULT_RESULTS = "dataset3_fixed_results.json"
_results_file    = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_RESULTS
RESULTS_PATH    = os.path.join(EVAL_DIR, _results_file)
DB_DIR          = os.getenv("PERSIST_DIRECTORY", "./vector_db").strip('"')
COLLECTION_NAME = "operating_systems"
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")

# Categories that RAGAS cannot evaluate (no context retrieved / no real answer)
SKIP_CATEGORIES = {"OUT_OF_SCOPE"}


def should_skip(item: dict) -> bool:
    """Return True for rows that RAGAS cannot meaningfully evaluate."""
    category = item.get("category", "STANDARD")
    if category in SKIP_CATEGORIES:
        return True
    # Multi-turn REFINE: router didn't search → no context retrieved
    if category == "MULTI_TURN" and item.get("expected_decision") == "REFINE":
        return True
    return False


def load_chromadb():
    """Connect to the existing ChromaDB vector store."""
    print("Connecting to ChromaDB...")
    client     = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_collection(COLLECTION_NAME)
    print(f"  → Collection '{COLLECTION_NAME}' loaded ({collection.count()} chunks)\n")
    return collection


def get_chunk_text(collection, source: str, page: int) -> Optional[str]:
    """
    Fetch the text of a specific slide from ChromaDB by source and page number.
    Returns None if the chunk is not found.
    """
    results = collection.get(
        where={"$and": [{"source": source}, {"page": page}]},
        include=["documents", "metadatas"],
    )
    if results["documents"]:
        return results["documents"][0]
    return None


def build_ragas_samples(results: list, collection) -> list:
    """
    Convert pipeline result rows into RAGAS SingleTurnSample objects.
    Fetches the full slide text from ChromaDB for each retrieved chunk.
    Returns a list of (original_item, SingleTurnSample) tuples.
    """
    paired  = []
    skipped = []

    for item in results:
        # ── filter unevaluable rows ─────────────────────────────────────────
        if should_skip(item):
            skipped.append(item.get("query", "?"))
            continue

        # ── pull retrieved chunk texts from ChromaDB ────────────────────────
        contexts = []
        for chunk in item.get("retrieved_chunks", []):
            text = get_chunk_text(collection, chunk["source"], chunk["page"])
            if text:
                contexts.append(text)

        # If somehow no context was retrieved, skip gracefully
        if not contexts:
            skipped.append(item.get("query", "?") + " [no contexts]")
            continue

        sample = SingleTurnSample(
            user_input         = item["query"],
            response           = item.get("generated_answer", ""),
            retrieved_contexts = contexts,
            reference          = item.get("gold_answer", ""),
        )
        paired.append((item, sample))

    print(f"  → Evaluating {len(paired)} questions with RAGAS")
    print(f"  → Skipping   {len(skipped)} questions (OUT_OF_SCOPE / REFINE)\n")
    for s in skipped:
        print(f"     [SKIP] {s[:72]}")
    print()

    return paired


def print_report(items_evaluated: list, scores_df):
    """Print a clean, formatted RAGAS evaluation report to the terminal."""

    metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    metric_labels = {
        "faithfulness"      : "Faithfulness      ",
        "answer_relevancy"  : "Answer Relevance  ",
        "context_precision" : "Context Precision ",
        "context_recall"    : "Context Recall    ",
    }

    def interpret(score: float) -> str:
        if score >= 0.85: return "✅ Excellent"
        if score >= 0.70: return "⚠️  Good"
        return "❌  Needs Work"

    # ── Overall metric averages ─────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  RAGAS EVALUATION REPORT — dataset3_fixed_results.json")
    print("  Judge LLM  : OpenAI GPT-4o-mini")
    print(f"  Questions  : {len(scores_df)} evaluated  |  5 skipped (OOS / Refine)")
    print("=" * 65)
    print(f"  {'Metric':<22} {'Score':>6}   Interpretation")
    print("  " + "-" * 55)

    valid_cols = [c for c in metric_cols if c in scores_df.columns]
    overall_sum = 0.0
    for col in valid_cols:
        avg = scores_df[col].dropna().mean()
        overall_sum += avg
        print(f"  {metric_labels[col]:<22} {avg:>6.3f}   {interpret(avg)}")

    overall = overall_sum / len(valid_cols) if valid_cols else 0.0
    print("  " + "-" * 55)
    print(f"  {'Overall RAGAS Score':<22} {overall:>6.3f}   {interpret(overall)}")
    print("=" * 65)

    # ── Per-category breakdown ───────────────────────────────────────────────
    categories: dict = {}
    for item, row in zip(items_evaluated, scores_df.itertuples()):
        cat = item.get("category", "STANDARD")
        if cat not in categories:
            categories[cat] = {"faithfulness": [], "answer_relevancy": []}
        f_val  = getattr(row, "faithfulness",    float("nan"))
        ar_val = getattr(row, "answer_relevancy", float("nan"))
        if f_val  == f_val:  categories[cat]["faithfulness"].append(f_val)
        if ar_val == ar_val: categories[cat]["answer_relevancy"].append(ar_val)

    print("\n  By Category:")
    print(f"  {'Category':<24} {'Faithfulness':>13} {'Ans. Relevance':>15}")
    print("  " + "-" * 55)
    for cat, vals in categories.items():
        f_avg  = sum(vals["faithfulness"])     / len(vals["faithfulness"])     if vals["faithfulness"]     else float("nan")
        ar_avg = sum(vals["answer_relevancy"]) / len(vals["answer_relevancy"]) if vals["answer_relevancy"] else float("nan")
        f_str  = f"{f_avg:.3f}"  if f_avg  == f_avg  else "N/A"
        ar_str = f"{ar_avg:.3f}" if ar_avg == ar_avg else "N/A"
        print(f"  {cat:<24} {f_str:>13} {ar_str:>15}")

    print("=" * 65 + "\n")


def save_scores(scores_df, items_evaluated: list):
    """
    Persist RAGAS scores to ragas_scores.json after every run.
    Overwrites the previous file so there is always one clean record.
    """
    from datetime import datetime

    metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    # ── Overall averages ─────────────────────────────────────────────────────
    overall = {}
    valid_scores = []
    for col in metric_cols:
        if col in scores_df.columns:
            avg = float(scores_df[col].dropna().mean())
            overall[col] = round(avg, 4)
            valid_scores.append(avg)
    overall["overall_ragas"] = round(sum(valid_scores) / len(valid_scores), 4) if valid_scores else None

    # ── Per-category averages ────────────────────────────────────────────────
    categories: dict = {}
    for item, row in zip(items_evaluated, scores_df.itertuples()):
        cat = item.get("category", "STANDARD")
        if cat not in categories:
            categories[cat] = {"faithfulness": [], "answer_relevancy": []}
        f_val  = getattr(row, "faithfulness",    float("nan"))
        ar_val = getattr(row, "answer_relevancy", float("nan"))
        if f_val  == f_val:  categories[cat]["faithfulness"].append(f_val)
        if ar_val == ar_val: categories[cat]["answer_relevancy"].append(ar_val)

    cat_summary = {}
    for cat, vals in categories.items():
        cat_summary[cat] = {
            "faithfulness":    round(sum(vals["faithfulness"])     / len(vals["faithfulness"]),     4) if vals["faithfulness"]     else None,
            "answer_relevancy": round(sum(vals["answer_relevancy"]) / len(vals["answer_relevancy"]), 4) if vals["answer_relevancy"] else None,
        }

    record = {
        "meta": {
            "dataset":              "dataset3_fixed_results.json",
            "questions_evaluated":  len(scores_df),
            "questions_skipped":    5,
            "skipped_reason":       "OUT_OF_SCOPE (3) and MULTI_TURN REFINE (2)",
            "judge_llm":            "OpenAI GPT-4o-mini",
            "judge_embeddings":     "OpenAI text-embedding-3-small",
            "run_date":             datetime.now().strftime("%Y-%m-%d %H:%M"),
        },
        "overall_scores":  overall,
        "by_category":     cat_summary,
    }

    save_path = os.path.join(EVAL_DIR, "ragas_scores.json")
    with open(save_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"  Scores saved → {save_path}\n")


def run():
    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY not found in .env")
        sys.exit(1)

    # ── Load existing pipeline results ───────────────────────────────────────
    print(f"\nLoading results from: {RESULTS_PATH}")
    with open(RESULTS_PATH, "r") as f:
        results = json.load(f)
    print(f"  → {len(results)} total results loaded\n")

    # ── Connect to ChromaDB ──────────────────────────────────────────────────
    collection = load_chromadb()

    # ── Build RAGAS samples ──────────────────────────────────────────────────
    print("Building RAGAS evaluation samples...")
    paired = build_ragas_samples(results, collection)
    items_evaluated = [p[0] for p in paired]
    ragas_samples   = [p[1] for p in paired]

    if not ragas_samples:
        print("No evaluable samples found. Exiting.")
        sys.exit(1)

    dataset = EvaluationDataset(samples=ragas_samples)

    # ── Build the RAGAS-native OpenAI LLM + LangChain embeddings ────────────
    # - llm_factory creates the InstructorLLM needed by the old metric singletons
    # - LCOpenAIEmbeddings (LangChain) has embed_query() which answer_relevancy needs
    print("Configuring OpenAI GPT-4o-mini as judge LLM...")
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    judge_llm     = llm_factory("gpt-4o-mini", client=openai_client)
    judge_emb     = LCOpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=OPENAI_API_KEY,
    )

    # ── Inject LLM into the classic pre-instantiated metric singletons ───────
    # These singletons inherit from ragas.metrics.Metric, which is what
    # ragas.evaluate() type-checks against via isinstance(m, Metric).
    faithfulness.llm       = judge_llm
    answer_relevancy.llm   = judge_llm
    answer_relevancy.embeddings = judge_emb
    context_precision.llm  = judge_llm
    context_recall.llm     = judge_llm

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    print("  → Ready\n")

    # ── Run RAGAS evaluation ─────────────────────────────────────────────────
    print("Running RAGAS evaluation (this takes ~2-3 minutes)...")
    print("  Metrics: Faithfulness | Answer Relevance | Context Precision | Context Recall\n")

    eval_result = evaluate(
        dataset = dataset,
        metrics = metrics,
    )

    scores_df = eval_result.to_pandas()

    # ── Print the final report ───────────────────────────────────────────────
    print_report(items_evaluated, scores_df)

    # ── Auto-save scores to JSON for permanent record ────────────────────────
    save_scores(scores_df, items_evaluated)


if __name__ == "__main__":
    run()
