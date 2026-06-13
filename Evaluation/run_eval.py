import os, sys, json, traceback

# Add parent folder so we can import retrieval.py and ui.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval import search_pipeline
from ui import generate_grounded_answer

# Accept dataset filename as argument, default to dataset2.json
EVAL_DIR   = os.path.dirname(os.path.abspath(__file__))
DATASET    = sys.argv[1] if len(sys.argv) > 1 else "dataset2.json"
INPUT_PATH = os.path.join(EVAL_DIR, DATASET)
OUTPUT_PATH = os.path.join(EVAL_DIR, DATASET.replace(".json", "_results.json"))


def run():
    with open(INPUT_PATH, "r") as f:
        dataset = json.load(f)

    print(f"Running {len(dataset)} queries from {DATASET}...\n")
    results = []

    for i, item in enumerate(dataset):
        query   = item["query"]
        history = item.get("conversation_history", None)
        print(f"[{i+1}/{len(dataset)}] {query}")

        try:
            decision, rewritten, chunks = search_pipeline(query, history=history)

            answer = generate_grounded_answer(
                query=query,
                context_chunks=chunks,
                history=history,
                is_refinement=(decision == "REFINE")
            )

            result = {
                "query":               query,
                "gold_answer":         item["gold_standard_answer"],
                "target_chunks":       item["target_chunks"],
                "decision":            decision,
                "rewritten_query":     rewritten,
                "retrieved_chunks":    [{"source": c["metadata"]["source"],
                                         "page":   c["metadata"]["page"],
                                         "score":  round(c.get("rerank_score", 0), 4)} for c in chunks],
                "generated_answer":    answer,
            }
            # Carry over optional fields if present
            if "_category"        in item: result["category"]         = item["_category"]
            if "expected_decision" in item: result["expected_decision"] = item["expected_decision"]

            results.append(result)
            print(f"  → Decision: {decision} | Chunks retrieved: {len(chunks)}\n")

        except Exception as e:
            results.append({"query": query, "error": str(e)})
            traceback.print_exc()

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Done. Results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    run()
