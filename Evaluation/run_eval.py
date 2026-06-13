import os
import sys
import json
import traceback

# Setup paths to import retrieval and ui from the parent workspace root
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(EVAL_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from retrieval import search_pipeline
from ui import generate_grounded_answer

GOLDEN_DATASET_PATH = os.path.join(EVAL_DIR, "dataset2.json")
RESULTS_PATH = os.path.join(EVAL_DIR, "evaluation_results.json")

def run_evaluation():
    # 1. Load the golden dataset
    if not os.path.exists(GOLDEN_DATASET_PATH):
        print(f"Error: {GOLDEN_DATASET_PATH} not found!")
        return

    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"Loaded {len(dataset)} evaluation queries from {GOLDEN_DATASET_PATH}.\n")

    results = []

    # 2. Iterate through each query and run the pipeline
    for idx, item in enumerate(dataset):
        query = item["query"]
        print(f"[{idx+1}/{len(dataset)}] Processing Query: '{query}'")

        try:
            # Step A: Run the search pipeline
            decision, rewritten_query, retrieved_chunks = search_pipeline(query, history=None)

            # Step B: Generate the answer using the retrieved context
            generated_answer = generate_grounded_answer(
                query=query, 
                context_chunks=retrieved_chunks, 
                history=None, 
                is_refinement=False
            )

            # Step C: Format retrieved chunks for logging
            retrieved_info = []
            for chunk in retrieved_chunks:
                retrieved_info.append({
                    "id": chunk.get("id"),
                    "source": chunk["metadata"].get("source"),
                    "page": chunk["metadata"].get("page"),
                    "rerank_score": chunk.get("rerank_score", 0.0),
                    "text_snippet": chunk.get("text", "")[:150] + "..."
                })

            # Save results
            results.append({
                "query": query,
                "gold_standard_answer": item["gold_standard_answer"],
                "target_chunks": item["target_chunks"],
                "pipeline_output": {
                    "decision": decision,
                    "rewritten_query": rewritten_query,
                    "retrieved_chunks": retrieved_info,
                    "generated_answer": generated_answer
                }
            })
            print(f"Success! Retrieved {len(retrieved_chunks)} chunks.\n")

        except Exception as e:
            print(f"ERROR processing query: {e}")
            traceback.print_exc()
            results.append({
                "query": query,
                "gold_standard_answer": item["gold_standard_answer"],
                "target_chunks": item["target_chunks"],
                "pipeline_output": {
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
            })
            print()

    # 3. Save the results to evaluation_results.json
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Evaluation complete! Results saved to {RESULTS_PATH}")

if __name__ == "__main__":
    run_evaluation()
