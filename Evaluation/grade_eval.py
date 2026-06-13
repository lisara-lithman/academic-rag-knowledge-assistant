import os
import json

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(EVAL_DIR, "evaluation_results.json")

def grade_evaluation():
    if not os.path.exists(RESULTS_PATH):
        print(f"Error: {RESULTS_PATH} not found! Run the benchmark script first.")
        return

    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        results = json.load(f)

    total_queries = len(results)
    retrieval_hits = 0
    routing_hits = 0
    failures = []

    print("=" * 70)
    print(f"RAG EVALUATION REPORT: {total_queries} Test Cases")
    print("=" * 70)
    print(f"{'#':<3} | {'Query':<50} | {'Retrieval':<9} | {'Routing':<7}")
    print("-" * 70)

    for idx, item in enumerate(results):
        query = item["query"]
        target_chunks = item["target_chunks"]
        pipeline = item.get("pipeline_output", {})

        if "error" in pipeline:
            print(f"{idx+1:<3} | {query[:50]:<50} | {'ERROR':<9} | {'ERROR':<7}")
            failures.append({
                "query": query,
                "reason": f"Pipeline Error: {pipeline['error']}"
            })
            continue

        decision = pipeline.get("decision")
        retrieved_chunks = pipeline.get("retrieved_chunks", [])

        # 1. Evaluate Routing
        router_success = (decision == "SEARCH")
        if router_success:
            routing_hits += 1

        # 2. Evaluate Retrieval
        retrieval_success = False
        for target in target_chunks:
            target_src = target["source"].lower()
            target_pg = target["page"]
            
            for ret in retrieved_chunks:
                ret_src = ret["source"].lower()
                ret_pg = ret["page"]
                
                # Match source slide and page number
                if target_src == ret_src and target_pg == ret_pg:
                    retrieval_success = True
                    break
            if retrieval_success:
                break

        if retrieval_success:
            retrieval_hits += 1

        # Print row status
        ret_status = "HIT" if retrieval_success else "MISS"
        rout_status = "OK" if router_success else "FAIL"
        print(f"{idx+1:<3} | {query[:50]:<50} | {ret_status:<9} | {rout_status:<7}")

        # Track failure reasons for summary
        if not retrieval_success or not router_success:
            reason = []
            if not retrieval_success:
                targets_str = ", ".join([f"{t['source']} p.{t['page']}" for t in target_chunks])
                retrieved_str = ", ".join([f"{r['source']} p.{r['page']}" for r in retrieved_chunks])
                reason.append(f"Retrieval MISS (Expected: [{targets_str}], Got: [{retrieved_str}])")
            if not router_success:
                reason.append(f"Routing FAIL (Expected SEARCH, Got {decision})")
            
            failures.append({
                "query": query,
                "reason": " & ".join(reason),
                "answer_snippet": pipeline.get("generated_answer", "")[:120] + "..."
            })

    # 3. Calculate Overall Metrics
    overall_recall = (retrieval_hits / total_queries) * 100
    overall_routing = (routing_hits / total_queries) * 100

    print("=" * 70)
    print("SUMMARY METRICS:")
    print("-" * 70)
    print(f"Overall Retrieval Recall (Recall@K): {overall_recall:.1f}% ({retrieval_hits}/{total_queries} Hits)")
    print(f"Overall Query Routing Accuracy   : {overall_routing:.1f}% ({routing_hits}/{total_queries} Correct)")
    print("=" * 70)

    # 4. Detail Failures/Defects
    if failures:
        print("\n" + "!" * 30 + " DETECTED FAILURES & DEFECTS " + "!" * 30)
        for f in failures:
            print(f"\nQuery: '{f['query']}'")
            print(f"  Reason : {f['reason']}")
            if "answer_snippet" in f:
                print(f"  Answer : \"{f['answer_snippet']}\"")
        print("!" * 89)

if __name__ == "__main__":
    grade_evaluation()
