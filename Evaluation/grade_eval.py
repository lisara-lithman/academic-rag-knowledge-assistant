import os, sys, json

EVAL_DIR     = os.path.dirname(os.path.abspath(__file__))
RESULTS_FILE = sys.argv[1] if len(sys.argv) > 1 else "evaluation_results.json"
RESULTS_PATH = os.path.join(EVAL_DIR, RESULTS_FILE)

REFUSAL_PHRASES = ["cannot find", "not covered", "not mentioned",
                   "not in the module", "no information", "outside"]


def get_fields(item):
    """Reads fields from either the old nested format or the new flat format."""
    if "pipeline_output" in item:
        # Old format: fields nested inside pipeline_output
        p = item["pipeline_output"]
        return (
            p.get("decision"),
            p.get("retrieved_chunks", []),
            p.get("generated_answer", ""),
        )
    else:
        # New flat format from simplified run_eval.py
        return (
            item.get("decision"),
            item.get("retrieved_chunks", []),
            item.get("generated_answer", ""),
        )


def check_retrieval(item):
    """
    Returns (success: bool|None, status_label: str)
    None means N/A (skip from count).
    """
    category         = item.get("category", "STANDARD")
    target_chunks    = item["target_chunks"]
    _, retrieved, generated_answer = get_fields(item)

    # Multi-turn REFINE: no retrieval expected
    if category == "MULTI_TURN" and item.get("expected_decision") == "REFINE":
        return None, "N/A"

    # Out-of-scope: check LLM refused to answer
    if category == "OUT_OF_SCOPE":
        refused = any(p in generated_answer.lower() for p in REFUSAL_PHRASES)
        return refused, "REFUSED✓" if refused else "HALLUCINATED"

    # Standard: check if any target slide was retrieved
    for target in target_chunks:
        for ret in retrieved:
            if target["source"].lower() == ret["source"].lower() and target["page"] == ret["page"]:
                return True, "HIT"
    return False, "MISS"


def check_routing(item):
    expected = item.get("expected_decision", "SEARCH")
    decision, _, _ = get_fields(item)
    ok = (decision == expected)
    return ok, "OK" if ok else "FAIL"



def grade():
    with open(RESULTS_PATH, "r") as f:
        results = json.load(f)

    retrieval_hits = routing_hits = skipped = 0
    failures = []

    print("=" * 75)
    print(f"REPORT — {RESULTS_FILE} ({len(results)} test cases)")
    print("=" * 75)
    print(f"{'#':<3} | {'Category':<18} | {'Query':<36} | {'Retrieval':<11} | {'Routing'}")
    print("-" * 75)

    for i, item in enumerate(results):
        if "error" in item:
            print(f"{i+1:<3} | {'ERROR':<18} | {item['query'][:36]:<36} | {'ERROR':<11} | ERROR")
            continue

        ret_ok, ret_label  = check_retrieval(item)
        rout_ok, rout_label = check_routing(item)

        if ret_ok is None:
            skipped += 1
        elif ret_ok:
            retrieval_hits += 1

        if rout_ok:
            routing_hits += 1

        print(f"{i+1:<3} | {item.get('category','STANDARD'):<18} | {item['query'][:36]:<36} | {ret_label:<11} | {rout_label}")

        if not rout_ok or ret_ok is False:
            reason = []
            if not rout_ok:
                reason.append(f"Routing FAIL (expected {item.get('expected_decision','SEARCH')}, got {item.get('decision')})")
            if ret_ok is False:
                if item.get("category") == "OUT_OF_SCOPE":
                    reason.append("LLM did not refuse out-of-scope question")
                else:
                    targets = [f"{t['source']} p.{t['page']}" for t in item["target_chunks"]]
                    got     = [f"{r['source']} p.{r['page']}" for r in item.get("retrieved_chunks", [])]
                    reason.append(f"Retrieval MISS | Expected: {targets} | Got: {got}")
            failures.append({"query": item["query"], "category": item.get("category",""), "reason": " & ".join(reason)})

    # Summary
    gradeable = len(results) - skipped
    print("=" * 75)
    print(f"Retrieval Score  : {retrieval_hits}/{gradeable} = {retrieval_hits/gradeable*100:.1f}%  ({skipped} N/A skipped)")
    print(f"Routing Accuracy : {routing_hits}/{len(results)} = {routing_hits/len(results)*100:.1f}%")
    print("=" * 75)

    if failures:
        print("\n--- FAILURES ---")
        for f in failures:
            print(f"\n[{f['category']}] {f['query']}")
            print(f"  {f['reason']}")
    else:
        print("\n✅ No failures.")


if __name__ == "__main__":
    grade()
