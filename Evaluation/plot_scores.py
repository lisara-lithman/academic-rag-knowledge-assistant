import json
import os
import matplotlib.pyplot as plt
import numpy as np

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
SCORES_PATH = os.path.join(EVAL_DIR, "ragas_scores.json")
OUTPUT_PATH = os.path.join(EVAL_DIR, "ragas_improvement.png")

def main():
    if not os.path.exists(SCORES_PATH):
        print(f"Error: {SCORES_PATH} not found.")
        return

    with open(SCORES_PATH, "r") as f:
        data = json.load(f)

    if "history" not in data:
        print("Error: No history found in scores file.")
        return

    iterations = []
    faithfulness = []
    answer_relevancy = []
    context_precision = []
    context_recall = []
    overall = []

    for entry in data["history"]:
        iter_name = entry["iteration"].split(" (")[0] # e.g. "Iteration 1"
        if iter_name == "Baseline":
             iter_name = "Baseline\n(No Fixes)"
        elif "Prompt" in entry["iteration"]:
             iter_name = "Iteration 1\n(Prompt Fix)"
        elif "Dual" in entry["iteration"]:
             iter_name = "Iteration 2\n(Dual-Query)"
        elif "Cohere" in entry["iteration"]:
             iter_name = "Iteration 3\n(Cohere Reranker)"
        elif "OpenAI" in entry["iteration"]:
             iter_name = "Iteration 4\n(OpenAI Embedder)"
        elif "GPT-4o LLM" in entry["iteration"]:
             iter_name = "Iteration 5\n(GPT-4o LLM)"
        elif "Strict" in entry["iteration"]:
             iter_name = "Iteration 6\n(Strict Prompt)"
        elif "Semantic" in entry["iteration"]:
             iter_name = "Iteration 7\n(Semantic Chunk)"
             
        iterations.append(iter_name)
        scores = entry["scores"]
        faithfulness.append(scores["faithfulness"])
        answer_relevancy.append(scores["answer_relevancy"])
        context_precision.append(scores["context_precision"])
        context_recall.append(scores["context_recall"])
        overall.append(scores["overall_ragas"])

    x = np.arange(len(iterations))
    width = 0.15

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot bars
    rects1 = ax.bar(x - width*2, faithfulness, width, label='Faithfulness', color='#3b82f6')
    rects2 = ax.bar(x - width, answer_relevancy, width, label='Answer Relevance', color='#10b981')
    rects3 = ax.bar(x, context_precision, width, label='Context Precision', color='#8b5cf6')
    rects4 = ax.bar(x + width, context_recall, width, label='Context Recall', color='#f59e0b')
    rects5 = ax.bar(x + width*2, overall, width, label='Overall RAGAS', color='#ef4444', alpha=0.9)

    ax.set_ylabel('Score (0.0 to 1.0)', fontsize=12, fontweight='bold')
    ax.set_title('RAGAS Evaluation Metrics Across Iterations', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(iterations, fontsize=11)
    ax.set_ylim(0.4, 1.05)
    
    # Grid lines behind bars
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='dashed', alpha=0.3)
    
    # Legend outside plot
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Add text labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', rotation=90, fontsize=9)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)
    autolabel(rects5)

    fig.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
    print(f"Chart saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
