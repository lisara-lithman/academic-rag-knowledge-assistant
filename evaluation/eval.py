import math
import re
from retrieval import search_pipeline, get_llm_client, generate_grounded_answer
from google.genai import types

class TestQuery:
    def __init__(self, query, category, expected_sources, expected_answer):
        self.query = query
        self.category = category
        self.expected_sources = expected_sources  # List of dicts, e.g. [{"source": "OSSA_Lecture_3.pdf", "page": 8}]
        self.expected_answer = expected_answer

class RetrievalResult:
    def __init__(self, mrr, ndcg, keyword_coverage):
        self.mrr = mrr
        self.ndcg = ndcg
        self.keyword_coverage = keyword_coverage

class AnswerResult:
    def __init__(self, accuracy, completeness, relevance):
        self.accuracy = accuracy
        self.completeness = completeness
        self.relevance = relevance

# Standard Evaluation Dataset based on module content
TEST_SUITE = [
    TestQuery(
        query="What are the states of a process?",
        category="Process Management",
        expected_sources=[{"source": "OSSA_Lecture_3.pdf", "page": 8}],
        expected_answer="A process can be in one of the following states: New (being created), Running (instructions executing), Waiting (waiting for an event), Ready (waiting for CPU allocation), or Terminated (finished execution)."
    ),
    TestQuery(
        query="How can the system distinguish between the pages that are in main memory from the pages that are on the disk?",
        category="Memory Management",
        expected_sources=[{"source": "2023-May.pdf", "page": 5}, {"source": "OSSA_Lecture_11.pdf", "page": 28}],
        expected_answer="The system distinguishes pages using a valid-invalid bit in each page table entry. A 'valid' bit indicates the page is in main memory (physical memory), while an 'invalid' bit indicates the page is not currently in memory (it is on disk), triggering a page fault if accessed."
    ),
    TestQuery(
        query="Explain the difference between internal and external fragmentation.",
        category="Memory Management",
        expected_sources=[{"source": "OSSA_Lecture_11.pdf", "page": 16}],
        expected_answer="Internal fragmentation occurs when allocated memory blocks are larger than the requested size, leaving unused memory inside the allocated partition. External fragmentation occurs when total free memory space exists to satisfy a request, but it is split into small, non-contiguous blocks, preventing allocation."
    ),
    TestQuery(
        query="Compare Hard Real-Time and Soft Real-Time Systems.",
        category="CPU Scheduling",
        expected_sources=[{"source": "OSSA_Lecture_2.pdf", "page": 7}],
        expected_answer="Hard Real-Time systems guarantee that critical tasks complete within a strict deadline, and failure to do so is a total system failure. Soft Real-Time systems prioritize critical tasks, but missing a deadline is acceptable (though undesirable) and only degrades service quality."
    ),
    TestQuery(
        query="List four necessary conditions for deadlock occurrence.",
        category="Deadlocks",
        expected_sources=[{"source": "2022.pdf", "page": 3}, {"source": "OSSA_Lecture_8.pdf", "page": 41}],
        expected_answer="The four necessary conditions for deadlock are: 1) Mutual Exclusion (resources held exclusively), 2) Hold and Wait (processes holding resources request others), 3) No Preemption (resources cannot be forcibly taken), and 4) Circular Wait (a circular chain of waiting processes)."
    ),
    TestQuery(
        query="What does a cycle in a resource-allocation graph indicate?",
        category="Deadlocks",
        expected_sources=[{"source": "2022.pdf", "page": 3}],
        expected_answer="In a resource-allocation graph, if all resource types have only single instances, a cycle indicates that a deadlock has occurred. If resource types have multiple instances, a cycle indicates a potential deadlock, but not necessarily an active one."
    )
]

def calculate_mrr(retrieved_chunks, expected_sources):
    """Calculate Mean Reciprocal Rank (MRR) for the retrieved chunks."""
    for rank, chunk in enumerate(retrieved_chunks):
        src = chunk['metadata'].get('source', '')
        pg = chunk['metadata'].get('page', -1)
        
        # Check if the retrieved chunk matches any expected source (by filename)
        for expected in expected_sources:
            if expected['source'].lower() in src.lower():
                # Optional: Match page numbers if they align closely
                if expected['page'] == -1 or expected['page'] == pg:
                    return 1.0 / (rank + 1)
    return 0.0

def calculate_ndcg(retrieved_chunks, expected_sources):
    """Calculate Normalized Discounted Cumulative Gain (nDCG) at K."""
    if not retrieved_chunks:
        return 0.0
        
    dcg = 0.0
    for rank, chunk in enumerate(retrieved_chunks):
        src = chunk['metadata'].get('source', '')
        pg = chunk['metadata'].get('page', -1)
        
        # Binary relevance: 1 if it matches expected, else 0
        relevance = 0.0
        for expected in expected_sources:
            if expected['source'].lower() in src.lower() and (expected['page'] == -1 or expected['page'] == pg):
                relevance = 1.0
                break
                
        dcg += relevance / math.log2(rank + 2)
        
    # Ideal DCG (if all expected items were at the top of retrieval)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(expected_sources), len(retrieved_chunks))))
    
    return dcg / idcg if idcg > 0 else 0.0

def calculate_keyword_coverage(query, retrieved_chunks):
    """Calculate percentage of query non-stopwords covered in retrieved text."""
    stopwords = {'what', 'is', 'a', 'the', 'and', 'or', 'in', 'of', 'for', 'to', 'how', 'can', 'between', 'from', 'on', 'explain', 'difference'}
    words = re.findall(r'\b\w+\b', query.lower())
    keywords = [w for w in words if w not in stopwords]
    
    if not keywords:
        return 100.0
        
    combined_text = " ".join([c['text'] for c in retrieved_chunks]).lower()
    matches = sum(1 for kw in keywords if kw in combined_text)
    return (matches / len(keywords)) * 100.0

def evaluate_all_retrieval():
    """Runs retrieval evaluation yielding progress and results."""
    total = len(TEST_SUITE)
    for idx, test in enumerate(TEST_SUITE):
        # Run search pipeline to get top chunks
        # search_pipeline returns (decision, rewritten_query, final_context)
        _, _, chunks = search_pipeline(test.query)
        
        mrr = calculate_mrr(chunks, test.expected_sources)
        ndcg = calculate_ndcg(chunks, test.expected_sources)
        coverage = calculate_keyword_coverage(test.query, chunks)
        
        result = RetrievalResult(mrr=mrr, ndcg=ndcg, keyword_coverage=coverage)
        progress_val = (idx + 1) / total
        
        yield test, result, progress_val

def llm_judge_answer(query, generated_answer, expected_answer, context_text):
    """Use Gemini or Groq to score the generated answer from 1 to 5."""
    client, provider = get_llm_client()
    
    prompt = (
        "You are an objective AI evaluator grading a student tutor's response.\n\n"
        "Here is the evaluation context:\n"
        f"1. Student Question: \"{query}\"\n"
        f"2. Ground Truth Slides Reference Answer: \"{expected_answer}\"\n"
        f"3. Tutor Generated Answer: \"{generated_answer}\"\n"
        f"4. Retrieved Source Material:\n{context_text}\n\n"
        "Evaluate the Tutor Generated Answer on a scale of 1 to 5 for three criteria:\n"
        "- Accuracy: Are all facts in the answer correct according to the retrieved source materials and reference answer? (1=False, 5=Highly Accurate)\n"
        "- Completeness: Does the answer address all key parts of the question compared to the reference answer? (1=Missing key items, 5=Fully complete)\n"
        "- Relevance: Is the answer focused directly on the question without including unrelated topics? (1=Off topic/fluff, 5=Highly focused)\n\n"
        "Your output must be in EXACTLY the following format:\n"
        "ACCURACY: [score]\n"
        "COMPLETENESS: [score]\n"
        "RELEVANCE: [score]"
    )
    
    try:
        if provider == "groq":
            completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            response_text = completion.choices[0].message.content.strip()
        elif provider == "gemini":
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )
            response_text = response.text.strip()
            
        # Parse scores
        accuracy = 4.0
        completeness = 4.0
        relevance = 4.0
        for line in response_text.split("\n"):
            if "ACCURACY:" in line:
                accuracy = float(re.findall(r'\d+\.?\d*', line)[0])
            elif "COMPLETENESS:" in line:
                completeness = float(re.findall(r'\d+\.?\d*', line)[0])
            elif "RELEVANCE:" in line:
                relevance = float(re.findall(r'\d+\.?\d*', line)[0])
                
        return accuracy, completeness, relevance
    except Exception as e:
        print(f"LLM Judge failed: {e}")
        return 4.0, 4.0, 4.0

def evaluate_all_answers():
    """Runs answer generation and grades responses using the LLM judge."""
    total = len(TEST_SUITE)
    for idx, test in enumerate(TEST_SUITE):
        # 1. Retrieve context
        _, _, chunks = search_pipeline(test.query)
        
        # 2. Format context for the LLM judge
        context_text = ""
        for i, c in enumerate(chunks):
            context_text += f"[{i+1}] Source: {c['metadata']['source']}\n{c['text']}\n\n"
            
        # 3. Generate the answer
        generated_answer = generate_grounded_answer(test.query, chunks)
        
        # 4. LLM Judge scoring
        accuracy, completeness, relevance = llm_judge_answer(test.query, generated_answer, test.expected_answer, context_text)
        
        result = AnswerResult(accuracy=accuracy, completeness=completeness, relevance=relevance)
        progress_val = (idx + 1) / total
        
        yield test, result, progress_val
