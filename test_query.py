import chromadb
import chromadb.utils.embedding_functions as ef
import os
from dotenv import load_dotenv

load_dotenv()
openai_ef = ef.OpenAIEmbeddingFunction(api_key=os.getenv("OPENAI_API_KEY"), model_name="text-embedding-3-large")
client = chromadb.PersistentClient(path="./vector_db")
collection = client.get_collection("operating_systems", embedding_function=openai_ef)

results = collection.query(query_texts=["Translation Lookaside Buffer TLB"], n_results=3, include=['documents'])
for doc in results['documents'][0]:
    print(f"--- \n{doc}\n")
