import os
import glob
import pypdf
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder

# Load environment variables (.env)
load_dotenv()

# Configuration
KNOWLEDGE_BASE_DIR = "knowledge_base/operating_systems"
DB_DIR = os.getenv("PERSIST_DIRECTORY", "./vector_db")
COLLECTION_NAME = "operating_systems"
EMBEDDING_MODEL_NAME = "text-embedding-3-large"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Supported document types and their subdirectories
DOC_TYPES = {
    "lectures": "lecture",
    "tutorials": "tutorial",
    "labs": "lab",
    "past_papers": "past_paper"
}

def get_openai_ef():
    """Get the OpenAI Embedding Function for ChromaDB."""
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise ValueError("OPENAI_API_KEY not found in .env!")
    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=openai_key,
        model_name=EMBEDDING_MODEL_NAME
    )

import re

def chunk_text_semantically(text, max_chars=800):
    """
    Split text semantically into paragraphs and sentences.
    Groups sentences together up to max_chars to keep semantic context intact.
    """
    chunks = []
    current_chunk = ""
    
    # First, split by paragraphs
    paragraphs = re.split(r'\n\n+', text)
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
            
        # If the paragraph is already small enough, just add it
        if len(current_chunk) + len(paragraph) <= max_chars:
            current_chunk += paragraph + "\n\n"
        else:
            # If current_chunk is full, save it and start fresh
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = ""
            
            # If the paragraph itself is huge, split it by sentences
            if len(paragraph) > max_chars:
                sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) <= max_chars:
                        current_chunk += sentence + " "
                    else:
                        if current_chunk.strip():
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence + " "
            else:
                current_chunk = paragraph + "\n\n"
                
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
        
    return chunks

def extract_text_from_pdf(file_path):
    """Extract text page-by-page from a PDF file."""
    pages_content = []
    try:
        reader = pypdf.PdfReader(file_path)
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                pages_content.append({
                    "text": text.strip(),
                    "page_number": page_num + 1
                })
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
    return pages_content

def extract_text_from_markdown(file_path):
    """Extract text from markdown or text files."""
    pages_content = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            if content.strip():
                # Since markdown doesn't have native pages, we treat the whole document as page 1
                pages_content.append({
                    "text": content.strip(),
                    "page_number": 1
                })
    except Exception as e:
        print(f"Error reading markdown file {file_path}: {e}")
    return pages_content

def process_file(file_path, doc_type):
    """Process a file, chunk its content, and return chunks with metadata."""
    file_name = os.path.basename(file_path)
    ext = os.path.splitext(file_name)[1].lower()
    
    if ext == ".pdf":
        pages = extract_text_from_pdf(file_path)
    elif ext in [".md", ".txt"]:
        pages = extract_text_from_markdown(file_path)
    else:
        print(f"Unsupported file format: {file_name}")
        return []
        
    file_chunks = []
    for page in pages:
        # Chunk text semantically on a page/slide basis
        raw_chunks = chunk_text_semantically(page["text"])
        for chunk_idx, chunk in enumerate(raw_chunks):
            metadata = {
                "source": file_name,
                "type": doc_type,
                "page": page["page_number"],
                "chunk_index": chunk_idx
            }
            file_chunks.append({
                "text": chunk,
                "metadata": metadata
            })
            
    print(f"Processed {file_name}: extracted {len(file_chunks)} chunks.")
    return file_chunks

def ingest_documents():
    """Main ingestion pipeline."""
    # 1. Initialize ChromaDB client
    chroma_client = chromadb.PersistentClient(path=DB_DIR)
    
    openai_ef = get_openai_ef()
    
    # Get or create the vector collection
    # Note: If the collection exists, we reuse it.
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=openai_ef
    )
    
    # 2. Pre-download the reranker so it is
    #    cached on disk before the UI is ever launched.
    print(f"Pre-downloading/caching reranker model '{RERANK_MODEL_NAME}'...")
    CrossEncoder(RERANK_MODEL_NAME)
    print("Reranker model cached successfully.")
    
    all_chunks = []
    
    # 3. Scan the directories and process files
    print(f"Scanning knowledge base directory: {KNOWLEDGE_BASE_DIR}")
    for folder_name, doc_type in DOC_TYPES.items():
        search_path = os.path.join(KNOWLEDGE_BASE_DIR, folder_name, "*")
        files = glob.glob(search_path)
        
        for file_path in files:
            if os.path.isfile(file_path):
                chunks = process_file(file_path, doc_type)
                all_chunks.extend(chunks)
                
    if not all_chunks:
        print("No documents found to ingest. Make sure you put some files in the knowledge base folders!")
        return

    print(f"Total chunks generated: {len(all_chunks)}")
    
    # 4. Generate embeddings and save to vector database
    documents = [c["text"] for c in all_chunks]
    metadatas = [c["metadata"] for c in all_chunks]
    
    # Create unique IDs for each chunk based on source, page, and chunk index
    ids = [
        f"{c['metadata']['source']}_p{c['metadata']['page']}_c{c['metadata']['chunk_index']}"
        for c in all_chunks
    ]
    
    print(f"Generating embeddings and writing to ChromaDB using {EMBEDDING_MODEL_NAME}...")
    # Add to ChromaDB in batches to prevent payload limits
    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        end_idx = min(i + batch_size, len(all_chunks))
        collection.add(
            ids=ids[i:end_idx],
            documents=documents[i:end_idx],
            metadatas=metadatas[i:end_idx]
        )
        
    print("Ingestion complete! All files successfully indexed in the vector database.")

if __name__ == "__main__":
    ingest_documents()
