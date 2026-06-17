import os
import glob
import pypdf
import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder

# Load environment variables (.env)
load_dotenv()

# Configuration
KNOWLEDGE_BASE_DIR = "knowledge_base/operating_systems"
DB_DIR = os.getenv("PERSIST_DIRECTORY", "./vector_db")
COLLECTION_NAME = "operating_systems"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Supported document types and their subdirectories
DOC_TYPES = {
    "lectures": "lecture",
    "tutorials": "tutorial",
    "labs": "lab",
    "past_papers": "past_paper"
}

def load_embedding_model():
    """Load the local sentence-transformer model."""
    print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}'...")
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

def chunk_text_by_words(text, chunk_size=150, chunk_overlap=30):
    """
    Split text into chunks based on word count.
    This helps keep semantic contexts (like sentences) intact
    without splitting words in half.
    """
    words = text.split()
    chunks = []
    
    if not words:
        return chunks
        
    step = chunk_size - chunk_overlap
    for i in range(0, len(words), step):
        chunk_words = words[i:i + chunk_size]
        chunks.append(" ".join(chunk_words))
        # Stop if we reached the end of the text
        if i + chunk_size >= len(words):
            break
            
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
        # Chunk text on a page/slide basis
        raw_chunks = chunk_text_by_words(page["text"])
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
    
    # Get or create the vector collection
    # Note: If the collection exists, we reuse it.
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
    
    # 2. Load the embedding model and pre-download the reranker so it is
    #    cached on disk before the UI is ever launched.
    embedding_model = load_embedding_model()
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
    
    print("Generating embeddings...")
    embeddings = embedding_model.encode(documents, show_progress_bar=True).tolist()
    
    print("Writing embeddings and metadata to ChromaDB...")
    # Add to ChromaDB in batches to prevent payload limits
    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        end_idx = min(i + batch_size, len(all_chunks))
        collection.add(
            ids=ids[i:end_idx],
            embeddings=embeddings[i:end_idx],
            documents=documents[i:end_idx],
            metadatas=metadatas[i:end_idx]
        )
        
    print("Ingestion complete! All files successfully indexed in the vector database.")

if __name__ == "__main__":
    ingest_documents()
