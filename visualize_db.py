import os
import chromadb
import numpy as np
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from dotenv import load_dotenv

# Load config
load_dotenv()
DB_DIR = os.getenv("PERSIST_DIRECTORY", "./vector_db")
COLLECTION_NAME = "operating_systems"

def main():
    print("Connecting to ChromaDB...")
    chroma_client = chromadb.PersistentClient(path=DB_DIR)
    
    try:
        collection = chroma_client.get_collection(name=COLLECTION_NAME)
    except Exception as e:
        print(f"Error: Could not find collection '{COLLECTION_NAME}'. Make sure to run ingest.py first!")
        return
        
    print("Retrieving vectors...")
    # Retrieve all items along with their embeddings, documents, and metadatas
    result = collection.get(include=['embeddings', 'documents', 'metadatas'])
    
    embeddings = result.get('embeddings')
    documents = result.get('documents')
    metadatas = result.get('metadatas')
    
    if embeddings is None or len(embeddings) == 0:
        print("No embeddings found in the database. Run ingest.py first.")
        return
        
    print(f"Found {len(embeddings)} chunks. Processing visualization...")
    
    vectors = np.array(embeddings)
    doc_types = [meta.get('type', 'unknown') for meta in metadatas]
    sources = [meta.get('source', 'unknown') for meta in metadatas]
    pages = [meta.get('page', 0) for meta in metadatas]
    
    # Map each doc type to a specific color
    # Supported: 'lecture', 'tutorial', 'lab', 'past_paper'
    color_map = {
        'lecture': 'royalblue',
        'tutorial': 'forestgreen',
        'lab': 'crimson',
        'past_paper': 'darkorange',
        'unknown': 'gray'
    }
    colors = [color_map.get(t, 'gray') for t in doc_types]
    
    print("Running t-SNE dimensionality reduction (reducing 384 dimensions to 2)...")
    # Fit t-SNE (adjust perplexity if the dataset is small)
    perplexity = min(30, max(5, len(vectors) - 1))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    reduced_vectors = tsne.fit_transform(vectors)
    
    print("Creating Plotly 2D interactive scatter plot...")
    # Create hover text
    hover_texts = []
    for t, src, pg, doc in zip(doc_types, sources, pages, documents):
        # Snippet of the text chunk (first 150 chars)
        snippet = doc.replace('\n', '<br>')[:150]
        hover_text = (
            f"<b>Type:</b> {t.upper()}<br>"
            f"<b>Source:</b> {src} (Page {pg})<br>"
            f"<b>Content:</b> {snippet}..."
        )
        hover_texts.append(hover_text)
        
    fig = go.Figure()
    
    # Plot each class separately so they appear nicely in the interactive legend
    for category, color in color_map.items():
        indices = [i for i, t in enumerate(doc_types) if t == category]
        if not indices:
            continue
            
        fig.add_trace(go.Scatter(
            x=reduced_vectors[indices, 0],
            y=reduced_vectors[indices, 1],
            mode='markers',
            name=category.capitalize(),
            marker=dict(
                size=8,
                color=color,
                opacity=0.85,
                line=dict(width=1, color='white')
            ),
            text=[hover_texts[i] for i in indices],
            hoverinfo='text'
        ))
        
    fig.update_layout(
        title='<b>2D Chroma Vector Store Clusters (Operating Systems)</b><br><sup>Visualizing the semantic similarity of slide chunks, labs, and tutorials</sup>',
        xaxis=dict(title='', showgrid=True, zeroline=False),
        yaxis=dict(title='', showgrid=True, zeroline=False),
        width=1000,
        height=700,
        template='plotly_white',
        legend_title="Material Type"
    )
    
    # Save chart as HTML file and open it automatically in the default browser
    output_html = "vector_store_visualization.html"
    fig.write_html(output_html)
    print(f"\nSuccess! Visualization saved as: '{output_html}'")
    print("Opening your browser to display the interactive chart...")
    fig.show()

if __name__ == "__main__":
    main()
