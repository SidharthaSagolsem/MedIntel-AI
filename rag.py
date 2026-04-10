# rag.py
import chromadb
from sentence_transformers import SentenceTransformer
import uuid

# 1. Initialize In-Memory Vector Database
chroma_client = chromadb.Client()
collection_name = "medintel_doc"

# 2. Load the Embedding Model
try:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    print(f"Error loading embedding model: {e}")
    embedding_model = None

def reset_index():
    """Clears the existing vector database collection for a new document."""
    try:
        chroma_client.delete_collection(name=collection_name)
    except Exception:
        pass 

def chunk_text(text: str, chunk_size: int = 50, overlap: int = 15) -> list:
    """
    Splits text into smaller 50-word chunks. 
    This allows the vector search to find specific sentences rather than large blobs.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

def index_document(raw_text: str):
    """Chunks the text, generates vector embeddings, and stores them in ChromaDB."""
    if not raw_text or not embedding_model:
        return

    collection = chroma_client.get_or_create_collection(name=collection_name)
    
    # Using smaller chunks for better fact-finding
    chunks = chunk_text(raw_text)
    
    for i, chunk in enumerate(chunks):
        embedding = embedding_model.encode(chunk).tolist()
        collection.add(
            embeddings=[embedding],
            documents=[chunk],
            ids=[str(uuid.uuid4())],
            metadatas=[{"chunk_index": i}]
        )

def answer_question(question: str, raw_text: str = "") -> dict:
    """
    Embeds the user's question and searches for the most relevant specific facts.
    """
    if not raw_text:
        return {
            "answer": "No document indexed yet. Please upload a PDF.",
            "confidence": "none",
            "sources": []
        }
        
    if not embedding_model:
         return {
            "answer": "The embedding model failed to load.",
            "confidence": "none",
            "sources": []
        }

    try:
        collection = chroma_client.get_collection(name=collection_name)
    except Exception:
        return {
            "answer": "Document index not found. Please re-upload the PDF.",
            "confidence": "none",
            "sources": []
        }

    # 1. Embed the Question
    question_embedding = embedding_model.encode(question).tolist()
    
    # 2. Query for the top 2 most specific chunks
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=2 
    )
    
    retrieved_chunks = results['documents'][0] if results['documents'] else []
    distances = results['distances'][0] if results['distances'] else [2.0]

    # 3. Precision Filter: If the best match is too far (math distance > 1.6), 
    # it's likely a generic match. Tell the user we can't find it.
    if not retrieved_chunks or distances[0] > 1.65:
        return {
            "answer": "I'm sorry, I couldn't find specific information in the document to answer that question accurately.",
            "confidence": "low",
            "sources": []
        }

    # 4. Determine Confidence
    best_distance = distances[0]
    confidence = "low"
    if best_distance < 1.0:
        confidence = "high"
    elif best_distance < 1.4:
        confidence = "medium"

    # 5. Formulate the Beautifully Spaced Answer
    formatted_chunks = []
    for chunk in retrieved_chunks:
        # Clean up PDF clumping
        clean_chunk = chunk.replace(" - ", "<br><br>• ")
        clean_chunk = clean_chunk.replace('\n', '<br>')
        
        formatted_chunks.append(
            f"<div style='background: rgba(15, 23, 42, 0.4); padding: 12px 14px; "
            f"border-radius: 8px; margin: 10px 0; border-left: 3px solid #38bdf8; "
            f"font-size: 0.88rem; line-height: 1.7;'>"
            f"<i>{clean_chunk}...</i>"
            f"</div>"
        )
        
    context_str = "".join(formatted_chunks)
    
    answer = (
        f"<div style='font-weight: 500; margin-bottom: 4px;'>"
        f"Based on the document context, here is what I found:"
        f"</div>"
        f"{context_str}"
    )

    return {
        "answer": answer,
        "confidence": confidence,
        "sources": retrieved_chunks
    }