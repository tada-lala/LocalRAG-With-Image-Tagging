import os
import tempfile
import streamlit as st
import pandas as pd
from PIL import Image
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
import fitz  # PyMuPDF
import numpy as np
from typing import List, Tuple, Optional
import logging

# --- CLIP for image understanding ---
import clip

# --- Ollama for local LLM ---
import ollama

# --- ChromaDB ---
import chromadb
from chromadb.config import Settings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from streamlit.runtime.uploaded_file_manager import UploadedFile
from langchain_community.document_loaders import PyMuPDFLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------- Streamlit Config --------------------
st.set_page_config(
    page_title="Local Secure RAG with Vision", 
    page_icon="🔍",
    layout="wide"
)

# -------------------- Enhanced Models --------------------
@st.cache_resource
def load_models():
    """Load and cache all models"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Text embedding model
    text_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    
    # CLIP for image understanding
    clip_model, preprocess_clip = clip.load("ViT-B/32", device=device)
    
    # Cross-encoder for re-ranking
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    return text_model, clip_model, preprocess_clip, cross_encoder, device

text_model, clip_model, preprocess_clip, cross_encoder, device = load_models()

# -------------------- Vector Stores --------------------
@st.cache_resource
def initialize_vector_stores():
    """Initialize ChromaDB collections"""
    chroma_client = chromadb.PersistentClient(
        path="./local_secure_rag_chroma",
        settings=Settings(anonymized_telemetry=False)
    )

    text_collection = chroma_client.get_or_create_collection(
        name="document_text_collection",
        metadata={"hnsw:space": "cosine"},
    )

    image_collection = chroma_client.get_or_create_collection(
        name="image_collection",
        metadata={"hnsw:space": "cosine"},
    )
    
    return chroma_client, text_collection, image_collection

chroma_client, text_collection, image_collection = initialize_vector_stores()

# -------------------- Embedding Functions --------------------
def embed_text(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for text"""
    if isinstance(texts, str):
        texts = [texts]
    return text_model.encode(texts, convert_to_numpy=True).tolist()

def embed_image(image_path: str) -> List[float]:
    """Generate CLIP embeddings for images"""
    try:
        image = preprocess_clip(Image.open(image_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = clip_model.encode_image(image)
        return embedding.cpu().numpy()[0].tolist()
    except Exception as e:
        logger.error(f"Error embedding image {image_path}: {e}")
        return [0.0] * 512

def embed_text_for_image_search(text: str) -> List[float]:
    """Generate CLIP text embedding for image search"""
    text_tokens = clip.tokenize([text]).to(device)
    with torch.no_grad():
        embedding = clip_model.encode_text(text_tokens)
    return embedding.cpu().numpy()[0].tolist()

# -------------------- Vision LLM for Image Tagging --------------------
def get_image_description_with_vision(image_path: str) -> str:
    """Use Llama3.2-Vision to generate detailed image descriptions"""
    try:
        response = ollama.chat(
            model='llama3.2-vision',
            messages=[{
                'role': 'user',
                'content': 'Describe this image in detail. Include objects, colors, scene, people, text, and artistic style. Be specific and comprehensive.',
                'images': [image_path]
            }]
        )
        return response['message']['content']
    except Exception as e:
        logger.warning(f"Vision model not available, using fallback: {e}")
        return f"Image: {os.path.basename(image_path)}"

def get_image_tags_with_vision(image_path: str) -> List[str]:
    """Extract searchable tags from image using Llama3.2-Vision"""
    try:
        response = ollama.chat(
            model='llama3.2-vision',
            messages=[{
                'role': 'user',
                'content': 'List 5-10 relevant tags for this image. Include objects, colors, scene type, and key features. Return only comma-separated tags.',
                'images': [image_path]
            }]
        )
        tags_text = response['message']['content']
        tags = [tag.strip() for tag in tags_text.split(',')]
        return tags[:10]
    except Exception as e:
        logger.warning(f"Could not extract tags: {e}")
        return []

# -------------------- PDF Processing --------------------
def process_pdf_enhanced(uploaded_file: UploadedFile) -> Tuple[List[Document], List[Document]]:
    """Enhanced PDF processing with better text chunking"""
    temp_file = tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False)
    temp_file.write(uploaded_file.read())
    temp_file.close()

    try:
        loader = PyMuPDFLoader(temp_file.name)
        docs = loader.load()
        
        # Enhanced text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=150,
            separators=["\n\n", "\n", ".", "!", "?", ";", ":", " ", ""],
            length_function=len,
        )
        
        text_docs = text_splitter.split_documents(docs)
        
        # Add enhanced metadata
        for doc in text_docs:
            doc.metadata.update({
                "source": uploaded_file.name,
                "type": "pdf_text"
            })
        
        # Extract images
        image_paths = extract_images_from_pdf(temp_file.name)
        image_docs = []
        
        for img_path in image_paths:
            try:
                # Use vision model to understand image content
                description = get_image_description_with_vision(img_path)
                tags = get_image_tags_with_vision(img_path)
                
                # Combine description and tags for better searchability
                content = f"{description}. Tags: {', '.join(tags)}"
                
                img_doc = Document(
                    page_content=content,
                    metadata={
                        "image_path": img_path,
                        "source": uploaded_file.name,
                        "type": "pdf_image",
                        "description": description,
                        "tags": ", ".join(tags)
                    }
                )
                image_docs.append(img_doc)
            except Exception as e:
                logger.warning(f"Could not process image {img_path}: {e}")
        
        return text_docs, image_docs
        
    finally:
        os.unlink(temp_file.name)

def extract_images_from_pdf(pdf_path: str) -> List[str]:
    """Extract images from PDF with better error handling"""
    doc = fitz.open(pdf_path)
    image_paths = []
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        image_list = page.get_images(full=True)
        
        for img_index, img in enumerate(image_list):
            try:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                img_path = os.path.join(
                    tempfile.gettempdir(),
                    f"pdf_{page_num}_{img_index}.{image_ext}"
                )
                
                with open(img_path, "wb") as f:
                    f.write(image_bytes)
                
                # Verify image can be opened
                Image.open(img_path).verify()
                image_paths.append(img_path)
                
            except Exception as e:
                logger.warning(f"Could not extract image {img_index} from page {page_num}: {e}")
                continue
    
    doc.close()
    return image_paths

# -------------------- Image Processing --------------------
def process_uploaded_images(uploaded_images) -> List[Document]:
    """Process uploaded images with vision-based understanding"""
    image_docs = []
    
    for img in uploaded_images:
        temp_path = os.path.join(tempfile.gettempdir(), img.name)
        
        try:
            with open(temp_path, "wb") as f:
                f.write(img.read())
            
            # Get image info
            pil_image = Image.open(temp_path)
            width, height = pil_image.size
            
            # Use vision model for understanding
            description = get_image_description_with_vision(temp_path)
            tags = get_image_tags_with_vision(temp_path)
            
            # Combine for searchable content
            content = f"{description}. Tags: {', '.join(tags)}"
            
            doc = Document(
                page_content=content,
                metadata={
                    "image_path": temp_path,
                    "filename": img.name,
                    "width": width,
                    "height": height,
                    "type": "uploaded_image",
                    "description": description,
                    "tags": ", ".join(tags)
                }
            )
            image_docs.append(doc)
            
        except Exception as e:
            logger.error(f"Error processing image {img.name}: {e}")
            st.error(f"Could not process image {img.name}: {e}")
    
    return image_docs

# -------------------- Vector Store Operations --------------------
def add_to_collection(docs: List[Document], collection_name: str, prefix: str = "doc"):
    """Add documents to collections"""
    if not docs:
        return
    
    collection_map = {
        "text": text_collection,
        "image": image_collection
    }
    
    target_collection = collection_map.get(collection_name)
    if not target_collection:
        st.error(f"Unknown collection: {collection_name}")
        return
    
    ids, metadatas, embeddings, documents_text = [], [], [], []
    
    for idx, doc in enumerate(docs):
        doc_id = f"{prefix}_{idx}_{len(ids)}"
        ids.append(doc_id)
        metadatas.append(doc.metadata)
        documents_text.append(doc.page_content)
        
        try:
            if collection_name == "image":
                embedding = embed_image(doc.metadata["image_path"])
            else:
                embedding = embed_text([doc.page_content])[0]
            embeddings.append(embedding)
        except Exception as e:
            logger.error(f"Error creating embedding for doc {idx}: {e}")
            continue
    
    if embeddings:
        target_collection.upsert(
            documents=documents_text,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings
        )
        st.success(f"✅ Added {len(embeddings)} items to {collection_name} collection!")

# -------------------- Query Functions --------------------
def query_text_collection(query: str, n_results: int = 10) -> dict:
    """Query text collection"""
    return text_collection.query(query_texts=[query], n_results=n_results)

def query_images(query: str, n_results: int = 10) -> dict:
    """Query images using CLIP text encoder"""
    query_embedding = embed_text_for_image_search(query)
    
    results = image_collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    
    return results

def enhanced_re_rank(query: str, documents: List[str], top_k: int = 3) -> Tuple[str, List[int]]:
    """Enhanced re-ranking with better error handling"""
    if not documents:
        return "", []
    
    try:
        pairs = [[query, doc] for doc in documents]
        scores = cross_encoder.predict(pairs)
        
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        relevant_text = " ".join([documents[i] for i in top_indices])
        
        return relevant_text, top_indices
    except Exception as e:
        logger.error(f"Error in re-ranking: {e}")
        fallback_text = " ".join(documents[:top_k])
        return fallback_text, list(range(min(top_k, len(documents))))

# -------------------- LLM Integration --------------------
def call_llm_enhanced(context: str, query: str, system_prompt: str = None):
    """Enhanced LLM call with better error handling"""
    if system_prompt is None:
        system_prompt = """You are a helpful AI assistant that answers questions based on the provided context.
        
        Guidelines:
        - Use only the information provided in the context
        - If the context doesn't contain relevant information, say so clearly
        - Be specific and cite relevant details from the context
        - For image-related queries, describe what you can infer from the image descriptions
        - Provide clear, concise, and accurate answers
        """
    
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
        ]
        
        response = ollama.chat(
            model="llama3.2:3b",
            stream=True,
            messages=messages,
        )
        
        for chunk in response:
            if chunk.get("done", False):
                break
            if "message" in chunk and "content" in chunk["message"]:
                yield chunk["message"]["content"]
                
    except Exception as e:
        logger.error(f"Error calling LLM: {e}")
        yield f"Error generating response: {e}"

# -------------------- Streamlit UI --------------------
def main():
    st.title("🔍 Local Secure RAG with Vision Intelligence")
    st.markdown("*Secure document processing with multimodal AI - Text, PDFs, and Images*")
    
    # Sidebar for file uploads
    with st.sidebar:
        st.header("📁 Upload Data")
        
        # PDF upload
        uploaded_file = st.file_uploader(
            "Upload PDF", 
            type=["pdf"], 
            help="Upload PDF documents for text extraction and image analysis"
        )
        
        # Image upload
        uploaded_images = st.file_uploader(
            "Upload Images", 
            type=["png", "jpg", "jpeg"], 
            accept_multiple_files=True,
            help="Upload images for vision-based understanding and retrieval"
        )
        
        # Processing button
        if st.button("⚡ Process & Store", type="primary"):
            with st.spinner("Processing files with vision AI..."):
                process_files(uploaded_file, uploaded_images)
        
        st.divider()
        
        # Collection stats
        st.subheader("📊 Database Stats")
        text_count = text_collection.count()
        image_count = image_collection.count()
        
        st.metric("Text Documents", text_count)
        st.metric("Images (with AI tags)", image_count)
        
        st.divider()
        
        st.info("💡 **Vision AI Powered**: Images are automatically analyzed using Llama3.2-Vision for intelligent tagging and description.")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Ask Questions")
        
        # Query input
        query_type = st.selectbox(
            "Search Mode",
            ["Text Documents", "Images", "Everything"],
            help="Choose what type of content to search"
        )
        
        query = st.text_area(
            "Your Question:",
            placeholder="Ask about your documents or describe images you're looking for...",
            height=100
        )
        
        col_ask, col_clear = st.columns([3, 1])
        with col_ask:
            ask_button = st.button("🔥 Search", type="primary", use_container_width=True)
        with col_clear:
            if st.button("🗑️ Clear", use_container_width=True):
                st.rerun()
        
        # Process query
        if ask_button and query:
            process_query(query, query_type)
    
    with col2:
        st.header("🎯 Features")
        st.markdown("""
        **Vision Intelligence:**
        - 🤖 Auto image tagging
        - 📝 Detailed descriptions
        - 🔍 Semantic image search
        
        **Document Processing:**
        - 📄 PDF text extraction
        - 🖼️ PDF image analysis
        - 💬 Context-aware Q&A
        
        **Privacy First:**
        - 🔒 100% local processing
        - 🏠 No cloud uploads
        - 🔐 Your data stays yours
        """)

def process_files(uploaded_file, uploaded_images):
    """Process uploaded files"""
    if uploaded_file:
        file_name = uploaded_file.name.replace("-", "_").replace(".", "_").replace(" ", "_")
        
        if uploaded_file.type == "application/pdf":
            with st.spinner("Extracting text and analyzing images..."):
                text_docs, image_docs = process_pdf_enhanced(uploaded_file)
                add_to_collection(text_docs, "text", f"pdf_{file_name}")
                
                if image_docs:
                    st.info(f"🤖 Analyzing {len(image_docs)} images with Vision AI...")
                    add_to_collection(image_docs, "image", f"pdf_{file_name}")
    
    if uploaded_images:
        with st.spinner("Analyzing images with Vision AI..."):
            image_docs = process_uploaded_images(uploaded_images)
            add_to_collection(image_docs, "image", "uploaded")

def process_query(query: str, query_type: str):
    """Process user query based on type"""
    with st.spinner("Searching..."):
        try:
            if query_type == "Text Documents":
                results = query_text_collection(query, n_results=10)
                documents = results["documents"][0]
                
                if documents:
                    relevant_text, top_indices = enhanced_re_rank(query, documents, top_k=3)
                    
                    st.subheader("🤖 AI Response")
                    response_container = st.empty()
                    
                    full_response = ""
                    for chunk in call_llm_enhanced(relevant_text, query):
                        full_response += chunk
                        response_container.markdown(full_response)
                    
                    with st.expander("📚 Source Documents"):
                        for i, idx in enumerate(top_indices):
                            st.markdown(f"**Source {i+1}:**")
                            st.text(documents[idx][:500] + "..." if len(documents[idx]) > 500 else documents[idx])
                            if results["metadatas"][0][idx]:
                                st.json(results["metadatas"][0][idx])
                            st.divider()
                else:
                    st.warning("No relevant documents found.")
            
            elif query_type == "Images":
                results = query_images(query, n_results=8)
                
                if results["documents"][0]:
                    st.subheader("🖼️ Found Images")
                    
                    cols = st.columns(2)
                    for i, (doc, metadata) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
                        if "image_path" in metadata:
                            try:
                                img = Image.open(metadata["image_path"])
                                with cols[i % 2]:
                                    st.image(img, use_container_width=True)
                                    st.caption(f"**Description:** {metadata.get('description', 'N/A')[:150]}...")
                                    st.caption(f"**Tags:** {metadata.get('tags', 'N/A')}")
                                    st.caption(f"**Relevance Score:** {1 - results['distances'][0][i]:.2%}")
                                    st.divider()
                            except Exception as e:
                                st.error(f"Could not display image: {e}")
                else:
                    st.warning("No relevant images found.")
            
            elif query_type == "Everything":
                st.subheader("🔍 Text Results")
                text_results = query_text_collection(query, n_results=5)
                if text_results["documents"][0]:
                    relevant_text, _ = enhanced_re_rank(query, text_results["documents"][0], top_k=2)
                    
                    response_container = st.empty()
                    full_response = ""
                    for chunk in call_llm_enhanced(relevant_text, query):
                        full_response += chunk
                        response_container.markdown(full_response)
                
                st.divider()
                st.subheader("🖼️ Image Results")
                image_results = query_images(query, n_results=4)
                if image_results["documents"][0]:
                    cols = st.columns(2)
                    for i, (doc, metadata) in enumerate(zip(image_results["documents"][0][:4], image_results["metadatas"][0][:4])):
                        if "image_path" in metadata:
                            try:
                                img = Image.open(metadata["image_path"])
                                with cols[i % 2]:
                                    st.image(img, use_container_width=True)
                                    st.caption(metadata.get('description', '')[:100] + "...")
                            except:
                                pass
                                
        except Exception as e:
            st.error(f"Error processing query: {e}")
            logger.error(f"Query processing error: {e}")

if __name__ == "__main__":
    main()
