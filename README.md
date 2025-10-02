# Local Secure RAG with Vision Intelligence

This project is a **local multimodal Retrieval-Augmented Generation (RAG)** system with:
- PDF processing (text + images)
- Image understanding via CLIP and LLaMA Vision
- Local embeddings with ChromaDB
- Question answering via Ollama LLaMA models

1. **Clone this repo**
     git clone https://github.com/yourusername/local-secure-rag-vision.git
     cd local-secure-rag-vision
     python3.11 -m venv .venv
     source .venv/bin/activate   # Linux / Mac
    .venv\Scripts\activate      # Windows
   
2. **Install dependencies** 
    pip install --upgrade pip
    pip install -r requirements.txt

3. **Install Ollama (for LLaMA models)**
    Download Ollama from: https://ollama.com/download
    After installation, verify: ollama --version
  
4.**Downloading LLaMA Models: Ollama manages models locally. Pull the required ones:**
      # LLaMA 3.2 (for text Q&A)
      ollama pull llama3.2:3b
      # LLaMA 3.2 Vision (for image descriptions + tags)
      ollama pull llama3.2-vision

5. **Start Streamlit: streamlit run app.py**
Then open http://localhost:8501 in your browser.
