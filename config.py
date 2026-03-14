import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
ROOT_DIR   = Path(__file__).parent
INPUT_DIR  = ROOT_DIR / "data" / "sample_pdfs"
OUTPUT_DIR = ROOT_DIR / "output"
IMAGE_DIR  = ROOT_DIR / "extracted_images"
VECTOR_DIR = ROOT_DIR / "vector_store"

for d in [INPUT_DIR, OUTPUT_DIR, IMAGE_DIR, VECTOR_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# HuggingFace — BLIP image captioning only (optional)
HF_TOKEN   = os.getenv("HF_TOKEN", "")
BLIP_MODEL = "Salesforce/blip-image-captioning-base"

# Ollama — fully local, no API key required
OLLAMA_URL     = "http://localhost:11434/api/embeddings"
OLLAMA_LLM_URL = "http://localhost:11434/api/generate"
EMBED_MODEL    = "nomic-embed-text"   # ollama pull nomic-embed-text
LLM_MODEL      = "gemma3:latest"      # ollama pull gemma3

# LLM generation
LLM_MAX_TOKENS  = 700
LLM_TEMPERATURE = 0.1

# Chunking
CHUNK_SIZE    = 1200
CHUNK_OVERLAP = 150
MIN_CHUNK_LEN = 15   # skip truly empty fragments only

# Retrieval
TOP_K_DEFAULT    = 6
FAISS_INDEX_PATH = str(VECTOR_DIR / "index.faiss")
METADATA_PATH    = str(VECTOR_DIR / "metadata.json")

# Image extraction
MIN_IMAGE_WIDTH  = 80
MIN_IMAGE_HEIGHT = 80

# API
API_HOST = "0.0.0.0"
API_PORT = 8000
