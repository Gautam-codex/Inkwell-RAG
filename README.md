# Autonomous Multimodal RAG Platform for Complex Documents

Upload PDFs (reports, manuals, scanned pages, tables, diagrams) and ask questions in a
Streamlit chat. Every answer cites the document and page it came from.

Pipeline:

PDF upload → layout-aware parsing and chunking → embedding generation → ChromaDB storage →
user query → hybrid retrieval (dense + BM25) → cross-encoder re-ranking → LLM context
synthesis → answer with page-level citations

> Status: Phase 1 (setup). A full README with an architecture diagram arrives in Phase 9.

## Setup (Windows 11, PowerShell)

```powershell
# 1. Virtual environment (Python 3.11)
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

# 2. Ollama (local LLM + vision model)
winget install Ollama.Ollama
ollama pull llama3.2
ollama pull llama3.2-vision

# 3. Tesseract OCR (for scanned pages)
winget install UB-Mannheim.TesseractOCR

# 4. Config
Copy-Item .env.example .env
# then set TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe in .env
```

If `Activate.ps1` is blocked, run once:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

## Setup (Linux)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2 && ollama pull llama3.2-vision
sudo apt install tesseract-ocr
cp .env.example .env
```

## Run

```bash
streamlit run app.py
python -m pytest -q
```
