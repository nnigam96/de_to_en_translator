# DocAssist

An efficient local document retrieval and question-answering system powered by LLaMA 2. Built with a modular RAG pipeline using hybrid search and `llama-cpp-python` for fast, quantized inference.

## Features

- **Local LLM Inference**: Uses `llama-cpp-python` for running quantized LLaMA 2 GGUF models locally
- **Hybrid Retrieval System**: Combines FAISS-based semantic search with keyword-based filtering
- **Real-Time Indexing**: Watches document directory and reindexes on file updates
- **Streamlit Interface**: Minimal web UI for querying and document previews
- **Metadata-aware Downloads**: Retrieve document source from every answer
- **Session Management**: Reset context every 30 minutes to avoid memory bloat

## Project Structure

```
doc_assist/
├── models/                  # Place GGUF models here
├── docs/                    # Directory to drop documents (PDF, TXT, DOCX)
├── src/
│   ├── app.py               # Streamlit frontend
│   ├── document_parser.py   # Chunking and preprocessing logic
│   ├── embedding.py         # Embedding generation and storage
│   ├── hybrid_retriever.py  # FAISS + keyword retrieval logic
│   ├── incremental_indexer.py # Live indexing and file watcher
│   ├── llama_cpp_interface.py # Inference using llama-cpp-python
│   └── utils.py             # Helper utilities
```

## Quick Start

### Prerequisites

```bash
brew install cmake
pip install -r requirements.txt
```

### Setup

1. Download a GGUF quantized LLaMA model (e.g., from TheBloke on Hugging Face)  
   Place the `.gguf` file inside the `models/` directory.

2. Drop `.txt`, `.pdf`, or `.docx` files into `docs/`.

3. Launch the app:

```bash
streamlit run src/app.py
```

## Usage

- Ask a question in the chat interface (e.g., “What does the NDA say about IP ownership?”)
- DocAssist retrieves relevant chunks and answers using the local model
- Click “Download Document” to fetch the original source

## Development

### Indexing and Embeddings

Run once or run continuously in background:

```bash
python src/incremental_indexer.py
```

### Eval Framework (WIP)

Evaluation support for exact match and similarity scoring using test questions.

### Streaming Support (Optional)

Adaptable to use streaming with `llama-cpp-python`'s token-level callback.

## Technical Details

- **Embedding Model**: `all-MiniLM-L6-v2` via `sentence-transformers`
- **VectorDB**: `faiss-cpu` for fast ANN retrieval
- **LLM Runtime**: `llama-cpp-python` (Metal or CPU backend)
- **Prompt Format**: RAG-style `[INST]` instructions for LLaMA 2 compatibility
- **File Watching**: `watchdog` for tracking document changes

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Got ideas or suggestions? Feel free to open an issue or reach out.
