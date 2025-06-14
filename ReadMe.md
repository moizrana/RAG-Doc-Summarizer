# RAG Document Summarization System

A complete implementation of Retrieval-Augmented Generation (RAG) for intelligent document summarization using Together AI and vector embeddings.

## Features

- **Multi-format Support**: Process PDF, TXT, and Markdown files
- **Semantic Chunking**: Intelligent document segmentation with overlap
- **Vector Search**: FAISS-powered similarity search for relevant content retrieval
- **AI-Powered Summarization**: Together AI integration with Llama models
- **Customizable Queries**: Target specific aspects of documents
- **JSON Export**: Save results for further analysis
- **CLI Interface**: Easy-to-use command-line tool

## Requirements

- Python 3.8 or higher
- Together AI API key

## File Structure

### `main.py`
- **Purpose**: Main script to run the complete pipeline.
- **What it does**:
  - Loads the document
  - Chunks the text
  - Builds vector store
  - Retrieves top-k relevant chunks
  - Sends context to Together AI to generate a concise summary
  - Prints and optionally saves the summary result

---

### `core_components.py`
- **Purpose**: Handles all document ingestion and preprocessing tasks.
- **Includes**:
  - `DocumentProcessor`: Loads `.pdf`, `.txt`, and `.md` files
  - `DocumentChunker`: Splits documents into overlapping text chunks
  - `ChunkInfo`: Metadata class for each chunk

---

### `models.py`
- **Purpose**: Embedding + vector search using FAISS.
- **Includes**:
  - `VectorStore`: Generates embeddings and retrieves top chunks
  - `RetrievalResult`: Stores metadata for each retrieved chunk

---

### `rag_pipeline.py`
- **Purpose**: Summarizes content using Together AI's large language model (LLaMA 3).
- **Includes**:
  - `SummaryGenerator`: Prepares prompt and calls the Together AI API to generate the summary

---


### Install Dependencies

```bash
pip install -r requirements.txt
```

### API Key Setup

The system uses Together AI for text generation. You need to:

1. Get your Together AI API key from [Together AI](https://together.ai/)
2. The API key is already embedded in the code for this demo
3. Use environment variables:

```python
self.api_key = os.getenv("TOGETHER_API_KEY", "your-api-key-here")
```

## Quick Start

### Basic Usage

```bash
# Summarize a document with default settings
python main.py document.pdf

# Summarize with a specific focus
python main.py document.pdf --query "What are the main research findings?"
```

### Example Output

```
================================================================================
RAG DOCUMENT SUMMARIZATION RESULTS
================================================================================

Document: document.pdf
Query: Summarize this document
Processing Time: 15.30 seconds
Token Usage: 1247 tokens

========================================
GENERATED SUMMARY
========================================

The provided document excerpts appear to be a collection of research papers and articles related to deep learning and neural networks. Here is a comprehensive summary of the main points and key information: ....

========================================
RETRIEVED CONTEXT
========================================

[Rank 1] Similarity: 0.543
Chunk ID: 23
Text: Augmented Reality has shown significant promise in medical education...
```

## Detailed Usage

### Command Line Arguments

| Argument | Description | Default | Example |
|----------|-------------|---------|---------|
| `document` | Path to document file | Required | `research.pdf` |
| `--query` | Summarization focus | "Summarize this document" | `--query "Key findings"` |
| `--k` | Number of chunks to retrieve | 5 | `--k 10` |
| `--max-length` | Maximum summary length (words) | 200 | `--max-length 300` |
| `--chunk-size` | Document chunk size (chars) | 512 | `--chunk-size 1024` |
| `--output` | Output JSON file path | None | `--output results.json` |
| `--embedding-model` | Sentence transformer model | "all-MiniLM-L6-v2" | Custom model |
| `--generation-model` | Together AI model | "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo" | Custom model |
| `--hide-chunks` | Hide retrieved chunks | False | `--hide-chunks` |

### Supported File Formats

- **PDF Files** (`.pdf`): Extracts text using PyPDF2
- **Text Files** (`.txt`): Direct text processing
- **Markdown Files** (`.md`, `.markdown`): Converts to plain text

## 🔧 Configuration

### Embedding Models

You can use different sentence transformer models:

```bash
# Lightweight model (faster, less accurate)
python main_together.py document.pdf --embedding-model "all-MiniLM-L6-v2"

# More accurate model (slower, better quality)
python main_together.py document.pdf --embedding-model "all-mpnet-base-v2"
```

### Generation Models

Available Together AI models:

- `meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo` (default, fast)
- `meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo` (more capable)
- `mistralai/Mixtral-8x7B-Instruct-v0.1` (alternative option)

```bash
python main_together.py document.pdf \
    --generation-model "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
```

## Output Formats

### Console Output

The system displays:
- Document metadata
- Processing time and token usage
- Generated summary
- Retrieved context chunks (with similarity scores)

### JSON Output

Use `--output filename.json` to save structured results:

```json
{
  "summary": "Generated summary text...",
  "retrieved_chunks": [
    {
      "rank": 0,
      "similarity_score": 0.845,
      "chunk_id": 23,
      "text": "Relevant chunk content..."
    }
  ],
  "token_usage": {
    "prompt_tokens": 892,
    "completion_tokens": 156,
    "total_tokens": 1048
  },
  "latency": 15.3,
  "metadata": {
    "document_path": "document.pdf",
    "total_chunks": 45,
    "document_length": 23040,
    "query": "Summarize this document"
  }
}
```
