# GPU-Accelerated RAG System

A Retrieval-Augmented Generation (RAG) system that leverages GPU acceleration for efficient document processing, embedding generation, and semantic search.

## Overview

This system allows you to:
1. Process PDF documents and build a vector database
2. Search for relevant information using semantic similarity
3. Generate responses using the Ollama API with local LLMs

The system is optimized to run on NVIDIA GPUs for maximum performance.

## Directory Structure

```
rag-system/
├── pdf/                 # Place your PDF files in this directory
├── database.py          # Script to build the vector database
├── query.py             # Script to query the database and generate responses
├── embeddings.json      # Generated vector database (will be created)
├── sentences.json       # Generated sentence database (will be created)
└── README.md            # This file
```

## Installation

### Prerequisites

- Python 3.8+
- NVIDIA GPU with CUDA support (recommended)
- CUDA Toolkit 11.x+

### Dependencies

Install the required packages:

```pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install sentence-transformers requests PyMuPDF numpy tqdm
```

### Setting up Ollama

1. Download and install Ollama from [ollama.ai](https://ollama.ai/)
2. Pull the Gemma model:
   ```bash
   ollama pull gemma3:1b
   ```
3. Start the Ollama server

## Usage

### Step 1: Build the Vector Database

1. Place your PDF files in the `pdf/` directory
2. Run the database builder:
   ```python
   python database.py
   ```
   This will:
   - Extract text from all PDFs
   - Generate embeddings using GPU acceleration
   - Save the embeddings and sentences to JSON files

### Step 2: Query the Database

Run the query script:
```python
python query.py
```

Enter your queries at the prompt. The system will:
1. Find the most relevant sentences in the database
2. Send them to Ollama as context
3. Generate a response based on the context and your query

Type `exit` to quit the program.

## Performance Optimization

The system is optimized for GPU acceleration:

- Embeddings are generated and processed directly in GPU VRAM
- Similarity search is performed on the GPU
- Batch processing is used to maximize GPU utilization
- Memory management is optimized for large documents

## Files Description

- `database.py`: GPU-accelerated system to process PDFs and build the vector database
- `query.py`: Retrieves relevant information and generates responses via Ollama
- `embeddings.json`: Contains vector embeddings of sentences from your documents
- `sentences.json`: Contains the actual text corresponding to the embeddings

## Troubleshooting

1. **CUDA/GPU Issues**:
   - Verify CUDA installation: `nvidia-smi`
   - Check PyTorch CUDA support: `python -c "import torch; print(torch.cuda.is_available())"`

2. **Ollama Connection Issues**:
   - Check if Ollama server is running
   - Verify Ollama API port (default: 11434)
   - Ensure the model is correctly installed: `ollama list`

3. **Memory Issues**:
   - For large PDFs, reduce batch size in `database.py`
   - For large databases, consider using a chunking approach
