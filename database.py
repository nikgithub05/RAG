import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import os
import torch
import time
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Check if CUDA is available and set device accordingly
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Initialize the model with GPU support
model = SentenceTransformer('gemma3:1b')
model = model.to(device)
print(f"Model loaded on {device}")

def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text("text")
        return text
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return ""

def create_embeddings_and_sentences(text):
    # Split text into sentences and filter out empty ones
    sentences = [s for s in text.split('\n') if s.strip()]
    
    if not sentences:
        return [], []
    
    # Generate embeddings with GPU acceleration
    print(f"Generating embeddings for {len(sentences)} sentences using {device}...")
    start_time = time.time()
    
    # Process in batches to avoid VRAM issues
    batch_size = 64  # Larger batch size for GPU efficiency
    all_embeddings = []
    
    # Use tqdm for a nice progress bar
    for i in tqdm(range(0, len(sentences), batch_size)):
        batch = sentences[i:i+batch_size]
        
        # Keep tensors on GPU during encoding
        with torch.no_grad():  # Disable gradient calculation for memory efficiency
            batch_embeddings = model.encode(batch, convert_to_tensor=True, show_progress_bar=False)
            # Only transfer to CPU at the final step
            batch_embeddings_cpu = batch_embeddings.cpu().numpy().astype(np.float32)
            all_embeddings.extend(batch_embeddings_cpu.tolist())
        
        # Optional: Force GPU memory cleanup
        torch.cuda.empty_cache()
    
    elapsed_time = time.time() - start_time
    print(f"Embeddings generated in {elapsed_time:.2f} seconds ({len(sentences)/elapsed_time:.2f} sentences/sec)")
    
    # Calculate memory usage
    if device == "cuda":
        gpu_memory_used = torch.cuda.max_memory_allocated() / (1024**3)  # GB
        print(f"Peak GPU memory used: {gpu_memory_used:.2f} GB")
        torch.cuda.reset_peak_memory_stats()
    
    return all_embeddings, sentences

def save_embeddings_to_file(embeddings, file_path='embeddings.json'):
    start_time = time.time()
    with open(file_path, 'w') as f:
        json.dump(embeddings, f)
    elapsed_time = time.time() - start_time
    print(f"Saved {len(embeddings)} embeddings to {file_path} in {elapsed_time:.2f} seconds")

def save_sentences_to_file(sentences, file_path='sentences.json'):
    start_time = time.time()
    with open(file_path, 'w') as f:
        json.dump(sentences, f)
    elapsed_time = time.time() - start_time
    print(f"Saved {len(sentences)} sentences to {file_path} in {elapsed_time:.2f} seconds")

def process_pdf(pdf_path):
    print(f"Processing: {pdf_path}")
    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        print(f"No text extracted from {pdf_path}")
        return [], []
        
    embeddings, sentences = create_embeddings_and_sentences(text)
    return embeddings, sentences

def update_database(pdf_folder):
    print(f"Starting VRAM-optimized database update from PDFs in {pdf_folder}")
    total_start_time = time.time()
    
    # Ensure the folder exists
    if not os.path.exists(pdf_folder):
        print(f"Error: Folder {pdf_folder} does not exist")
        return
    
    # Get list of PDF files
    pdf_files = [os.path.join(pdf_folder, filename) for filename in os.listdir(pdf_folder) if filename.endswith('.pdf')]
    
    if not pdf_files:
        print(f"No PDF files found in {pdf_folder}")
        return
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    all_embeddings = []
    all_sentences = []
    
    # Process PDFs one by one to maximize GPU usage for embeddings
    for pdf_file in pdf_files:
        embeddings, sentences = process_pdf(pdf_file)
        all_embeddings.extend(embeddings)
        all_sentences.extend(sentences)
        
        # Force GPU memory cleanup after each file
        if device == "cuda":
            torch.cuda.empty_cache()
    
    # Save results
    if all_embeddings and all_sentences:
        save_embeddings_to_file(all_embeddings)
        save_sentences_to_file(all_sentences)
        
        total_elapsed_time = time.time() - total_start_time
        print(f"Database update completed in {total_elapsed_time:.2f} seconds")
        print(f"Created database with {len(all_sentences)} sentences and {len(all_embeddings)} embeddings")
        
        # Generate some stats
        embedding_dimension = len(all_embeddings[0])
        embedding_size_mb = len(all_embeddings) * embedding_dimension * 4 / (1024*1024)  # 4 bytes per float32
        print(f"Embedding dimension: {embedding_dimension}")
        print(f"Approximate embedding database size: {embedding_size_mb:.2f} MB")
    else:
        print("No data extracted from PDFs. Database not updated.")

if __name__ == "__main__":
    # Set PDF folder path
    pdf_folder = './pdf'
    
    # Print CUDA info
    if device == "cuda":
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    
    update_database(pdf_folder)
