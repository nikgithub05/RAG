import json
import torch
from sentence_transformers import SentenceTransformer, util
import requests
import time

# Check if CUDA is available and set device accordingly
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Initialize the model with GPU support
model = SentenceTransformer('all-MiniLM-L6-v2')
model = model.to(device)

def load_embeddings(file_path='embeddings.json'):
    print(f"Loading embeddings from {file_path}...")
    start_time = time.time()
    with open(file_path, 'r') as f:
        embeddings = json.load(f)
    
    # Convert to PyTorch tensor and move directly to GPU
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32).to(device)
    
    elapsed_time = time.time() - start_time
    print(f"Loaded {len(embeddings)} embeddings into VRAM in {elapsed_time:.2f} seconds")
    return embeddings_tensor

def load_sentences(file_path='sentences.json'):
    print(f"Loading sentences from {file_path}...")
    with open(file_path, 'r') as f:
        sentences = json.load(f)
    print(f"Loaded {len(sentences)} sentences")
    return sentences

def find_most_similar(query, embeddings_tensor, sentences, top_k=5):
    # Start timer for performance measurement
    start_time = time.time()
    
    # Encode query and move to same device
    query_embedding = model.encode(query, convert_to_tensor=True).to(device)
    
    # Reshape for cosine similarity calculation
    query_embedding = query_embedding.reshape(1, -1)
    
    # Compute cosine similarities
    cos_scores = util.pytorch_cos_sim(query_embedding, embeddings_tensor)[0]
    
    # Get top-k results
    top_results = torch.topk(cos_scores, k=min(top_k, len(cos_scores)))
    
    # Retrieve the matching sentences and scores
    result_sentences = []
    for score, idx in zip(top_results[0], top_results[1]):
        result_sentences.append({
            "sentence": sentences[idx.item()],
            "score": score.item()
        })
    
    # End timer and print performance info
    elapsed_time = time.time() - start_time
    print(f"Similarity search completed in {elapsed_time:.4f} seconds")
    
    return result_sentences

def send_to_ollama(context_items, query):
    url = "http://127.0.0.1:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    
    # Start timer for performance measurement
    start_time = time.time()
    
    # Prepare context with relevance scores
    context_text = "\n".join([f"[Relevance: {item['score']:.4f}] {item['sentence']}" 
                             for item in context_items])
    
    data = {
        "model": "dolphin-phi",
        "prompt": f"Context information is below. Use this information to answer the user's question.\n\n{context_text}\n\nQuestion: {query}\n\nAnswer:",
        "stream": False
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        
        # End timer and print performance info
        elapsed_time = time.time() - start_time
        print(f"Ollama response generated in {elapsed_time:.4f} seconds")
        
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        return {"error": f"HTTP error occurred: {http_err}", "response": response.text}
    except Exception as err:
        return {"error": f"An error occurred: {err}", "response": ""}

def main():
    print("VRAM-Optimized RAG System")
    print("=========================")
    
    # Load embeddings directly into GPU memory
    embeddings_tensor = load_embeddings()
    sentences = load_sentences()
    
    print(f"Memory usage: Embeddings stored in {'GPU VRAM' if device == 'cuda' else 'system RAM'}")
    
    while True:
        # User query
        query = input("\nEnter your query (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break
            
        # Find most similar sentences
        print("Finding most similar sentences...")
        similar_items = find_most_similar(query, embeddings_tensor, sentences, top_k=5)
        
        # Display retrieved context
        print("\nRetrieved context:")
        for i, item in enumerate(similar_items):
            print(f"{i+1}. [Score: {item['score']:.4f}] {item['sentence']}")
        
        # Send the similar sentences and query to Ollama
        print("\nGenerating response from Ollama...")
        response = send_to_ollama(similar_items, query)
        
        print("\nOllama Response:")
        if "error" in response:
            print(f"Error: {response['error']}")
        else:
            # The Ollama API typically returns the response in the 'response' field
            print(response.get("response", ""))

if __name__ == "__main__":
    main()
