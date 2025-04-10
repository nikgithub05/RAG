import json
import torch
from sentence_transformers import SentenceTransformer, util
import requests
import time
import re
from bs4 import BeautifulSoup

# Let's see if we can use GPU - I read this is faster
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Loading the model - I chose this one because someone on Reddit said it's good
model = SentenceTransformer('all-MiniLM-L6-v2')
model = model.to(device)  # Put on GPU if available


# This function loads embeddings from a file
# I had to look up how to do this
def load_embeddings(file_path='embeddings.json'):
    print(f"Loading embeddings from {file_path}...")
    start_time = time.time()  #shows time taken 
    
    # Open the file and read the data
    with open(file_path, 'r') as f:
        embeddings = json.load(f)
    
    # Convert to PyTorch tensor - not sure if this is the best way
    # Had to add dtype after getting weird errors
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32).to(device)
    
    # Let's see how long it took
    elapsed_time = time.time() - start_time
    print(f"Loaded {len(embeddings)} embeddings into VRAM in {elapsed_time:.2f} seconds")
    return embeddings_tensor

# Loading sentences
def load_sentences(file_path='sentences.json'):
    print(f"Loading sentences from {file_path}...")
    try:  # Adding try-except just to be safe
        with open(file_path, 'r') as f:
            sentences = json.load(f)
        print(f"Loaded {len(sentences)} sentences")
        return sentences
    except Exception as e:
        print(f"Something went wrong: {e}")
        return []  # Return empty list if something breaks

# This function finds similar words
# top_k = no.of similar values to look for
def find_most_similar(query, embeddings_tensor, sentences, top_k=100):
    # Let's time this too
    start_time = time.time()

    query_embedding = model.encode(query, convert_to_tensor=True).to(device)

    query_embedding = query_embedding.reshape(1, -1)
    
    cos_scores = util.pytorch_cos_sim(query_embedding, embeddings_tensor)[0]
    
    # Get the best matches
    top_results = torch.topk(cos_scores, k=min(top_k, len(cos_scores)))
    
    result_sentences = []
    for score, idx in zip(top_results[0], top_results[1]):
        result_sentences.append({
            "sentence": sentences[idx.item()],
            "score": score.item()  # had to use .item() to convert from tensor
        })
    
    # Print total time taken
    elapsed_time = time.time() - start_time
    print(f"Search took {elapsed_time:.4f} seconds")
    
    return result_sentences

# This function finds URLs in text
# ChatGPT helped me 
def extract_urls_from_text(text):
    url_pattern = r'https?://[^\s)]+'
    urls = re.findall(url_pattern, text)
    return urls

# This function downloads web pages
def scrape_url(url):
    try:
        print(f"Trying to download: {url}")
        start_time = time.time()
      
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status() 

        # We use BeautifulSoup to parse HTML
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Get rid of the junk
        for tag in soup(["script", "style", "header", "footer", "nav", "form"]):
            tag.decompose()

        # Extracts the text
        text = soup.get_text(separator=' ', strip=True)
        
        # Clean up the text - Stack Overflow
        text = re.sub(r'\s+', ' ', text).strip()
 
        result = text[:1000] + ("..." if len(text) > 1000 else "")
        
        elapsed_time = time.time() - start_time
        print(f"Got the website in {elapsed_time:.2f} seconds!")
        
        return {
            "url": url,
            "content": result,
            "success": True
        }
    except Exception as e:
        print(f"FAILED to get {url}: {e}")
        return {
            "url": url,
            "content": f"[Couldn't get this URL: {str(e)}]",
            "success": False
        }


def enhance_context_with_url_content(similar_items, max_urls=10):
    # First we need to find all the URLs
    all_urls = []
    for item in similar_items:
        urls = extract_urls_from_text(item["sentence"])
        if urls:
            # Add any URLs we find
            all_urls.extend(urls)
    
    # Remove duplicates
    unique_urls = list(set(all_urls))
    
    urls_to_scrape = unique_urls[:max_urls]
    
    if urls_to_scrape:
        print(f"\nFound {len(unique_urls)} URLs")
    else:
        print("No URLs found")
        return similar_items, []
    
    # Download each URL
    scraped_contents = []
    for url in urls_to_scrape:
        scraped_content = scrape_url(url)
        scraped_contents.append(scraped_content)
    
    return similar_items, scraped_contents

# Sends everything to Ollama for the final answer
def send_to_ollama(context_items, query, scraped_contents=None):
    url = "http://127.0.0.1:11434/api/generate"  # local Ollama server
    headers = {"Content-Type": "application/json"}
    
    # Time this too!
    start_time = time.time()
    
    # Put together all the text we found
    context_text = "\n".join([f"[Relevance: {item['score']:.4f}] {item['sentence']}" 
                             for item in context_items])
    
    # Add the website content we downloaded
    scraped_text = ""
    if scraped_contents and len(scraped_contents) > 0:
        scraped_text = "\n\nAdditional content from URLs found in retrieved sentences:\n" + "\n\n".join(
            [f"[URL: {item['url']}]\n{item['content']}" for item in scraped_contents]
        )
    
    # Combine everything
    full_context = context_text + scraped_text
    
    data = {
        "model": "dolphin-phi", 
        "prompt": f"Context information is below. Use this information to answer the user's question. make it sound human, if possilbe give some context as well. \n\n{full_context}\n\nQuestion: {query}\n\nAnswer:",
        "stream": False  
    }
    
    try:
        # Send to Ollama
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        
        # Print timing info
        elapsed_time = time.time() - start_time
        print(f"Got answer in {elapsed_time:.4f} seconds")
        
        return response.json()
    except requests.exceptions.HTTPError as http_err:
     
        print("HTTP ERROR!!!")
        return {"error": f"HTTP error occurred: {http_err}", "response": response.text}
    except Exception as err:
        print("Error")
        return {"error": f"An error occurred: {err}", "response": ""}


def main():
    print("==============Loading=================")
    
    # Load everything we need
    embeddings_tensor = load_embeddings()
    sentences = load_sentences()
    
    print(f"Using {'GPU VRAM' if device == 'cuda' else ''}")
    
    # Main loop
    while True:
        # Get user question
        query = input("\nAsk me something: ")
            
        # Search for similar stuff
        print("Looking for answers...")
        similar_items = find_most_similar(query, embeddings_tensor, sentences, top_k=20)
        
        # Show what we found
        print("\nRelated Search:")
        for i, item in enumerate(similar_items):
            print(f"{i+1}. [Score: {item['score']:.4f}] {item['sentence']}")
        
        # Get web pages too
        print("\nLooking for web pages now...")
        enhanced_items, scraped_contents = enhance_context_with_url_content(similar_items)
        
        # Show what websites we found
        if scraped_contents:
            print("\nScrapted Websites:")
            for i, content in enumerate(scraped_contents):
                if content["success"]:
                    print(f"{i+1}. {content['url']} - Successful")
                else:
                    print(f"{i+1}. {content['url']} - Failed :(")
        
        # Send everything to Ollama
        print("\nAsking Ollama for the final answer...")
        response = send_to_ollama(enhanced_items, query, scraped_contents)
        
        # Show the answer
        print("\n=== ANSWER ===")
        if "error" in response:
            print(f"ERROR! Something went wrong: {response['error']}")
        else:
            print(response.get("response", "No response from Ollama"))

if __name__ == "__main__":
    main()
