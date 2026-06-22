import os
import csv
import json
import time
import urllib.request
import urllib.error

# Configurable: change this to the local model you have pulled in Ollama
# e.g., 'llama3', 'mistral', 'phi3', etc.
OLLAMA_MODEL = "llama3"
OLLAMA_URL = "http://localhost:11434/api/generate"

def generate_variations(q, a, intent):
    prompt = f"""You are an expert conversational data generator for an AI system.
Given the following user query and assistant response pair, generate 5 distinct, high-quality variations of the user query and corresponding variations of the assistant response.
Keep the original intent exactly the same.
For the assistant response, add a brief, thoughtful "reasoning prefix" in brackets where appropriate (e.g., "[Evaluating user status...] I am doing well, thank you!"). This helps the model's MoE router learn structured thinking before generating text.

Original Query: {q}
Original Answer: {a}
Intent: {intent}

Respond ONLY with a JSON array of objects, where each object has "query" and "answer" keys. Do not include markdown formatting, markdown code blocks, or any other text outside the JSON array.
"""
    
    data = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json" # Forces Ollama to output valid JSON
    }
    
    req = urllib.request.Request(
        OLLAMA_URL, 
        data=json.dumps(data).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode('utf-8'))
            text = result.get('response', '').strip()
            
            # Basic cleanup just in case
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
                
            return json.loads(text.strip())
    except urllib.error.URLError as e:
        print(f"Connection Error: Is Ollama running on localhost:11434? ({e.reason})")
        return []
    except Exception as e:
        print(f"Error generating for '{q}': {e}")
        return []

def main():
    print(f"Using local Ollama model: {OLLAMA_MODEL}")
    
    input_file = "data/training/trainingdata/conversing.csv"
    output_file = "data/training/trainingdata/conversing_augmented.csv"
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return
        
    with open(input_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
        
    augmented_rows = []
    
    print(f"Starting data augmentation for {len(rows)} existing pairs...")
    
    count = 0
    for row in rows:
        if len(row) < 3:
            continue
        q, a, intent = row[0], row[1], row[2]
        grammar = row[3] if len(row) > 3 else "OTHER"
        
        # Keep the original pair
        augmented_rows.append([q, a, intent, grammar])
        
        print(f"[{count+1}/{len(rows)}] Generating variations for: {q}")
        variations = generate_variations(q, a, intent)
        
        if variations:
            for var in variations:
                vq = var.get("query", "")
                va = var.get("answer", "")
                if vq and va:
                    augmented_rows.append([vq, va, intent, grammar])
        else:
            print(f"  -> Failed to parse variations or connection refused.")
                
        count += 1
        
        # Periodically save progress
        if count % 10 == 0:
            with open(output_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerows(augmented_rows)
            print(f"  -> Checkpoint saved to {output_file} ({len(augmented_rows)} rows)")
            
    # Final save
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(augmented_rows)
        
    print(f"\nDone! Augmented data saved to {output_file}.")
    print(f"Generated {len(augmented_rows)} total rows from {len(rows)} original pairs.")
    print("To use this new dataset, back up the original conversing.csv and replace it with conversing_augmented.csv, then run 'make train'.")

if __name__ == "__main__":
    main()
