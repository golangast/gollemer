import json
import sys

def main(vocab_path, missing_tokens_path):
    with open(vocab_path) as f:
        vocab = json.load(f)
    with open(missing_tokens_path) as f:
        missing = [line.split(':')[0].strip() for line in f if ':' in line]
    max_id = max(vocab.values()) if vocab else 0
    for token in missing:
        if token not in vocab:
            max_id += 1
            vocab[token] = max_id
    with open(vocab_path, 'w') as f:
        json.dump(vocab, f, indent=2)
    print(f"Added {len(missing)} missing tokens to {vocab_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python add_missing_tokens_to_vocab.py <vocab.json> <missing_tokens.txt>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
