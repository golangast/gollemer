import json
from collections import Counter
import sys

# Usage: python analyze_token_frequency.py <path_to_json>
def extract_tokens(obj):
    if isinstance(obj, dict):
        tokens = []
        for k, v in obj.items():
            tokens.append(str(k))
            tokens.extend(extract_tokens(v))
        return tokens
    elif isinstance(obj, list):
        tokens = []
        for item in obj:
            tokens.extend(extract_tokens(item))
        return tokens
    else:
        return [str(obj)]

def main(path):
    with open(path, 'r') as f:
        data = json.load(f)
    counter = Counter()
    for entry in data:
        if 'semantic_output' in entry:
            tokens = extract_tokens(entry['semantic_output'])
            counter.update(tokens)
    print("Token Frequency:")
    for token, count in counter.most_common():
        print(f"{token}: {count}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_token_frequency.py <path_to_json>")
        sys.exit(1)
    main(sys.argv[1])
