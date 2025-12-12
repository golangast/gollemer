import json
import sys
from collections import Counter

def main(vocab_path, data_path):
    with open(vocab_path) as f:
        vocab = json.load(f)
    with open(data_path) as f:
        data = json.load(f)
    vocab_set = set(vocab.keys())
    missing = Counter()
    for entry in data:
        if 'semantic_output' in entry:
            def recurse(obj):
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        if k not in vocab_set:
                            missing[k] += 1
                        recurse(v)
                elif isinstance(obj, list):
                    for item in obj:
                        recurse(item)
                else:
                    if str(obj) not in vocab_set:
                        missing[str(obj)] += 1
            recurse(entry['semantic_output'])
    print("Missing tokens:")
    for token, count in missing.most_common():
        print(f"{token}: {count}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python check_vocab_coverage.py <vocab.json> <data.json>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
