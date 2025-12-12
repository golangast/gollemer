#!/usr/bin/env python3
"""
Convert semantic_output_data.json from nested JSON to flat key-value format.
Adds a 'flat_output' field to each training example.
"""

import json
import sys
from pathlib import Path


def flatten_semantic_output(semantic_output):
    """
    Convert nested semantic output to flat key:value format.
    
    Example:
    {
      "operation": "Create",
      "target_resource": {
        "type": "Webserver",
        "name": "jill",
        "properties": {"port": "8080", "runtime": "go"}
      },
      "context": {"user_role": "admin"}
    }
    
    Becomes:
    "operation:Create type:Webserver name:jill port:8080 runtime:go user_role:admin"
    """
    parts = []
    
    # Add operation
    if "operation" in semantic_output:
        parts.append(f"operation:{semantic_output['operation']}")
    
    # Add target_resource fields
    if "target_resource" in semantic_output:
        resource = semantic_output["target_resource"]
        
        if "type" in resource:
            # Simplify type (remove :: separators)
            resource_type = resource["type"].replace("::", "_")
            parts.append(f"type:{resource_type}")
        
        if "name" in resource:
            parts.append(f"name:{resource['name']}")
        
        # Add properties
        if "properties" in resource:
            for key, value in resource["properties"].items():
                parts.append(f"{key}:{value}")
    
    # Add context fields
    if "context" in semantic_output:
        for key, value in semantic_output["context"].items():
            parts.append(f"{key}:{value}")
    
    return " ".join(parts)


def convert_file(input_path, output_path):
    """Convert training data file to include flat_output field."""
    print(f"Reading {input_path}...")
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    print(f"Converting {len(data)} examples...")
    converted = 0
    for example in data:
        if "semantic_output" in example:
            try:
                example["flat_output"] = flatten_semantic_output(example["semantic_output"])
                converted += 1
            except Exception as e:
                print(f"Warning: Failed to convert example: {example.get('query', 'unknown')}")
                print(f"  Error: {e}")
    
    print(f"Successfully converted {converted}/{len(data)} examples")
    
    print(f"Writing to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print("Done!")


def main():
    # Paths
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    input_file = project_dir / "trainingdata" / "semantic_output_data.json"
    output_file = project_dir / "trainingdata" / "semantic_output_data_flat.json"
    
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)
    
    convert_file(input_file, output_file)
    
    # Show example
    print("\n--- Example conversion ---")
    with open(output_file, 'r') as f:
        data = json.load(f)
    
    if data:
        example = data[0]
        print(f"Query: {example['query']}")
        print(f"Flat output: {example.get('flat_output', 'N/A')}")


if __name__ == "__main__":
    main()
