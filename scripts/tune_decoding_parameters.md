# Tuning Decoding Parameters for LLM Output

## Parameters to Tune

- **top-k**: Controls diversity. Higher values = more diverse, lower = more focused.
- **temperature**: Controls randomness. Higher = more random, lower = more deterministic.
- **repetition penalty**: Penalizes repeated tokens for less repetition.

## How to Tune

- Edit the values in your decoding function calls (e.g., `GreedySearchDecode`).
- Try different combinations and observe output quality.

## Example

```go
// Example call
predictedIDs, err := model.GreedySearchDecode(contextVector, maxSeqLength, bosID, eosID, 1.2, 100) // topK=100
```

- Lower top-k for more focused output.
- Increase temperature for more randomness.
- Increase repetition penalty to avoid repeated tokens.

## Next Steps

- Experiment with values interactively.
- Use validation set to measure output quality.
