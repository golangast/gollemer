---
description: how to train and test the MoE model with current stability fixes
---
To train and test your Mixture of Experts model with the latest stability and diversity fixes:

1. Build the updated code to ensure all fixes are applied
// turbo
go build -o gollemer .

2. Run the chat training loop
   - This will use the new Gradient Accumulation (Batch Size 128)
   - It will apply Diversity Loss to differentiate experts
   - It will use Expert Multipliers to jump-start starved experts
// turbo
./gollemer -train-chat

3. (Optional) Run overfitting test if you suspect signal collapse
   - Use this to verify that the model can learn a single pattern perfectly
./gollemer -overfit

4. Test in interactive mode
   - The new Retrieval Logic will now prioritize exact matches
./gollemer -llm
