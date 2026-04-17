---
description: how to train and test the MoE model with current stability fixes
---
To train and test your Mixture of Experts model with the latest stability and diversity fixes:

1. Build the updated code to ensure all fixes are applied
// turbo
go build -o gollemer ./cmd/tools/train_moe/main.go

2. Run the chat training loop
   - **Phase 0**: MLM pre-training runs first (5 epochs of fill-in-the-blank) to teach grammar
   - **Phase 1**: Then the main seq2seq training begins with curriculum learning
   - The batch size is reduced to 8 to fit 8GB GPU memory
   - Use -batch-size 4 and -acc-steps 16 if you still hit OOM
// turbo
./gollemer -train-chat -gpu

3. (Optional) Run overfitting test if you suspect signal collapse
   - Use this to verify that the model can learn a single pattern perfectly
   - Note: overfit mode SKIPS MLM pre-training to focus on the single example
./gollemer -overfit -gpu

4. Test in interactive mode
   - The new Retrieval Logic will now prioritize exact matches
./gollemer -llm
