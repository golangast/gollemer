# Gollemer MoE Advanced Observability Guide

## Overview

Gollemer now includes four powerful real-time observability features designed to understand how your Mixture of Experts model learns and specializes:

1. **Expert Lexicon (Top In-Domain Tokens)** - See what each expert learns
2. **Live Loss Delta per Token Category** - Track loss improvements by semantic groups
3. **Weight Velocity Heatmap** - Identify learning hotspots in real-time
4. **Semantic Drift Tracker** - Monitor embedding space evolution

## Architecture

### Core Components

#### 1. MoE Observability Module (`moe_observability.go`)
The central observability engine combining all four metrics:

```go
obs := NewMoEObservability(
    numExperts,        // Number of experts (e.g., 8)
    windowSize,        // Metrics collection window (e.g., 500 steps)
    vocab,             // Token vocabulary
    w2vModel,          // Word2Vec embeddings for semantic drift
)
```

#### 2. Trainer Integration (`trainer.go`)
Extended Trainer with observability methods:

```go
trainer.InitializeObservability(8, 500, vocab, w2vModel)
trainer.RecordTrainingStep(expertIDs, tokenIDs, loss)
trainer.UpdateWeightVelocity("layer_0", weights, embedding)
trainer.FinishEpoch(vocab)
```

#### 3. Metrics Aggregator (`metrics_aggregator.go`)
HTTP server exposing metrics for the dashboard:

```go
aggregator := StartMetricsServer(trainer, vocab, ":8080")
```

### Endpoints

- `/api/metrics/current` - Latest metric snapshot
- `/api/metrics/history` - All collected metrics over time
- `/api/metrics/timeseries` - Time-series data for charts
- `/api/metrics/snapshot` - Fresh snapshot with all 4 features

## Feature Details

### 1. Expert Lexicon: Top In-Domain Tokens

**What it shows:** The top 10 tokens that route to each expert, revealing specialization patterns.

**Implementation:**
```go
histogram := NewTokenRoutingHistogram(numExperts, windowSize)
histogram.RecordTokenRoute(expertID, tokenID)

topTokens := histogram.GetTopKTokensPerExpert(10, vocab)
// Output: map[expertID][]tokenString
```

**Interpretation:**
```
Expert 0: ["the", "to", "and", "a", "is"] 
  → Grammar specialist anchoring structural patterns

Expert 5: ["file", "directory", "make", "command"]
  → Computer domain specialist learning technical concepts
```

### 2. Live Loss Delta per Token Category

**What it shows:** How loss improves separately for different semantic token groups (Structural Words, Core Verbs, Technical Terms, Punctuation).

**Configuration:** Edit `data/config/observability_config.json` to customize categories:

```json
{
  "token_categories": [
    {
      "name": "Technical Terms",
      "keywords": ["file", "directory", "command", "script", ...]
    }
  ]
}
```

**Implementation:**
```go
tracker := NewTokenCategoryLossTracker(categories)
tracker.RecordLossForTokens(loss, tokenIDs)  // Each step

// At epoch end:
tracker.UpdateImprovement()
metrics := tracker.GetCategoryLossMetrics()
// Output: map[categoryName]{"average_loss": 0.24, "improvement": 0.15}
```

**Interpretation:**
```
📊 Technical Terms: Loss=0.24, Improvement=0.15 📉
  → Model successfully learning computer domain concepts

📈 Structural Words: Loss=0.19, Improvement=-0.02 📈
  → Struggling with grammar; may need curriculum adjustment
```

### 3. Weight Velocity Heatmap (Frobenius Norm)

**What it shows:** Real-time intensity map of which network layers are actively learning.

**Formula:**
$$\text{Velocity} = \sqrt{\sum_{i} \sum_{j} (W_{\text{new}} - W_{\text{old}})^2}$$

**Implementation:**
```go
velocity := NewWeightVelocityTracker()

// Each epoch:
velocity.RecordWeightSnapshot("layer_0", weights)
velocity.UpdateWeightVelocity("layer_0", newWeights)

heatmap := velocity.GetVelocityHeatmap()
// Output: map["velocities"] → {layer→magnitude}
```

**Visualization Guide:**
```
🔴 Red (>0.1)    → Hot zone - aggressive weight updates
🟡 Yellow (0.01-0.1) → Active learning
🟢 Green (<0.01) → Stabilized, learning complete
```

**Interpretation:**
- **High velocity on E2-E5**: Experts are learning new patterns
- **Low velocity on E0-E1**: Foundation experts have stabilized (expected)
- **Sudden velocity spike**: Check for gradient explosion or learning rate issues

### 4. Semantic Drift Tracker

**What it shows:** Top 3 tokens whose embedding vectors have drifted furthest from Word2Vec baseline during training.

**Implementation:**
```go
drift := NewSemanticDriftTracker(w2vModel, vocab)

// During training:
drift.RecordEmbeddingState(embeddingTensor)

// At epoch end:
topShifts := drift.GetTopSemanticShifts(3)
// Output: []{"token": "file", "drift": 0.34}
```

**Example Output:**
```
🔄 "file" drifted by 0.34
  → Moved closer to "directory", away from "document"
  → Model understands file = technical object, not text

🔄 "command" drifted by 0.28
  → Moved closer to "script", away from "order"
  → Model learns domain-specific meaning
```

## Integration Steps

### Step 1: Initialize Observability in Your Trainer

```go
package main

import (
    "github.com/golangast/gollemer/internal/ai/moe"
    // ... other imports
)

func main() {
    trainer := &moe.Trainer{
        BestModelPath: "path/to/model.gob",
    }
    
    // Initialize observability
    trainer.InitializeObservability(
        8,        // 8 experts
        500,      // Window of 500 steps
        vocab,    // Your vocabulary
        w2vModel, // Word2Vec model
    )
}
```

### Step 2: Record Metrics During Training Loop

```go
for step, batch := range trainingData {
    // Forward pass
    expertIDs, tokenIDs, loss := model.Forward(batch)
    
    // Record metrics BEFORE backprop
    trainer.RecordTrainingStep(expertIDs, tokenIDs, loss)
    
    // Backprop and update
    model.Backward(loss)
    optimizer.Step()
    
    // Record weight snapshots for velocity
    if step%10 == 0 {
        trainer.RecordWeightSnapshot("layer_0", layer0.Weights)
        trainer.UpdateWeightVelocity("layer_0", layer0.Weights, embeddings)
    }
}

// Finish epoch
trainer.FinishEpoch(vocab)  // Logs report and resets windowed metrics
```

### Step 3: Start Metrics Server

```go
import "net/http"

aggregator := moe.StartMetricsServer(trainer, vocab, ":8080")

// In a goroutine:
go func() {
    log.Println("Dashboard: http://localhost:8080/docs/dashboard-enhanced.html")
    if err := http.ListenAndServe(":8080", nil); err != nil {
        log.Fatal(err)
    }
}()
```

### Step 4: Open Dashboard

Open `docs/dashboard-enhanced.html` in your browser (served by the HTTP server) or open it locally with `file:///` protocol if serving metrics at `localhost:8080`.

## Example Output

### Console Log (Trainer.Log)
```
╔════ MoE Observability Report (Epoch 42) ════╗
📚 Expert Lexicon (Top 5 tokens per expert):
   Expert 0: [the to and a is]
   Expert 1: [in of at on by]
   Expert 2: [file directory command script make]
   ...

📊 Category Loss Breakdown:
   Structural Words: Loss=0.12, Improvement=0.08 📉
   Core Verbs: Loss=0.19, Improvement=0.06 📉
   Technical Terms: Loss=0.34, Improvement=0.12 📉
   Punctuation: Loss=0.05, Improvement=0.02 📉

🔥 Weight Velocity Hotspots:
   Max Velocity: 0.045320

🔄 Top Semantic Drifts:
   Token "file" drifted by 0.3421
   Token "make" drifted by 0.2890
   Token "directory" drifted by 0.2156
╚═════════════════════════════════════════════╝
```

### Dashboard UI (4 Panels)
1. **📚 Expert Lexicon** - Grid showing top tokens per expert
2. **📊 Category Loss** - Card breakdown with color-coded improvements
3. **🔥 Weight Velocity** - Heatmap grid with intensity coloring
4. **🔄 Semantic Drift** - Progress bars showing token embedding shifts

## Configuration

Edit `data/config/observability_config.json`:

```json
{
  "token_categories": [
    {
      "name": "Your Custom Category",
      "keywords": ["word1", "word2", ...]
    }
  ],
  "observability_settings": {
    "window_size": 500,
    "top_k_tokens": 10,
    "top_k_drifts": 3,
    "metrics_refresh_interval_ms": 1000
  }
}
```

## Performance Considerations

### Memory Impact
- **TokenRoutingHistogram**: O(N_experts × N_vocab)
- **TokenCategoryLossTracker**: O(N_categories)
- **WeightVelocityTracker**: O(N_layers) temporary storage
- **SemanticDriftTracker**: O(N_vocab × embedding_dim)

### CPU Impact
- **Recording**: ~1-2% overhead per training step
- **Aggregation**: ~50ms per metrics collection (non-blocking)
- **Dashboard rendering**: Browser-side, no impact on training

### Optimization Tips
1. Reduce `window_size` if memory is tight (tradeoff: less statistics)
2. Increase `metrics_refresh_interval_ms` if dashboard feels slow
3. Set `top_k_tokens` to 5 instead of 10 to reduce visualization clutter

## Troubleshooting

### No metrics showing on dashboard
1. Check that metrics server is running: `curl localhost:8080/api/metrics/current`
2. Verify observability is initialized: `trainer.ObservabilityEnabled == true`
3. Ensure `RecordTrainingStep()` is called during training

### Missing semantic drift data
- Requires Word2Vec model: `w2vModel != nil`
- Initialize with: `trainer.InitializeObservability(..., w2vModel)`

### High CPU usage
- Reduce frequency of `UpdateWeightVelocity()` calls
- Increase `metrics_refresh_interval_ms` in config
- Reduce `top_k_tokens` (fewer tokens to decode)

## Advanced Usage

### Custom Token Categories

Load from JSON:
```go
config := LoadObservabilityConfig("data/config/observability_config.json")
categories := config.TokenCategories
```

Or define programmatically:
```go
categories := []TokenCategory{
    {
        Name: "My Domain",
        Color: "#ff6b6b",
        TokenIDs: []int{...},
    },
}
```

### Exporting Metrics

JSON export for analysis:
```go
jsonStr, _ := trainer.ExportObservabilityMetrics(vocab)
// Write to file or send to logging service
```

### Real-Time Alerts

Hook into metrics for automated alerts:
```go
metrics := aggregator.GetCurrentMetrics()
if metrics["weight_velocity"]["max_velocity"] > 0.5 {
    log.Println("⚠️ WARNING: Gradient explosion detected!")
}
```

## References

- Froebenius Norm: https://en.wikipedia.org/wiki/Frobenius_norm
- Cosine Similarity: https://en.wikipedia.org/wiki/Cosine_similarity
- MoE Load Balancing: https://arxiv.org/abs/2101.03961
- Word2Vec Embeddings: https://arxiv.org/abs/1301.3781

## Support

For issues or feature requests, refer to the main Gollemer README and documentation.
