package semantic

import (
	"fmt"
	. "github.com/zendrulat/nlptagger/neural/tensor"
	"math"
)

// StructuredSemanticLoss computes loss for structured semantic graph outputs
// It combines multiple loss components:
// 1. Operation classification loss
// 2. Resource identification loss
// 3. Argument role classification loss
// 4. Graph structure validity loss
type StructuredSemanticLoss struct {
	OperationWeight float64 // Weight for operation classification (default: 0.4)
	ResourceWeight  float64 // Weight for resource identification (default: 0.3)
	ArgumentWeight  float64 // Weight for argument classification (default: 0.2)
	StructureWeight float64 // Weight for graph structure validity (default: 0.1)
	LabelSmoothing  float64 // Label smoothing factor for regularization
}

// NewStructuredSemanticLoss creates a new loss function with default weights
func NewStructuredSemanticLoss() *StructuredSemanticLoss {
	return &StructuredSemanticLoss{
		OperationWeight: 0.4,
		ResourceWeight:  0.3,
		ArgumentWeight:  0.2,
		StructureWeight: 0.1,
		LabelSmoothing:  0.1,
	}
}

// ComputeLoss calculates the total structured semantic loss
func (ssl *StructuredSemanticLoss) ComputeLoss(
	operationLogits *Tensor,
	operationTargets []int,
	resourceLogits *Tensor,
	resourceTargets []int,
	argumentLogits *Tensor,
	argumentTargets []int,
	asgEdgeLogits *Tensor,
	asgEdgeTargets []int,
	paddingID int,
) (float64, map[string]float64, error) {

	losses := make(map[string]float64)
	totalLoss := 0.0

	// 1. Operation classification loss
	if operationLogits != nil && len(operationTargets) > 0 {
		opLoss, err := ssl.computeOperationLoss(operationLogits, operationTargets, paddingID)
		if err != nil {
			return 0, nil, fmt.Errorf("operation loss computation failed: %w", err)
		}
		losses["operation"] = opLoss
		totalLoss += ssl.OperationWeight * opLoss
	}

	// 2. Resource identification loss
	if resourceLogits != nil && len(resourceTargets) > 0 {
		resLoss, err := ssl.computeResourceLoss(resourceLogits, resourceTargets, paddingID)
		if err != nil {
			return 0, nil, fmt.Errorf("resource loss computation failed: %w", err)
		}
		losses["resource"] = resLoss
		totalLoss += ssl.ResourceWeight * resLoss
	}

	// 3. Argument role classification loss
	if argumentLogits != nil && len(argumentTargets) > 0 {
		argLoss, err := ssl.computeArgumentLoss(argumentLogits, argumentTargets, paddingID)
		if err != nil {
			return 0, nil, fmt.Errorf("argument loss computation failed: %w", err)
		}
		losses["argument"] = argLoss
		totalLoss += ssl.ArgumentWeight * argLoss
	}

	// 4. Graph structure validity loss
	if asgEdgeLogits != nil && len(asgEdgeTargets) > 0 {
		structLoss, err := ssl.computeStructureLoss(asgEdgeLogits, asgEdgeTargets, paddingID)
		if err != nil {
			return 0, nil, fmt.Errorf("structure loss computation failed: %w", err)
		}
		losses["structure"] = structLoss
		totalLoss += ssl.StructureWeight * structLoss
	}

	losses["total"] = totalLoss
	return totalLoss, losses, nil
}

// computeOperationLoss calculates loss for operation classification
func (ssl *StructuredSemanticLoss) computeOperationLoss(
	logits *Tensor,
	targets []int,
	paddingID int,
) (float64, error) {
	if logits == nil || len(targets) == 0 {
		return 0.0, nil
	}

	// Use standard cross-entropy loss
	loss := 0.0
	batchSize := len(targets)

	if len(logits.Shape) == 2 {
		// logits shape: [batchSize, vocabSize]
		vocabSize := logits.Shape[1]

		for i := 0; i < batchSize; i++ {
			targetID := targets[i]

			// Skip padding
			if targetID == paddingID {
				continue
			}

			// Get logits for this sample
			if i*vocabSize+targetID >= len(logits.Data) {
				continue
			}

			logit := logits.Data[i*vocabSize+targetID]

			// Apply label smoothing
			labelSmoothed := 1.0 - ssl.LabelSmoothing*(1.0/float64(vocabSize-1))

			// Cross-entropy with label smoothing
			loss -= labelSmoothed * logit

			// Add regularization term
			maxLogit := getMaxLogit(logits.Data, i*vocabSize, vocabSize)
			loss += math.Log(computeSumExp(logits.Data, i*vocabSize, vocabSize) - maxLogit)
		}
	}

	return loss / float64(batchSize), nil
}

// computeResourceLoss calculates loss for resource identification
// Resources are typically multi-token spans, so we use sequence loss
func (ssl *StructuredSemanticLoss) computeResourceLoss(
	logits *Tensor,
	targets []int,
	paddingID int,
) (float64, error) {
	if logits == nil || len(targets) == 0 {
		return 0.0, nil
	}

	loss := 0.0
	count := 0

	// For resource identification (sequence tagging style)
	vocabSize := logits.Shape[len(logits.Shape)-1]

	for i, targetID := range targets {
		if targetID == paddingID {
			continue
		}

		idx := i * vocabSize

		if idx+targetID >= len(logits.Data) {
			continue
		}

		logit := logits.Data[idx+targetID]
		loss -= logit
		loss += math.Log(computeSumExp(logits.Data, idx, vocabSize))
		count++
	}

	if count == 0 {
		return 0.0, nil
	}

	return loss / float64(count), nil
}

// computeArgumentLoss calculates loss for argument role classification
func (ssl *StructuredSemanticLoss) computeArgumentLoss(
	logits *Tensor,
	targets []int,
	paddingID int,
) (float64, error) {
	if logits == nil || len(targets) == 0 {
		return 0.0, nil
	}

	loss := 0.0
	count := 0

	vocabSize := logits.Shape[len(logits.Shape)-1]

	for i, targetID := range targets {
		if targetID == paddingID {
			continue
		}

		idx := i * vocabSize

		if idx+targetID >= len(logits.Data) {
			continue
		}

		logit := logits.Data[idx+targetID]
		loss -= logit
		loss += math.Log(computeSumExp(logits.Data, idx, vocabSize))
		count++
	}

	if count == 0 {
		return 0.0, nil
	}

	return loss / float64(count), nil
}

// computeStructureLoss calculates loss for graph edge validity
// This ensures the generated ASG has valid structure
func (ssl *StructuredSemanticLoss) computeStructureLoss(
	logits *Tensor,
	targets []int,
	paddingID int,
) (float64, error) {
	if logits == nil || len(targets) == 0 {
		return 0.0, nil
	}

	// For edge classification (valid/invalid relation types)
	loss := 0.0
	count := 0

	vocabSize := logits.Shape[len(logits.Shape)-1]

	for i, targetID := range targets {
		if targetID == paddingID {
			continue
		}

		idx := i * vocabSize

		if idx+targetID >= len(logits.Data) {
			continue
		}

		logit := logits.Data[idx+targetID]
		loss -= logit
		loss += math.Log(computeSumExp(logits.Data, idx, vocabSize))
		count++
	}

	if count == 0 {
		return 0.0, nil
	}

	// Apply structure penalty - invalid edges should have higher loss
	structurePenalty := 0.1 * float64(len(targets)-count) / float64(len(targets))
	return (loss / float64(count)) + structurePenalty, nil
}

// ValidateSemanticStructure checks if generated ASG is structurally valid
func (ssl *StructuredSemanticLoss) ValidateSemanticStructure(asg *AbstractSemanticGraph) (bool, []string) {
	errors := make([]string, 0)

	if asg == nil || asg.Root == nil {
		errors = append(errors, "ASG is nil or has no root")
		return false, errors
	}

	// Check 1: Root must be OPERATION
	if asg.Root.NodeType != "OPERATION" {
		errors = append(errors, fmt.Sprintf("root node type should be OPERATION, got %s", asg.Root.NodeType))
	}

	// Check 2: All edges must reference existing nodes
	for _, edge := range asg.Edges {
		if _, exists := asg.Nodes[edge.SourceID]; !exists {
			errors = append(errors, fmt.Sprintf("edge references non-existent source: %s", edge.SourceID))
		}
		if _, exists := asg.Nodes[edge.TargetID]; !exists {
			errors = append(errors, fmt.Sprintf("edge references non-existent target: %s", edge.TargetID))
		}
	}

	// Check 3: Must have at least one resource (unless it's a query-only operation)
	hasResource := false
	for _, node := range asg.Nodes {
		if node.NodeType == "RESOURCE" {
			hasResource = true
			break
		}
	}

	if !hasResource {
		// Some operations like "list" might not require explicit resources
		// This is more of a warning
		errors = append(errors, "ASG has no resource nodes (this may be valid for query-only operations)")
	}

	// Check 4: Operation node should have outgoing edges
	hasOutgoingEdges := false
	for _, edge := range asg.Edges {
		if edge.SourceID == asg.Root.ID {
			hasOutgoingEdges = true
			break
		}
	}

	if !hasOutgoingEdges && len(asg.Nodes) > 1 {
		errors = append(errors, "root operation has no outgoing edges")
	}

	return len(errors) == 0, errors
}

// Helper functions

func getMaxLogit(data []float64, start, size int) float64 {
	if len(data) == 0 || start+size > len(data) {
		return 0.0
	}

	maxVal := data[start]
	for i := start + 1; i < start+size; i++ {
		if data[i] > maxVal {
			maxVal = data[i]
		}
	}
	return maxVal
}

func computeSumExp(data []float64, start, size int) float64 {
	if len(data) == 0 || start+size > len(data) {
		return 0.0
	}

	maxVal := getMaxLogit(data, start, size)
	sum := 0.0

	for i := start; i < start+size; i++ {
		sum += math.Exp(data[i] - maxVal)
	}

	return sum + maxVal // Adding back maxVal for numerical stability
}

// ComputeMetrics calculates evaluation metrics for semantic parsing
func (ssl *StructuredSemanticLoss) ComputeMetrics(
	predictions *AbstractSemanticGraph,
	ground_truth *AbstractSemanticGraph,
) map[string]float64 {
	metrics := make(map[string]float64)

	if predictions == nil || ground_truth == nil {
		metrics["node_accuracy"] = 0.0
		metrics["edge_accuracy"] = 0.0
		metrics["structure_validity"] = 0.0
		return metrics
	}

	// Node accuracy: percentage of correctly predicted node types
	correctNodes := 0
	totalNodes := len(ground_truth.Nodes)

	for nodeID, gtNode := range ground_truth.Nodes {
		if predNode, exists := predictions.Nodes[nodeID]; exists {
			if predNode.NodeType == gtNode.NodeType && predNode.Value == gtNode.Value {
				correctNodes++
			}
		}
	}

	if totalNodes > 0 {
		metrics["node_accuracy"] = float64(correctNodes) / float64(totalNodes)
	}

	// Edge accuracy: percentage of correctly predicted edges
	correctEdges := 0
	totalEdges := len(ground_truth.Edges)

	for _, gtEdge := range ground_truth.Edges {
		for _, predEdge := range predictions.Edges {
			if predEdge.SourceID == gtEdge.SourceID &&
				predEdge.TargetID == gtEdge.TargetID &&
				predEdge.RelationType == gtEdge.RelationType {
				correctEdges++
				break
			}
		}
	}

	if totalEdges > 0 {
		metrics["edge_accuracy"] = float64(correctEdges) / float64(totalEdges)
	}

	// Structure validity
	isValid, _ := ssl.ValidateSemanticStructure(predictions)
	if isValid {
		metrics["structure_validity"] = 1.0
	} else {
		metrics["structure_validity"] = 0.0
	}

	return metrics
}
