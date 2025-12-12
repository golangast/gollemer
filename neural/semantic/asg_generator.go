package semantic

import (
	"fmt"
	"strings"
)

// GraphNode represents a node in the Abstract Semantic Graph
type GraphNode struct {
	ID           string                 `json:"id"`
	NodeType     string                 `json:"type"`         // "OPERATION", "RESOURCE", "ARGUMENT", "CONSTRAINT"
	Label        string                 `json:"label"`        // Human-readable label
	Value        string                 `json:"value"`        // Actual value
	Properties   map[string]interface{} `json:"properties"`   // Node-specific properties
	Dependencies []string               `json:"dependencies"` // IDs of dependent nodes
}

// GraphEdge represents an edge/relation in the ASG
type GraphEdge struct {
	SourceID     string                 `json:"source_id"`
	TargetID     string                 `json:"target_id"`
	RelationType string                 `json:"relation_type"`
	Weight       float64                `json:"weight"`
	Properties   map[string]interface{} `json:"properties"`
}

// AbstractSemanticGraph represents the complete semantic graph for a query
type AbstractSemanticGraph struct {
	Nodes    map[string]*GraphNode  `json:"nodes"`
	Edges    []*GraphEdge           `json:"edges"`
	Root     *GraphNode             `json:"root_node"`
	Metadata map[string]interface{} `json:"metadata"`
}

// ASGGenerator generates Abstract Semantic Graphs from semantic roles
type ASGGenerator struct {
	NodeIDCounter int
	RoleMapping   map[string]string // Maps role types to node types
}

// NewASGGenerator creates a new ASG generator
func NewASGGenerator() *ASGGenerator {
	return &ASGGenerator{
		NodeIDCounter: 0,
		RoleMapping: map[string]string{
			"OPERATION": "OPERATION",
			"RESOURCE":  "RESOURCE",
			"ARGUMENT":  "ARGUMENT",
			"MODIFIER":  "CONSTRAINT",
		},
	}
}

// GenerateFromSemanticRoles constructs an ASG from extracted semantic roles
// This is the core method that builds structured graphs from flat semantic role output
func (ag *ASGGenerator) GenerateFromSemanticRoles(
	operation string,
	resources []map[string]string,
	arguments []map[string]string,
	modifiers []map[string]string,
) *AbstractSemanticGraph {
	asg := &AbstractSemanticGraph{
		Nodes:    make(map[string]*GraphNode),
		Edges:    make([]*GraphEdge, 0),
		Metadata: make(map[string]interface{}),
	}

	// Create root operation node
	opNodeID := ag.generateNodeID()
	opNode := &GraphNode{
		ID:       opNodeID,
		NodeType: "OPERATION",
		Label:    strings.ToTitle(operation),
		Value:    operation,
		Properties: map[string]interface{}{
			"main_verb": true,
		},
		Dependencies: make([]string, 0),
	}
	asg.Nodes[opNodeID] = opNode
	asg.Root = opNode

	// Add resource nodes
	resourceNodeIDs := make([]string, 0)
	for _, resource := range resources {
		resNodeID := ag.generateNodeID()
		resType := resource["type"]
		resName := resource["name"]

		resNode := &GraphNode{
			ID:       resNodeID,
			NodeType: "RESOURCE",
			Label:    resName,
			Value:    resName,
			Properties: map[string]interface{}{
				"resource_type":  resType,
				"canonical_path": resource["path"],
			},
			Dependencies: make([]string, 0),
		}
		asg.Nodes[resNodeID] = resNode
		resourceNodeIDs = append(resourceNodeIDs, resNodeID)

		// Add edge from operation to resource
		asg.Edges = append(asg.Edges, &GraphEdge{
			SourceID:     opNodeID,
			TargetID:     resNodeID,
			RelationType: "ACTS_ON",
			Weight:       1.0,
			Properties:   make(map[string]interface{}),
		})

		opNode.Dependencies = append(opNode.Dependencies, resNodeID)
	}

	// Add argument nodes and connect to resources
	for _, arg := range arguments {
		argNodeID := ag.generateNodeID()
		argRole := arg["role"]
		argValue := arg["value"]

		argNode := &GraphNode{
			ID:       argNodeID,
			NodeType: "ARGUMENT",
			Label:    fmt.Sprintf("%s: %s", argRole, argValue),
			Value:    argValue,
			Properties: map[string]interface{}{
				"argument_role": argRole,
			},
			Dependencies: make([]string, 0),
		}
		asg.Nodes[argNodeID] = argNode

		// Connect argument to operation
		asg.Edges = append(asg.Edges, &GraphEdge{
			SourceID:     opNodeID,
			TargetID:     argNodeID,
			RelationType: "HAS_ARGUMENT",
			Weight:       0.9,
			Properties: map[string]interface{}{
				"arg_type": argRole,
			},
		})

		// Connect argument to all resources (properties apply to all)
		for _, resID := range resourceNodeIDs {
			if argRole == "PATH" || argRole == "NAME" || argRole == "MODE" {
				asg.Edges = append(asg.Edges, &GraphEdge{
					SourceID:     resID,
					TargetID:     argNodeID,
					RelationType: "HAS_PROPERTY",
					Weight:       0.85,
					Properties: map[string]interface{}{
						"property_name": argRole,
					},
				})
				asg.Nodes[resID].Dependencies = append(asg.Nodes[resID].Dependencies, argNodeID)
			}
		}
	}

	// Add modifier/constraint nodes
	for _, modifier := range modifiers {
		modNodeID := ag.generateNodeID()
		modValue := modifier["value"]

		modNode := &GraphNode{
			ID:       modNodeID,
			NodeType: "CONSTRAINT",
			Label:    modValue,
			Value:    modValue,
			Properties: map[string]interface{}{
				"constraint_type": "MODIFIER",
			},
			Dependencies: make([]string, 0),
		}
		asg.Nodes[modNodeID] = modNode

		// Modifiers apply to the operation
		asg.Edges = append(asg.Edges, &GraphEdge{
			SourceID:     modNodeID,
			TargetID:     opNodeID,
			RelationType: "MODIFIES",
			Weight:       0.8,
			Properties: map[string]interface{}{
				"affects_operation": true,
			},
		})

		opNode.Dependencies = append(opNode.Dependencies, modNodeID)
	}

	// Add metadata
	asg.Metadata["operation"] = operation
	asg.Metadata["resource_count"] = len(resourceNodeIDs)
	asg.Metadata["argument_count"] = len(arguments)
	asg.Metadata["modifier_count"] = len(modifiers)
	asg.Metadata["total_nodes"] = len(asg.Nodes)
	asg.Metadata["total_edges"] = len(asg.Edges)

	return asg
}

// GenerateExecutionPlan converts ASG into an executable workflow
func (ag *ASGGenerator) GenerateExecutionPlan(asg *AbstractSemanticGraph) map[string]interface{} {
	plan := make(map[string]interface{})

	if asg.Root == nil {
		return plan
	}

	// Main operation
	plan["operation"] = asg.Root.Value

	// Target resources (objects being operated on)
	targetResources := make([]map[string]interface{}, 0)
	for nodeID, node := range asg.Nodes {
		if node.NodeType == "RESOURCE" {
			resource := make(map[string]interface{})
			resource["id"] = nodeID
			resource["name"] = node.Value
			resource["type"] = node.Properties["resource_type"]
			resource["path"] = node.Properties["canonical_path"]

			// Find properties of this resource
			properties := make(map[string]interface{})
			for _, edge := range asg.Edges {
				if edge.TargetID == nodeID && edge.RelationType == "HAS_PROPERTY" {
					if propNode, exists := asg.Nodes[edge.SourceID]; exists {
						propRole := propNode.Properties["argument_role"]
						properties[fmt.Sprintf("%v", propRole)] = propNode.Value
					}
				}
			}
			if len(properties) > 0 {
				resource["properties"] = properties
			}

			targetResources = append(targetResources, resource)
		}
	}
	if len(targetResources) > 0 {
		plan["target_resources"] = targetResources
	}

	// Arguments/options
	arguments := make(map[string]interface{})
	for _, node := range asg.Nodes {
		if node.NodeType == "ARGUMENT" {
			argRole := node.Properties["argument_role"]
			arguments[fmt.Sprintf("%v", argRole)] = node.Value
		}
	}
	if len(arguments) > 0 {
		plan["arguments"] = arguments
	}

	// Constraints/modifiers
	constraints := make([]string, 0)
	for _, node := range asg.Nodes {
		if node.NodeType == "CONSTRAINT" {
			constraints = append(constraints, node.Value)
		}
	}
	if len(constraints) > 0 {
		plan["constraints"] = constraints
	}

	plan["metadata"] = asg.Metadata

	return plan
}

// ValidateASG checks the structural validity of the ASG
func (ag *ASGGenerator) ValidateASG(asg *AbstractSemanticGraph) error {
	if asg.Root == nil {
		return fmt.Errorf("ASG must have a root operation node")
	}

	if asg.Root.NodeType != "OPERATION" {
		return fmt.Errorf("root node must be an OPERATION, got %s", asg.Root.NodeType)
	}

	// Validate all edges reference existing nodes
	for _, edge := range asg.Edges {
		if _, exists := asg.Nodes[edge.SourceID]; !exists {
			return fmt.Errorf("edge references non-existent source node: %s", edge.SourceID)
		}
		if _, exists := asg.Nodes[edge.TargetID]; !exists {
			return fmt.Errorf("edge references non-existent target node: %s", edge.TargetID)
		}
	}

	// Validate at least one resource node exists (unless operation is standalone)
	hasResources := false
	for _, node := range asg.Nodes {
		if node.NodeType == "RESOURCE" {
			hasResources = true
			break
		}
	}

	if !hasResources && asg.Root.Value != "" {
		// Some operations (like "list") might work without explicit resources
		if !strings.EqualFold(asg.Root.Value, "list") && !strings.EqualFold(asg.Root.Value, "show") {
			// This is a warning, not an error
		}
	}

	return nil
}

// Helper function to generate unique node IDs
func (ag *ASGGenerator) generateNodeID() string {
	ag.NodeIDCounter++
	return fmt.Sprintf("node_%d", ag.NodeIDCounter)
}
