package nertagger

import (
	"strings"
)

// ========== SEMANTIC ROLE LABELING (SRL) ==========

// SemanticRole represents a semantic role in a sentence
type SemanticRole string

const (
	Agent      SemanticRole = "AGENT"      // Who performs the action
	Patient    SemanticRole = "PATIENT"    // What is affected by the action
	Instrument SemanticRole = "INSTRUMENT" // How the action is performed
	Location   SemanticRole = "LOCATION"   // Where the action happens
	TimeArg    SemanticRole = "TIME"       // When the action happens
	Attribute  SemanticRole = "ATTRIBUTE"  // Properties of entities
	Modifier   SemanticRole = "MODIFIER"   // Additional descriptors
	Predicate  SemanticRole = "PREDICATE"  // The action/verb
)

// SRLAnalysis represents the semantic role analysis of a sentence
type SRLAnalysis struct {
	Tokens    []string
	PredIdx   int                    // Index of the main predicate (verb)
	Predicate string                 // The main predicate word
	Roles     map[int]SemanticRole   // Maps token index to semantic role
	Arguments map[SemanticRole][]int // Maps roles to token indices
}

// ========== DEPENDENCY PARSING ==========

// DependencyRelation represents a dependency edge in the parse tree
type DependencyRelation string

const (
	// Core relations
	Root    DependencyRelation = "ROOT"    // Root of the sentence
	Nsubj   DependencyRelation = "NSUBJ"   // Nominal subject
	Obj     DependencyRelation = "OBJ"     // Object
	IObj    DependencyRelation = "IOBJ"    // Indirect object
	OBlique DependencyRelation = "OBLIQUE" // Oblique argument
	CSubj   DependencyRelation = "CSUBJ"   // Clausal subject
	CComp   DependencyRelation = "CCOMP"   // Clausal complement

	// Modifier relations
	Amod   DependencyRelation = "AMOD"   // Adjectival modifier
	Advmod DependencyRelation = "ADVMOD" // Adverbial modifier
	Nmod   DependencyRelation = "NMOD"   // Nominal modifier
	Appos  DependencyRelation = "APPOS"  // Appositional modifier
	Acl    DependencyRelation = "ACL"    // Adjectival clause
	Relcl  DependencyRelation = "RELCL"  // Relative clause

	// Function word relations
	Case DependencyRelation = "CASE" // Case marking (prepositions)
	Mark DependencyRelation = "MARK" // Marker (subordinating)
	Cop  DependencyRelation = "COP"  // Copula (is, are, etc.)
	Aux  DependencyRelation = "AUX"  // Auxiliary
	Det  DependencyRelation = "DET"  // Determiner (the, a, etc.)

	// Compound relations
	Compound DependencyRelation = "COMPOUND" // Compound words
	Flat     DependencyRelation = "FLAT"     // Multi-word expressions
	MWE      DependencyRelation = "MWE"      // Multi-word expressions

	// Coordination
	Conj DependencyRelation = "CONJ" // Conjunct
	Cc   DependencyRelation = "CC"   // Coordinating conjunction

	// Other
	Punct DependencyRelation = "PUNCT" // Punctuation
	Dep   DependencyRelation = "DEP"   // Unknown dependency
)

// DependencyEdge represents an edge in the dependency graph
type DependencyEdge struct {
	Governor  int // Index of the head
	Dependent int // Index of the dependent
	Relation  DependencyRelation
}

// DependencyTree represents the dependency parse tree of a sentence
type DependencyTree struct {
	Tokens    []string
	Edges     []DependencyEdge
	Children  map[int][]int              // Maps token index to child indices
	Parent    map[int]int                // Maps token index to parent index
	Relations map[int]DependencyRelation // Maps token index to its relation to parent
}

// ========== COMBINED SRL + DEPENDENCY PARSER ==========

// SemanticParser combines SRL and dependency parsing
type SemanticParser struct {
	tokens  []string
	posTags []string
	tree    *DependencyTree
	srl     *SRLAnalysis
}

// NewSemanticParser creates a new semantic parser
func NewSemanticParser(tokens []string, posTags []string) *SemanticParser {
	return &SemanticParser{
		tokens:  tokens,
		posTags: posTags,
	}
}

// Parse performs semantic role labeling and dependency parsing
func (sp *SemanticParser) Parse() (*DependencyTree, *SRLAnalysis) {
	sp.tree = sp.buildDependencyTree()
	sp.srl = sp.performSRL()
	return sp.tree, sp.srl
}

// buildDependencyTree constructs a simplified dependency tree
// Based on POS tags and heuristics
func (sp *SemanticParser) buildDependencyTree() *DependencyTree {
	tree := &DependencyTree{
		Tokens:    sp.tokens,
		Edges:     []DependencyEdge{},
		Children:  make(map[int][]int),
		Parent:    make(map[int]int),
		Relations: make(map[int]DependencyRelation),
	}

	// Initialize parent as -1 (unattached)
	for i := range sp.tokens {
		tree.Parent[i] = -1
	}

	// Find the main predicate (usually the main verb)
	rootIdx := sp.findMainVerb()
	if rootIdx == -1 {
		rootIdx = 0 // Fallback to first token
	}

	tree.Parent[rootIdx] = rootIdx
	tree.Relations[rootIdx] = Root

	// Attach subjects, objects, and modifiers to the root
	for i := range sp.posTags {
		if i == rootIdx {
			continue
		}

		relation, headIdx := sp.determineRelation(i, rootIdx)
		if headIdx == -1 {
			headIdx = rootIdx
		}

		tree.Parent[i] = headIdx
		tree.Relations[i] = relation
		tree.Children[headIdx] = append(tree.Children[headIdx], i)

		tree.Edges = append(tree.Edges, DependencyEdge{
			Governor:  headIdx,
			Dependent: i,
			Relation:  relation,
		})
	}

	return tree
}

// findMainVerb locates the main verb in the sentence
func (sp *SemanticParser) findMainVerb() int {
	for i, pos := range sp.posTags {
		// Look for main verbs (VB, VBZ, VBP, VBD, VBN, VBG)
		if strings.HasPrefix(pos, "VB") && i > 0 {
			// Prefer verbs that aren't modals or auxiliaries
			token := strings.ToLower(sp.tokens[i])
			if !isAuxiliary(token) && !isModal(token) {
				return i
			}
		}
	}

	// Fallback: look for action verbs based on known keywords
	for i, token := range sp.tokens {
		lower := strings.ToLower(token)
		if isActionVerb(lower) {
			return i
		}
	}

	return -1
}

// determineRelation determines the dependency relation between a token and a head
func (sp *SemanticParser) determineRelation(tokenIdx, rootIdx int) (DependencyRelation, int) {
	pos := sp.posTags[tokenIdx]

	// Subject detection (NNP, NN before verb)
	if (strings.HasPrefix(pos, "NN") || strings.HasPrefix(pos, "PRP")) && tokenIdx < rootIdx {
		return Nsubj, rootIdx
	}

	// Object detection (NN after verb)
	if strings.HasPrefix(pos, "NN") && tokenIdx > rootIdx {
		return Obj, rootIdx
	}

	// Adjectival modifier
	if strings.HasPrefix(pos, "JJ") {
		// Find the noun it modifies
		for i := tokenIdx + 1; i < len(sp.tokens); i++ {
			if strings.HasPrefix(sp.posTags[i], "NN") {
				return Amod, i
			}
		}
		return Amod, rootIdx
	}

	// Adverbial modifier
	if strings.HasPrefix(pos, "RB") {
		return Advmod, rootIdx
	}

	// Preposition (case marker)
	if pos == "IN" {
		return Case, rootIdx
	}

	// Determiner
	if pos == "DT" {
		// Find the noun it determines
		for i := tokenIdx + 1; i < len(sp.tokens); i++ {
			if strings.HasPrefix(sp.posTags[i], "NN") {
				return Det, i
			}
		}
		return Det, rootIdx
	}

	// Compound noun
	if strings.HasPrefix(pos, "NN") && tokenIdx > 0 {
		if strings.HasPrefix(sp.posTags[tokenIdx-1], "NN") {
			return Compound, tokenIdx - 1
		}
	}

	return Dep, rootIdx
}

// performSRL performs semantic role labeling
func (sp *SemanticParser) performSRL() *SRLAnalysis {
	analysis := &SRLAnalysis{
		Tokens:    sp.tokens,
		Roles:     make(map[int]SemanticRole),
		Arguments: make(map[SemanticRole][]int),
	}

	// Find the main predicate
	predIdx := sp.findMainVerb()
	if predIdx == -1 {
		predIdx = 0
	}

	analysis.PredIdx = predIdx
	analysis.Predicate = sp.tokens[predIdx]
	analysis.Roles[predIdx] = Predicate
	analysis.Arguments[Predicate] = append(analysis.Arguments[Predicate], predIdx)

	// Assign semantic roles based on dependency relations and POS tags
	for i := range sp.tokens {
		if i == predIdx {
			continue
		}

		role := sp.assignSemanticRole(i, predIdx)
		analysis.Roles[i] = role
		analysis.Arguments[role] = append(analysis.Arguments[role], i)
	}

	return analysis
}

// assignSemanticRole assigns a semantic role to a token
func (sp *SemanticParser) assignSemanticRole(tokenIdx, predIdx int) SemanticRole {
	token := strings.ToLower(sp.tokens[tokenIdx])
	pos := sp.posTags[tokenIdx]

	// Subjects are agents
	if strings.HasPrefix(pos, "NNP") || strings.HasPrefix(pos, "NN") {
		if tokenIdx < predIdx {
			return Agent
		} else {
			return Patient
		}
	}

	// Proper nouns are often entities
	if strings.HasPrefix(pos, "NNP") {
		return Patient
	}

	// Adjectives and adverbs are modifiers
	if strings.HasPrefix(pos, "JJ") {
		return Modifier
	}

	if strings.HasPrefix(pos, "RB") {
		return Modifier
	}

	// Prepositions indicate oblique arguments or locations
	if pos == "IN" {
		if isLocationPrep(token) {
			return Location
		}
		if isTimePrep(token) {
			return TimeArg
		}
		return Attribute
	}

	// Determiners don't have semantic roles
	if pos == "DT" {
		return Modifier
	}

	return Modifier
}

// ========== HELPER FUNCTIONS ==========

func isAuxiliary(token string) bool {
	auxiliaries := map[string]bool{
		"is": true, "are": true, "am": true, "was": true, "were": true,
		"be": true, "been": true, "being": true,
		"have": true, "has": true, "had": true,
		"do": true, "does": true, "did": true,
	}
	return auxiliaries[strings.ToLower(token)]
}

func isModal(token string) bool {
	modals := map[string]bool{
		"can": true, "could": true, "may": true, "might": true,
		"must": true, "should": true, "would": true, "will": true,
		"shall": true, "ought": true,
	}
	return modals[strings.ToLower(token)]
}

func isActionVerb(token string) bool {
	actions := map[string]bool{
		"create": true, "make": true, "build": true, "generate": true,
		"add": true, "insert": true, "append": true, "write": true,
		"delete": true, "remove": true, "drop": true,
		"update": true, "modify": true, "change": true,
		"read": true, "get": true, "fetch": true,
		"run": true, "execute": true, "start": true, "launch": true,
		"stop": true, "kill": true, "terminate": true,
		"move": true, "copy": true, "rename": true,
		"list": true, "show": true, "display": true, "print": true,
	}
	return actions[token]
}

func isLocationPrep(token string) bool {
	locPreps := map[string]bool{
		"in": true, "at": true, "on": true, "under": true,
		"over": true, "above": true, "below": true, "inside": true,
		"outside": true, "near": true, "between": true, "among": true,
	}
	return locPreps[token]
}

func isTimePrep(token string) bool {
	timePreps := map[string]bool{
		"at": true, "on": true, "in": true, "during": true,
		"before": true, "after": true, "since": true, "until": true,
		"throughout": true, "within": true,
	}
	return timePreps[token]
}

// ========== ENTITY EXTRACTION WITH SRL ==========

// ExtractEntitiesWithSRL extracts entities using semantic role labeling
// This improves accuracy by understanding grammatical roles
func ExtractEntitiesWithSRL(tokens []string, posTags []string) map[SemanticRole][]string {
	parser := NewSemanticParser(tokens, posTags)
	_, srl := parser.Parse()

	entities := make(map[SemanticRole][]string)

	// Extract tokens for each semantic role
	for role, indices := range srl.Arguments {
		for _, idx := range indices {
			entities[role] = append(entities[role], tokens[idx])
		}
	}

	return entities
}

// ========== STOP-WORD FILTER FOR PARAMETERS ==========

// StopWordFilter filters out intent keywords from parameter values
type StopWordFilter struct {
	IntentKeywords map[string]bool
	ObjectKeywords map[string]bool
	FilteredTokens map[string]bool
}

// NewStopWordFilter creates a new stop word filter
func NewStopWordFilter() *StopWordFilter {
	return &StopWordFilter{
		IntentKeywords: map[string]bool{
			"create": true, "make": true, "add": true, "generate": true,
			"delete": true, "remove": true, "update": true, "modify": true,
			"list": true, "show": true, "display": true,
			"go": true, "move": true, "copy": true,
			"run": true, "start": true, "stop": true,
			"read": true, "cat": true, "grep": true,
			"named": true, "called": true, "for": true, "with": true,
			"in": true, "into": true, "to": true, "from": true,
		},
		ObjectKeywords: map[string]bool{
			"file": true, "folder": true, "directory": true, "database": true,
			"handler": true, "webserver": true, "page": true, "form": true,
			"structure": true, "table": true, "component": true, "endpoint": true,
			"route": true, "view": true, "app": true, "application": true,
		},
		FilteredTokens: make(map[string]bool),
	}
}

// FilterParameter removes intent/object keywords from a parameter value
func (sf *StopWordFilter) FilterParameter(paramValue string) string {
	tokens := strings.Fields(paramValue)
	var filtered []string

	for _, token := range tokens {
		lower := strings.ToLower(token)

		// Remove trailing punctuation for comparison
		cleanToken := strings.TrimRight(lower, ".,!?;:")

		// Skip if it's an intent or object keyword
		if sf.IntentKeywords[cleanToken] || sf.ObjectKeywords[cleanToken] {
			sf.FilteredTokens[cleanToken] = true
			continue
		}

		filtered = append(filtered, token)
	}

	return strings.Join(filtered, " ")
}

// IsValidParameter checks if a parameter is valid (not empty after filtering)
func (sf *StopWordFilter) IsValidParameter(paramValue string) bool {
	filtered := sf.FilterParameter(paramValue)
	return strings.TrimSpace(filtered) != ""
}

// ========== LEAF NODE EXTRACTION ==========

// ExtractLeafNodes extracts leaf nodes (entities) from the dependency tree
// Leaf nodes are tokens with no dependent children
func (dt *DependencyTree) ExtractLeafNodes() []string {
	var leaves []string

	for i := range dt.Tokens {
		// Leaf nodes have no children or only determiners/case markers as children
		if len(dt.Children[i]) == 0 {
			leaves = append(leaves, dt.Tokens[i])
		} else {
			// Check if all children are function words
			allFunctionWords := true
			for _, childIdx := range dt.Children[i] {
				relation := dt.Relations[childIdx]
				if relation != Det && relation != Case && relation != Punct {
					allFunctionWords = false
					break
				}
			}
			if allFunctionWords {
				leaves = append(leaves, dt.Tokens[i])
			}
		}
	}

	return leaves
}

// FindEntityHeads finds the head nouns that are likely entity names
// In "Create a bash file named Jim", Jim is the entity head
func (dt *DependencyTree) FindEntityHeads() []int {
	var heads []int

	for i, pos := range dt.Tokens {
		// Entity heads are typically nouns (especially proper nouns)
		// that are not dependent on other nouns
		if strings.HasPrefix(pos, "NN") {
			// Check if it's attached to a noun or verb
			parent := dt.Parent[i]
			if parent != -1 && parent != i {
				parentPos := dt.Tokens[parent]
				// Good entity if attached to verb or adjective
				if strings.HasPrefix(parentPos, "VB") || strings.HasPrefix(parentPos, "JJ") {
					heads = append(heads, i)
				}
			}
		}
	}

	return heads
}
