package moe

import (
	"encoding/gob"
	"fmt"
	"strings"

	"github.com/golangast/gollemer/internal/ai/neural/nn"
	"github.com/golangast/gollemer/internal/ai/neural/tensor"
)

func init() {
	gob.Register(&GrammarExpert{})
}

// GrammarRoles maps a POS category to a learnable bias slot index (0-7).
// This lets each of the 8 grammar experts specialise in one structural role.
var GrammarRoles = []string{
	"PRON",  // 0 — pronouns
	"VERB",  // 1 — copula / aux verbs
	"AUX",   // 2 — auxiliary verbs (will, can, should)
	"ADJ",   // 3 — adjectives / adverbs
	"NOUN",  // 4 — nouns / names
	"PREP",  // 5 — prepositions / conjunctions
	"GREET", // 6 — greetings / discourse markers
	"OTHER", // 7 — everything else (residual)
}

// GrammarRoleIndex returns the role index for a coarse POS tag.
func GrammarRoleIndex(posTag string) int {
	t := strings.ToUpper(posTag)
	for i, r := range GrammarRoles {
		if t == r {
			return i
		}
	}
	return 7 // OTHER
}

// GrammarExpert is a lightweight MLP expert that adds a per-role learned logit
// bias after its standard two-layer computation.  It is intended to specialise
// in one syntactic role (pronoun, verb, adjective, etc.) so the MoE layer can
// route tokens to the structurally appropriate sub-network.
type GrammarExpert struct {
	ID       int
	RoleID   int    // which grammar role this expert owns (0-7)
	RoleName string // human-readable (e.g. "PRON")

	inputDim  int
	hiddenDim int
	outputDim int

	FC1 *nn.Linear
	FC2 *nn.Linear

	// Role bias: a small learned offset vector added to the output.
	// Shape [outputDim] — initialised near-zero so the expert starts neutral.
	RoleBias *tensor.Tensor

	health     float32
	isTraining bool

	lastInput   *tensor.Tensor
	lastReLUOut *tensor.Tensor
	
	// Multi-token memory (EMA of inputs to catch temporal patterns)
	ContextMemory *tensor.Tensor
}

// NewGrammarExpert creates a GrammarExpert that owns the given grammar role (0-7).
// hiddenDim is intentionally narrower than a standard expert to force syntactic focus.
func NewGrammarExpert(id, roleID, inputDim, outputDim int) (*GrammarExpert, error) {
	if roleID < 0 || roleID >= len(GrammarRoles) {
		roleID = roleID % len(GrammarRoles)
	}
	hiddenDim := inputDim / 2
	if hiddenDim < 64 {
		hiddenDim = 64
	}

	fc1, err := nn.NewLinear(inputDim, hiddenDim)
	if err != nil {
		return nil, fmt.Errorf("GrammarExpert FC1: %w", err)
	}
	fc2, err := nn.NewLinear(hiddenDim, outputDim)
	if err != nil {
		return nil, fmt.Errorf("GrammarExpert FC2: %w", err)
	}

	// Role bias — start near zero
	roleBias := tensor.NewTensor([]int{1, outputDim}, make([]float32, outputDim), true)

	return &GrammarExpert{
		ID:        id,
		RoleID:    roleID,
		RoleName:  GrammarRoles[roleID],
		inputDim:  inputDim,
		hiddenDim: hiddenDim,
		outputDim: outputDim,
		FC1:       fc1,
		FC2:       fc2,
		RoleBias:  roleBias,
		health:    0.125,
		ContextMemory: tensor.NewTensor([]int{1, inputDim}, make([]float32, inputDim), false),
	}, nil
}

// Forward: standard two-layer MLP + role bias.
func (e *GrammarExpert) Forward(input *tensor.Tensor) (*tensor.Tensor, error) {
	e.lastInput = input

	h1, err := e.FC1.Forward(input)
	if err != nil {
		return nil, fmt.Errorf("GrammarExpert(role=%s) FC1: %w", e.RoleName, err)
	}
	tensor.ReLUVector(h1.Data)
	e.lastReLUOut = h1

	out, err := e.FC2.Forward(h1)
	if err != nil {
		return nil, fmt.Errorf("GrammarExpert(role=%s) FC2: %w", e.RoleName, err)
	}

	// Add the learned role bias (broadcast over batch tokens)
	out.AddWithBroadcast(e.RoleBias)

	// Update Context Memory (EMA)
	if e.ContextMemory != nil && len(input.Data) >= e.inputDim {
		// Vectorized update: Context = 0.9*Context + 0.1*Mean(Input)
		inputMean, _ := input.Mean(0)
		e.ContextMemory.Scale(0.9)
		e.ContextMemory.Add(inputMean.Scale(0.1))
		inputMean.Release()
	}

	return out, nil
}

func (e *GrammarExpert) Backward(grad *tensor.Tensor) error {
	if !e.isTraining || e.lastReLUOut == nil {
		return nil
	}
	if err := e.FC2.Backward(grad); err != nil {
		return err
	}
	fc2InputGrad := e.FC2.Input().Grad
	if fc2InputGrad == nil {
		return nil
	}
	h1GradData := make([]float32, len(e.lastReLUOut.Data))
	for i, v := range e.lastReLUOut.Data {
		if v > 0 && i < len(fc2InputGrad.Data) {
			h1GradData[i] = fc2InputGrad.Data[i]
		}
	}
	h1Grad := tensor.NewTensor(e.lastReLUOut.Shape, h1GradData, false)
	return e.FC1.Backward(h1Grad)
}

func (e *GrammarExpert) Parameters() []*tensor.Tensor {
	params := e.FC1.Parameters()
	params = append(params, e.FC2.Parameters()...)
	params = append(params, e.RoleBias)
	return params
}

func (e *GrammarExpert) Inputs() []*tensor.Tensor {
	if e.lastInput == nil {
		return nil
	}
	return []*tensor.Tensor{e.lastInput}
}

func (e *GrammarExpert) Description() string {
	return fmt.Sprintf("GrammarExpert(id=%d role=%s)", e.ID, e.RoleName)
}

func (e *GrammarExpert) SetMode(training bool) { e.isTraining = training }

func (e *GrammarExpert) ClearState() {
	e.lastInput = nil
	e.lastReLUOut = nil
	if e.FC1 != nil {
		e.FC1.ClearState()
	}
	if e.FC2 != nil {
		e.FC2.ClearState()
	}
}

func (e *GrammarExpert) UpdateHealth(wasUsed bool) {
	const decay = 0.99
	var current float32
	if wasUsed {
		current = 1.0
	}
	e.health = current*(1-decay) + e.health*decay
}

func (e *GrammarExpert) IsStagnant() bool  { return e.health < 0.01 }
func (e *GrammarExpert) ClipWeights(_ float32) {}

func (e *GrammarExpert) EvolutionaryReset(winner Expert, jitter float32) {
	w, ok := winner.(*GrammarExpert)
	if !ok {
		return
	}
	e.FC1.Weights.CopyFrom(w.FC1.Weights)
	e.FC2.Weights.CopyFrom(w.FC2.Weights)
	e.FC1.Weights.ApplyJitter(jitter)
	e.FC2.Weights.ApplyJitter(jitter)
}

func (e *GrammarExpert) Shake(intensity float32) {
	e.FC1.Weights.ApplyJitter(intensity)
	e.FC2.Weights.ApplyJitter(intensity)
}

func (e *GrammarExpert) Resize(newOutputDim int) {
	if e.FC2 == nil {
		return
	}
	oldOut := e.outputDim
	inputDim := e.FC2.Weights.Shape[0]
	copyLimit := oldOut
	if newOutputDim < copyLimit {
		copyLimit = newOutputDim
	}
	newW := make([]float32, inputDim*newOutputDim)
	for row := 0; row < inputDim; row++ {
		copy(newW[row*newOutputDim:row*newOutputDim+copyLimit], e.FC2.Weights.Data[row*oldOut:row*oldOut+copyLimit])
	}
	newLinear, _ := nn.NewLinear(inputDim, newOutputDim)
	newLinear.Weights.Data = newW
	e.FC2 = newLinear
	e.outputDim = newOutputDim

	// Resize role bias
	newBias := tensor.NewTensor([]int{1, newOutputDim}, make([]float32, newOutputDim), true)
	copy(newBias.Data, e.RoleBias.Data[:min(len(e.RoleBias.Data), newOutputDim)])
	e.RoleBias = newBias
}

func (e *GrammarExpert) ToGPU() {
	if e.FC1 != nil {
		e.FC1.ToGPU()
	}
	if e.FC2 != nil {
		e.FC2.ToGPU()
	}
}

func (e *GrammarExpert) SyncParameters() error { return nil }
// SeedGrammarBias applies a structural prior to the expert's output bias.
// This jumpstarts specialization by making the expert naturally "prefer" tokens
// that match its assigned syntactic role (e.g. PRON expert gets a boost for 'i', 'you').
func (ge *GrammarExpert) SeedGrammarBias(vocabSize int, tokenToWord []string) {
	if ge.FC2.Biases == nil {
		ge.FC2.Biases = tensor.NewTensor([]int{vocabSize}, make([]float32, vocabSize), true)
	}
	
	boostCount := 0
	for id, word := range tokenToWord {
		if id >= vocabSize {
			continue
		}
		
		role := MapWordToGrammarType(word)
		if role == ge.RoleName {
			// Apply a significant structural prior (+5.0 logit boost)
			ge.FC2.Biases.Data[id] += 5.0
			boostCount++
		}
	}
	
	if boostCount > 0 {
		fmt.Printf("🧬 [MoE] Seeded Expert E%d (%s) with %d role-specific biases\n", ge.ID, ge.RoleName, boostCount)
	}
}

func (ge *GrammarExpert) GetID() int {
	return ge.ID
}
