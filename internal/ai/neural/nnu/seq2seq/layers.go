package seq2seq

// Layers pilot (2026-11-12): generic intermediate layers for the
// social seq2seq, each with one narrow job, each entering as an
// added feature or a conditioning signal — never as a channel the
// reply must pass through. The design, bars, and ablation schedule
// are pre-registered in ~/workspace/layers-pilot-scratch/DESIGN.md.
//
// LayersModel wraps the production Seq2Seq untouched: with every
// layer off it is not constructed at all (the layers trainer is a
// separate entry point), and the production trainer/chat paths
// never see this type. All layer parameters are trained jointly
// with the word model by TrainBatchLayers (train_layers.go).
//
// Layer inventory (DESIGN §3):
//   char    — mean char-trigram embedding added to token embeddings
//   class   — word-class embedding added to token embeddings
//   intent  — head on the encoder state + seed conditioning
//   tone    — head on the encoder state + seed conditioning
//   concept — reply gist codes (3 segments), head + decoder prefix
//   copy    — the house PointerGen over encoder input positions
//   turn    — previous reply's concept codes, seed conditioning
//   memory  — session facts as encoder-input prefix tokens
//             (trainer-side input construction; no parameters here)

import (
	"encoding/gob"
	"fmt"
	"hash/fnv"
	"math"
	"os"

	"github.com/golangast/gollemer/internal/ai/neural/nn"
	"github.com/golangast/gollemer/internal/ai/neural/tensor"
	"github.com/golangast/gollemer/internal/ai/neural/tokenizer"
)

// LayersConfig selects the active layers and fixes the derivation
// constants. It is stored in the layers gob so a loaded model
// knows exactly which layers it was trained with.
type LayersConfig struct {
	Char    bool
	Class   bool
	Intent  bool
	Tone    bool
	Concept bool
	Copy    bool
	Turn    bool
	Memory  bool

	NumClasses   int // word-class count (class 0 = specials)
	NumIntent    int // intent classes (0 = other)
	NumTone      int // tone classes (0 = neutral)
	NumConcepts  int // gist clusters; code 0 = EMPTY segment
	ConceptLen   int // concept codes per reply (segments)
	CharBuckets  int // char-trigram hash buckets
	CharDim      int // char embedding width (= encoder embedding dim)
	EmbedDim     int
	HiddenDim    int
	IntentWeight float32 // aux loss weights (DESIGN §3)
	ToneWeight   float32
	ConceptWt    float32
}

// DefaultLayersConfig returns the registered constants (DESIGN §3).
// The On switches are set by the caller.
func DefaultLayersConfig() LayersConfig {
	return LayersConfig{
		NumClasses:   48,
		NumIntent:    7,
		NumTone:      4,
		NumConcepts:  32,
		ConceptLen:   3,
		CharBuckets:  1024,
		CharDim:      128, // = EmbedDim: char vectors add into token embeddings
		EmbedDim:     128,
		HiddenDim:    256,
		IntentWeight: 0.2,
		ToneWeight:   0.2,
		ConceptWt:    0.5,
	}
}

// On returns the layer switch by name ("char", "class", ...).
func (c LayersConfig) On(name string) bool {
	switch name {
	case "char":
		return c.Char
	case "class":
		return c.Class
	case "intent":
		return c.Intent
	case "tone":
		return c.Tone
	case "concept":
		return c.Concept
	case "copy":
		return c.Copy
	case "turn":
		return c.Turn
	case "memory":
		return c.Memory
	}
	return false
}

// Set switches one layer by name; unknown names are an error.
func (c *LayersConfig) Set(name string, on bool) error {
	switch name {
	case "char":
		c.Char = on
	case "class":
		c.Class = on
	case "intent":
		c.Intent = on
	case "tone":
		c.Tone = on
	case "concept":
		c.Concept = on
	case "copy":
		c.Copy = on
	case "turn":
		c.Turn = on
	case "memory":
		c.Memory = on
	default:
		return fmt.Errorf("unknown layer %q", name)
	}
	return nil
}

// LayersModel is the production Seq2Seq plus the layer parameters.
// Fields are exported for gob. Parameters that belong to a layer
// are nil when that layer is off.
type LayersModel struct {
	Base   *Seq2Seq
	Config LayersConfig

	// char: trigram table (manual forward/backward, train_layers.go)
	CharTable *nn.Embedding
	// class: word-class table + the derived class per token id
	ClassTable *nn.Embedding
	ClassOf    []int
	// intent / tone: seed embeddings + prediction heads
	IntentEmb  *nn.Embedding
	IntentHead *nn.Linear
	ToneEmb    *nn.Embedding
	ToneHead   *nn.Linear
	// concept: code embeddings (NumConcepts+1 rows; row 0 = EMPTY),
	// prediction head, and the derivation centroids (kept so the
	// turn layer can name a generated reply's codes at inference)
	ConceptEmb       *nn.Embedding
	ConceptHead      *nn.Linear
	ConceptCentroids [][]float32
	// turn: seed-side table indexed by the concept codes (the
	// prefix table is EmbedDim wide for decoder inputs; the seed
	// slot is HiddenDim wide, so the turn layer reads the codes
	// through its own table — DESIGN §11) plus the learned
	// no-previous-turn vector
	TurnEmb  *nn.Embedding
	NoneTurn *tensor.Tensor
}

// NewLayersModel builds the layer parameters around a production
// Seq2Seq. classOf and centroids come from the corpus derivations
// (chat package) and are required when class/concept are on.
func NewLayersModel(base *Seq2Seq, cfg LayersConfig, classOf []int, centroids [][]float32) (*LayersModel, error) {
	if cfg.CharDim != cfg.EmbedDim {
		// The char vector adds into token embeddings, so its width
		// is not a free choice.
		cfg.CharDim = cfg.EmbedDim
	}
	m := &LayersModel{Base: base, Config: cfg}
	if cfg.Char {
		m.CharTable = nn.NewEmbedding(cfg.CharBuckets, cfg.CharDim)
	}
	if cfg.Class {
		if len(classOf) != base.OutputVocab.Size() {
			return nil, fmt.Errorf("layers: class map has %d entries, vocab has %d", len(classOf), base.OutputVocab.Size())
		}
		m.ClassTable = nn.NewEmbedding(cfg.NumClasses, cfg.EmbedDim)
		m.ClassOf = classOf
	}
	if cfg.Intent {
		m.IntentEmb = nn.NewEmbedding(cfg.NumIntent, cfg.HiddenDim)
		head, err := nn.NewLinear(cfg.HiddenDim, cfg.NumIntent)
		if err != nil {
			return nil, err
		}
		m.IntentHead = head
	}
	if cfg.Tone {
		m.ToneEmb = nn.NewEmbedding(cfg.NumTone, cfg.HiddenDim)
		head, err := nn.NewLinear(cfg.HiddenDim, cfg.NumTone)
		if err != nil {
			return nil, err
		}
		m.ToneHead = head
	}
	if cfg.Concept {
		if len(centroids) != cfg.NumConcepts {
			return nil, fmt.Errorf("layers: %d concept centroids, want %d", len(centroids), cfg.NumConcepts)
		}
		m.ConceptEmb = nn.NewEmbedding(cfg.NumConcepts+1, cfg.EmbedDim)
		head, err := nn.NewLinear(cfg.HiddenDim, cfg.ConceptLen*(cfg.NumConcepts+1))
		if err != nil {
			return nil, err
		}
		m.ConceptHead = head
		m.ConceptCentroids = centroids
	}
	if cfg.Turn {
		if !cfg.Concept {
			return nil, fmt.Errorf("layers: turn requires concept")
		}
		m.TurnEmb = nn.NewEmbedding(cfg.NumConcepts+1, cfg.HiddenDim)
		data := make([]float32, cfg.HiddenDim)
		for i := range data {
			data[i] = (randFloat32()*2 - 1) * 0.1
		}
		m.NoneTurn = tensor.NewTensor([]int{cfg.HiddenDim}, data, true)
	}
	if cfg.Copy {
		base.Decoder.PtrGen = NewPointerGen(cfg.HiddenDim)
	}
	m.ensureManualGrads()
	return m, nil
}

// randFloat32 is a tiny local source so layer init does not depend
// on any package-level seeding choices elsewhere.
var randState uint64 = 20261112

func randFloat32() float32 {
	// xorshift64*
	randState ^= randState >> 12
	randState ^= randState << 25
	randState ^= randState >> 27
	return float32((randState*2685821657736338717)>>40) / float32(1<<24)
}

// manualParams are the parameters whose gradients are scattered by
// hand in TrainBatchLayers (everything except the base model's own
// layers and the nn.Linear heads, whose Backward methods run).
func (m *LayersModel) manualParams() []*tensor.Tensor {
	var out []*tensor.Tensor
	if m.CharTable != nil {
		out = append(out, m.CharTable.Weight)
	}
	if m.ClassTable != nil {
		out = append(out, m.ClassTable.Weight)
	}
	if m.IntentEmb != nil {
		out = append(out, m.IntentEmb.Weight)
	}
	if m.ToneEmb != nil {
		out = append(out, m.ToneEmb.Weight)
	}
	if m.ConceptEmb != nil {
		out = append(out, m.ConceptEmb.Weight)
	}
	if m.TurnEmb != nil {
		out = append(out, m.TurnEmb.Weight)
	}
	if m.NoneTurn != nil {
		out = append(out, m.NoneTurn)
	}
	return out
}

// ensureManualGrads pre-creates zero gradient tensors for the
// hand-scattered parameters so the optimizer always sees a Grad.
func (m *LayersModel) ensureManualGrads() {
	for _, p := range m.manualParams() {
		if p.Grad == nil {
			p.Grad = tensor.NewTensor(p.Shape, make([]float32, len(p.Data)), false)
		}
	}
}

// Parameters returns every trainable parameter: the base model's,
// the heads', the copy head's (via Decoder.Parameters), and the
// hand-scattered tables'.
func (m *LayersModel) Parameters() []*tensor.Tensor {
	params := m.Base.Parameters()
	if m.IntentHead != nil {
		params = append(params, m.IntentHead.Parameters()...)
	}
	if m.ToneHead != nil {
		params = append(params, m.ToneHead.Parameters()...)
	}
	if m.ConceptHead != nil {
		params = append(params, m.ConceptHead.Parameters()...)
	}
	params = append(params, m.manualParams()...)
	return params
}

// CharBucketsFor returns the hashed trigram bucket ids of a surface
// word: trigrams of "^"+word+"$", FNV-1a, mod buckets. Every word
// yields at least one gram (the padded string is >= 3 runes for any
// non-empty word; a 1-rune word yields exactly its padded form).
func CharBucketsFor(word string, buckets int) []int {
	runes := []rune("^" + word + "$")
	if len(runes) < 3 {
		return []int{int(hashString(word) % uint32(buckets))}
	}
	seen := map[int]bool{}
	var out []int
	for i := 0; i+3 <= len(runes); i++ {
		g := string(runes[i : i+3])
		b := int(hashString(g) % uint32(buckets))
		if !seen[b] {
			seen[b] = true
			out = append(out, b)
		}
	}
	if len(out) == 0 {
		return []int{int(hashString(word) % uint32(buckets))}
	}
	return out
}

func hashString(s string) uint32 {
	h := fnv.New32a()
	_, _ = h.Write([]byte(s))
	return h.Sum32()
}

// charVectorInto writes the mean trigram vector of surface into
// dst (length CharDim). Special-token surfaces ("", "<unk>", ...)
// and empty surfaces yield zeros: specials get no char signal.
func (m *LayersModel) charVectorInto(surface string, special bool, dst []float32) {
	for i := range dst {
		dst[i] = 0
	}
	if m.CharTable == nil || special || surface == "" {
		return
	}
	bs := CharBucketsFor(surface, m.Config.CharBuckets)
	dim := m.Config.CharDim
	for _, b := range bs {
		off := b * dim
		for k := 0; k < dim; k++ {
			dst[k] += m.CharTable.Weight.Data[off+k]
		}
	}
	inv := float32(1) / float32(len(bs))
	for k := range dst {
		dst[k] *= inv
	}
}

// isSpecialID reports whether a token id is one of the vocab's
// special tokens (pad/unk/bos/eos): they carry no class or char
// signal beyond their own embedding.
func (m *LayersModel) isSpecialID(id int) bool {
	v := m.Base.OutputVocab
	return id == v.PaddingTokenID || id == v.UnkID || id == v.BosID || id == v.EosID
}

// classOf returns the derived word class of a token id (0 when the
// class layer is off or the id is special).
func (m *LayersModel) classOf(id int) int {
	if m.ClassOf == nil || m.isSpecialID(id) {
		return 0
	}
	if id < 0 || id >= len(m.ClassOf) {
		return 0
	}
	return m.ClassOf[id]
}

// DeriveReplyCodes names a token sequence's concept codes with the
// derivation centroids (DESIGN §3.5): the sequence is split into
// ConceptLen contiguous segments; each segment's L2-normalized
// bag-of-words is assigned its nearest centroid (cosine); an empty
// segment is code 0 (EMPTY). Used at inference for the turn layer
// and in training-data preparation for gold codes.
func (m *LayersModel) DeriveReplyCodes(tokenIDs []int) [3]int {
	return DeriveCodesWithCentroids(tokenIDs, m.ConceptCentroids, m.Config.ConceptLen)
}

// DeriveCodesWithCentroids is the shared segment-coding function
// (the chat derivations call it too, before a model exists).
func DeriveCodesWithCentroids(tokenIDs []int, centroids [][]float32, conceptLen int) [3]int {
	var codes [3]int
	if len(centroids) == 0 {
		return codes
	}
	vocabSize := len(centroids[0])
	segs := SplitSegments(tokenIDs, conceptLen)
	for s := 0; s < conceptLen && s < 3; s++ {
		seg := segs[s]
		if len(seg) == 0 {
			codes[s] = 0
			continue
		}
		vec := make([]float64, vocabSize)
		for _, id := range seg {
			if id >= 0 && id < vocabSize {
				vec[id]++
			}
		}
		var norm float64
		for _, x := range vec {
			norm += x * x
		}
		norm = math.Sqrt(norm)
		if norm == 0 {
			codes[s] = 0
			continue
		}
		best, bestSim := 0, math.Inf(-1)
		for ci, cen := range centroids {
			var dot, cn float64
			for i, cv := range cen {
				dot += vec[i] * float64(cv)
				cn += float64(cv) * float64(cv)
			}
			sim := dot / (norm * math.Sqrt(cn))
			if sim > bestSim {
				bestSim, best = sim, ci
			}
		}
		codes[s] = best + 1 // codes are 1-based; 0 = EMPTY
	}
	return codes
}

// SplitSegments splits ids into n contiguous segments as evenly as
// possible (earlier segments take the remainder).
func SplitSegments(ids []int, n int) [][]int {
	segs := make([][]int, n)
	if n <= 0 {
		return segs
	}
	base := len(ids) / n
	rem := len(ids) % n
	pos := 0
	for s := 0; s < n; s++ {
		sz := base
		if s < rem {
			sz++
		}
		if pos+sz > len(ids) {
			sz = len(ids) - pos
		}
		if sz < 0 {
			sz = 0
		}
		segs[s] = ids[pos : pos+sz]
		pos += sz
	}
	return segs
}

// serializableLayers is the gob form of a LayersModel: the base
// model in its own serializable shape plus the layer state.
type serializableLayers struct {
	Base             *SerializableSeq2Seq
	Config           LayersConfig
	CharTable        *nn.Embedding
	ClassTable       *nn.Embedding
	ClassOf          []int
	IntentEmb        *nn.Embedding
	IntentHead       *nn.Linear
	ToneEmb          *nn.Embedding
	ToneHead         *nn.Linear
	ConceptEmb       *nn.Embedding
	ConceptHead      *nn.Linear
	ConceptCentroids [][]float32
	TurnEmb          *nn.Embedding
	NoneTurn         *tensor.Tensor
}

// Save writes the layers model to filePath (gob). The production
// model's Save is never used for a layers model and vice versa.
func (m *LayersModel) Save(filePath string) error {
	file, err := os.Create(filePath)
	if err != nil {
		return fmt.Errorf("layers save: %w", err)
	}
	defer file.Close()
	s := &serializableLayers{
		Base: &SerializableSeq2Seq{
			Encoder:     m.Base.Encoder,
			Decoder:     m.Base.Decoder,
			OutputVocab: m.Base.OutputVocab,
			HiddenDim:   m.Base.HiddenDim,
			ExactMap:    m.Base.ExactMap,
		},
		Config:           m.Config,
		CharTable:        m.CharTable,
		ClassTable:       m.ClassTable,
		ClassOf:          m.ClassOf,
		IntentEmb:        m.IntentEmb,
		IntentHead:       m.IntentHead,
		ToneEmb:          m.ToneEmb,
		ToneHead:         m.ToneHead,
		ConceptEmb:       m.ConceptEmb,
		ConceptHead:      m.ConceptHead,
		ConceptCentroids: m.ConceptCentroids,
		TurnEmb:          m.TurnEmb,
		NoneTurn:         m.NoneTurn,
	}
	if err := gob.NewEncoder(file).Encode(s); err != nil {
		return fmt.Errorf("layers save encode: %w", err)
	}
	return nil
}

// LoadLayers reads a layers model written by Save.
func LoadLayers(filePath string, tok *tokenizer.Tokenizer) (*LayersModel, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("layers load: %w", err)
	}
	defer file.Close()
	var s serializableLayers
	if err := gob.NewDecoder(file).Decode(&s); err != nil {
		return nil, fmt.Errorf("layers load decode: %w", err)
	}
	base := &Seq2Seq{
		Encoder:     s.Base.Encoder,
		Decoder:     s.Base.Decoder,
		Tokenizer:   tok,
		OutputVocab: s.Base.OutputVocab,
		HiddenDim:   s.Base.HiddenDim,
		ExactMap:    s.Base.ExactMap,
	}
	m := &LayersModel{
		Base:             base,
		Config:           s.Config,
		CharTable:        s.CharTable,
		ClassTable:       s.ClassTable,
		ClassOf:          s.ClassOf,
		IntentEmb:        s.IntentEmb,
		IntentHead:       s.IntentHead,
		ToneEmb:          s.ToneEmb,
		ToneHead:         s.ToneHead,
		ConceptEmb:       s.ConceptEmb,
		ConceptHead:      s.ConceptHead,
		ConceptCentroids: s.ConceptCentroids,
		TurnEmb:          s.TurnEmb,
		NoneTurn:         s.NoneTurn,
	}
	m.ensureManualGrads()
	return m, nil
}

// TrainedLayers returns the names of the layers this model was
// trained with, in shelf order (for scorer reporting).
func (m *LayersModel) TrainedLayers() []string {
	var out []string
	for _, name := range []string{"char", "class", "intent", "tone", "concept", "copy", "turn", "memory"} {
		if m.Config.On(name) {
			out = append(out, name)
		}
	}
	return out
}
