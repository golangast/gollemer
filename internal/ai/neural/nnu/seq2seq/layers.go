package seq2seq

// Layers pilot (2026-11-12): generic intermediate layers for the
// social seq2seq, each with one narrow job, each entering as an
// added feature or a conditioning signal — never as a channel the
// reply must pass through. The design, bars, and ablation schedule
// are pre-registered in ~/workspace/layers-pilot-scratch/DESIGN.md.
//
// AMENDMENT (2026-11-12, DESIGN §12): the original flat CONCEPT
// layer (3 segment gist codes) plus the standalone INTENT and TONE
// layers are replaced by John's hybrid FRAME: one coarse TOP CODE
// for the reply's gist, a SLOTTED FRAME under it (ACT and TONE
// slots — the intent/tone derivations absorbed as the frame's
// slots — and a TOPIC slot carried from the input by construction),
// and a deterministic RE-CORRECTION LOOP at inference that checks
// the filled frame for consistency and re-picks offending slots.
// The frame is the organizing structure; char, class, copy, turn,
// and memory are otherwise unchanged, except TURN now consumes the
// previous reply's TOP CODE instead of the old segment codes.
//
// LayersModel wraps the production Seq2Seq untouched: with every
// layer off it is not constructed at all (the layers trainer is a
// separate entry point), and the production trainer/chat paths
// never see this type. All layer parameters are trained jointly
// with the word model by TrainBatchLayers (train_layers.go).
//
// Layer inventory (DESIGN §3 as amended by §12):
//   char    — mean char-trigram embedding added to token embeddings
//   class   — word-class embedding added to token embeddings
//   frame   — top code (prefix) + act/tone slots (seed), with the
//             repair loop at inference; variants frame-top (top
//             code only) and frame-slots (slots only) are training
//             arms selected through the same layer list
//   copy    — the house PointerGen over encoder input positions
//   turn    — previous reply's top code, seed conditioning
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
// knows exactly which layers it was trained with. All fields are
// scalars: the trainer compares configs with != on resume.
type LayersConfig struct {
	Char   bool
	Class  bool
	Frame  bool
	Copy   bool
	Turn   bool
	Memory bool

	// Frame variants (training arms): FrameTopOnly trains the top
	// code with no act/tone slots; FrameSlotsOnly trains the slots
	// with no top code (their heads then read the encoder state
	// alone). Both false = the full hybrid frame.
	FrameTopOnly   bool
	FrameSlotsOnly bool

	// FrameSoft is the concept-v2 variant (CONCEPT_V2_DESIGN §3):
	// the top code conditions the decoder as the head's
	// softmax-weighted BLEND of code embeddings — the prefix step
	// and the slot heads' top conditioning both read the
	// distribution, never an argmax row — and the word/slot
	// losses backpropagate through the blend into the head.
	// Requires the top code; with it off, every frame path is
	// the v1 argmax behavior, byte-identical.
	FrameSoft bool

	NumClasses  int // word-class count (class 0 = specials)
	NumAct      int // act slot values (intent derivation classes)
	NumTone     int // tone slot values
	NumTopCodes int // top-code clusters (DESIGN §12)
	CharBuckets int // char-trigram hash buckets
	CharDim     int // char embedding width (= encoder embedding dim)
	EmbedDim    int
	HiddenDim   int
	TopWeight   float32 // aux loss weights (§12: top heaviest,
	ActWeight   float32 // slots at the house intent/tone weight)
	ToneWeight  float32
}

// DefaultLayersConfig returns the registered constants (DESIGN §3
// as amended by §12).
func DefaultLayersConfig() LayersConfig {
	return LayersConfig{
		NumClasses:  48,
		NumAct:      7,
		NumTone:     4,
		NumTopCodes: 16,
		CharBuckets: 1024,
		CharDim:     128, // = EmbedDim: char vectors add into token embeddings
		EmbedDim:    128,
		HiddenDim:   256,
		TopWeight:   0.5,
		ActWeight:   0.2,
		ToneWeight:  0.2,
	}
}

// HasTop reports whether this configuration trains the frame's
// top code (full frame or the top-only arm).
func (c LayersConfig) HasTop() bool { return c.Frame && !c.FrameSlotsOnly }

// HasSlots reports whether this configuration trains the frame's
// act/tone slots (full frame or the slots-only arm).
func (c LayersConfig) HasSlots() bool { return c.Frame && !c.FrameTopOnly }

// On returns the layer switch by name ("char", "class", ...).
// "intent", "tone", and "concept" are pre-amendment names and are
// gone: the frame absorbed them (DESIGN §12).
func (c LayersConfig) On(name string) bool {
	switch name {
	case "char":
		return c.Char
	case "class":
		return c.Class
	case "frame":
		return c.Frame
	case "copy":
		return c.Copy
	case "turn":
		return c.Turn
	case "memory":
		return c.Memory
	}
	return false
}

// Set switches one layer by name; unknown names are an error. The
// frame variants are selected with "frame-top" / "frame-slots";
// the concept-v2 soft frame with "frame-soft".
func (c *LayersConfig) Set(name string, on bool) error {
	switch name {
	case "char":
		c.Char = on
	case "class":
		c.Class = on
	case "frame":
		c.Frame = on
	case "frame-top":
		c.Frame, c.FrameTopOnly = on, on
	case "frame-slots":
		c.Frame, c.FrameSlotsOnly = on, on
	case "frame-soft":
		c.Frame, c.FrameSoft = on, on
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

// ToneLexicon is the derived tone vocabulary (DESIGN §3.4). It
// travels inside the layers gob: the repair loop's tone check
// re-reads the input's tone with it at inference, so the scorer
// needs it without side files.
type ToneLexicon struct {
	Bright  []string `json:"bright"`
	Low     []string `json:"low"`
	Worried []string `json:"worried"`
}

// ToneOf applies the registered tone rule to a word sequence: any
// worried hit labels worried, else any bright hit labels bright,
// else any low hit labels low; no hit is neutral (0). Class ids:
// 0 neutral, 1 bright, 2 low, 3 worried. This is the single
// implementation of the rule — the chat derivations label training
// pairs with it and the repair loop re-reads inputs with it.
func (l ToneLexicon) ToneOf(words []string) int {
	bright, low := false, false
	in := func(list []string, w string) bool {
		for _, x := range list {
			if x == w {
				return true
			}
		}
		return false
	}
	for _, w := range words {
		switch {
		case in(l.Worried, w):
			return 3
		case in(l.Bright, w):
			bright = true
		case in(l.Low, w):
			low = true
		}
	}
	if bright {
		return 1
	}
	if low {
		return 2
	}
	return 0
}

// LayersModel is the production Seq2Seq plus the layer parameters.
// Fields are exported for gob. Parameters that belong to a layer
// are nil when that layer (or frame part) is off.
type LayersModel struct {
	Base   *Seq2Seq
	Config LayersConfig

	// char: trigram table (manual forward/backward, train_layers.go)
	CharTable *nn.Embedding
	// class: word-class table + the derived class per token id
	ClassTable *nn.Embedding
	ClassOf    []int
	// frame — top code: prefix table (EmbedDim wide, fed as one
	// decoder input step), seed-side table (HiddenDim wide, added
	// to the encoder state before the act/tone heads so slot
	// prediction is conditioned on the top code), prediction head,
	// and the derivation centroids (kept so any reply's top code
	// can be named at inference: turn threading + round-trip check)
	TopEmb       *nn.Embedding
	TopSeedEmb   *nn.Embedding
	TopHead      *nn.Linear
	TopCentroids [][]float32
	// frame — slots: act/tone seed embeddings + prediction heads,
	// the act-family table the repair loop checks against
	// (ActFamily[top][act], derived from training frames), and the
	// tone lexicon the tone check re-reads with
	ActEmb    *nn.Embedding
	ActHead   *nn.Linear
	ToneEmb   *nn.Embedding
	ToneHead  *nn.Linear
	ActFamily [][]bool
	ToneLex   ToneLexicon
	// turn: seed-side table indexed by the previous reply's top
	// code plus the learned no-previous-turn vector
	TurnEmb  *nn.Embedding
	NoneTurn *tensor.Tensor
}

// LayersDerivations carries the corpus derivations a layers model
// needs at construction (produced by the chat package, DESIGN §4).
type LayersDerivations struct {
	ClassOf      []int       // per token id (len = vocab size)
	TopCentroids [][]float32 // NumTopCodes x vocab size
	ActFamily    [][]bool    // NumTopCodes x NumAct
	ToneLex      ToneLexicon
}

// NewLayersModel builds the layer parameters around a production
// Seq2Seq. The derivation fields are required exactly when the
// layers that consume them are on.
func NewLayersModel(base *Seq2Seq, cfg LayersConfig, d LayersDerivations) (*LayersModel, error) {
	if cfg.CharDim != cfg.EmbedDim {
		// The char vector adds into token embeddings, so its width
		// is not a free choice.
		cfg.CharDim = cfg.EmbedDim
	}
	if cfg.FrameTopOnly && cfg.FrameSlotsOnly {
		return nil, fmt.Errorf("layers: frame cannot be both top-only and slots-only")
	}
	if cfg.FrameSoft && !cfg.HasTop() {
		return nil, fmt.Errorf("layers: frame-soft requires the top code (it cannot combine with frame-slots)")
	}
	m := &LayersModel{Base: base, Config: cfg}
	if cfg.Char {
		m.CharTable = nn.NewEmbedding(cfg.CharBuckets, cfg.CharDim)
	}
	if cfg.Class {
		if len(d.ClassOf) != base.OutputVocab.Size() {
			return nil, fmt.Errorf("layers: class map has %d entries, vocab has %d", len(d.ClassOf), base.OutputVocab.Size())
		}
		m.ClassTable = nn.NewEmbedding(cfg.NumClasses, cfg.EmbedDim)
		m.ClassOf = d.ClassOf
	}
	if cfg.HasTop() {
		if len(d.TopCentroids) != cfg.NumTopCodes {
			return nil, fmt.Errorf("layers: %d top centroids, want %d", len(d.TopCentroids), cfg.NumTopCodes)
		}
		m.TopEmb = nn.NewEmbedding(cfg.NumTopCodes, cfg.EmbedDim)
		m.TopSeedEmb = nn.NewEmbedding(cfg.NumTopCodes, cfg.HiddenDim)
		head, err := nn.NewLinear(cfg.HiddenDim, cfg.NumTopCodes)
		if err != nil {
			return nil, err
		}
		m.TopHead = head
		m.TopCentroids = d.TopCentroids
	}
	if cfg.HasSlots() {
		m.ActEmb = nn.NewEmbedding(cfg.NumAct, cfg.HiddenDim)
		actHead, err := nn.NewLinear(cfg.HiddenDim, cfg.NumAct)
		if err != nil {
			return nil, err
		}
		m.ActHead = actHead
		m.ToneEmb = nn.NewEmbedding(cfg.NumTone, cfg.HiddenDim)
		toneHead, err := nn.NewLinear(cfg.HiddenDim, cfg.NumTone)
		if err != nil {
			return nil, err
		}
		m.ToneHead = toneHead
		m.ToneLex = d.ToneLex
		if cfg.HasTop() {
			if len(d.ActFamily) != cfg.NumTopCodes {
				return nil, fmt.Errorf("layers: act family has %d rows, want %d top codes", len(d.ActFamily), cfg.NumTopCodes)
			}
			m.ActFamily = d.ActFamily
		}
	}
	if cfg.Turn {
		if !cfg.HasTop() {
			return nil, fmt.Errorf("layers: turn requires the frame's top code")
		}
		m.TurnEmb = nn.NewEmbedding(cfg.NumTopCodes, cfg.HiddenDim)
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
	if m.TopEmb != nil {
		out = append(out, m.TopEmb.Weight)
	}
	if m.TopSeedEmb != nil {
		out = append(out, m.TopSeedEmb.Weight)
	}
	if m.ActEmb != nil {
		out = append(out, m.ActEmb.Weight)
	}
	if m.ToneEmb != nil {
		out = append(out, m.ToneEmb.Weight)
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
	if m.TopHead != nil {
		params = append(params, m.TopHead.Parameters()...)
	}
	if m.ActHead != nil {
		params = append(params, m.ActHead.Parameters()...)
	}
	if m.ToneHead != nil {
		params = append(params, m.ToneHead.Parameters()...)
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

// DeriveReplyTop names a token sequence's TOP CODE with the
// derivation centroids (DESIGN §12): the sequence's L2-normalized
// bag-of-words is assigned its nearest centroid (cosine). The
// second result is false for an empty (or all-out-of-range)
// sequence, which has no top code. Used for teacher frames (chat
// derivations), turn threading, and the round-trip check.
func (m *LayersModel) DeriveReplyTop(tokenIDs []int) (int, bool) {
	return DeriveTopWithCentroids(tokenIDs, m.TopCentroids)
}

// DeriveTopWithCentroids is the shared top-coding function (the
// chat derivations call it too, before a model exists).
func DeriveTopWithCentroids(tokenIDs []int, centroids [][]float32) (int, bool) {
	if len(centroids) == 0 || len(tokenIDs) == 0 {
		return 0, false
	}
	vocabSize := len(centroids[0])
	vec := make([]float64, vocabSize)
	var used int
	for _, id := range tokenIDs {
		if id >= 0 && id < vocabSize {
			vec[id]++
			used++
		}
	}
	if used == 0 {
		return 0, false
	}
	var norm float64
	for _, x := range vec {
		norm += x * x
	}
	norm = math.Sqrt(norm)
	best, bestSim := 0, math.Inf(-1)
	for ci, cen := range centroids {
		var dot, cn float64
		for i, cv := range cen {
			dot += vec[i] * float64(cv)
			cn += float64(cv) * float64(cv)
		}
		if cn == 0 {
			continue
		}
		if sim := dot / (norm * math.Sqrt(cn)); sim > bestSim {
			bestSim, best = sim, ci
		}
	}
	return best, true
}

// serializableLayers is the gob form of a LayersModel: the base
// model in its own serializable shape plus the layer state.
type serializableLayers struct {
	Base         *SerializableSeq2Seq
	Config       LayersConfig
	CharTable    *nn.Embedding
	ClassTable   *nn.Embedding
	ClassOf      []int
	TopEmb       *nn.Embedding
	TopSeedEmb   *nn.Embedding
	TopHead      *nn.Linear
	TopCentroids [][]float32
	ActEmb       *nn.Embedding
	ActHead      *nn.Linear
	ToneEmb      *nn.Embedding
	ToneHead     *nn.Linear
	ActFamily    [][]bool
	ToneLex      ToneLexicon
	TurnEmb      *nn.Embedding
	NoneTurn     *tensor.Tensor
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
		Config:       m.Config,
		CharTable:    m.CharTable,
		ClassTable:   m.ClassTable,
		ClassOf:      m.ClassOf,
		TopEmb:       m.TopEmb,
		TopSeedEmb:   m.TopSeedEmb,
		TopHead:      m.TopHead,
		TopCentroids: m.TopCentroids,
		ActEmb:       m.ActEmb,
		ActHead:      m.ActHead,
		ToneEmb:      m.ToneEmb,
		ToneHead:     m.ToneHead,
		ActFamily:    m.ActFamily,
		ToneLex:      m.ToneLex,
		TurnEmb:      m.TurnEmb,
		NoneTurn:     m.NoneTurn,
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
		Base:         base,
		Config:       s.Config,
		CharTable:    s.CharTable,
		ClassTable:   s.ClassTable,
		ClassOf:      s.ClassOf,
		TopEmb:       s.TopEmb,
		TopSeedEmb:   s.TopSeedEmb,
		TopHead:      s.TopHead,
		TopCentroids: s.TopCentroids,
		ActEmb:       s.ActEmb,
		ActHead:      s.ActHead,
		ToneEmb:      s.ToneEmb,
		ToneHead:     s.ToneHead,
		ActFamily:    s.ActFamily,
		ToneLex:      s.ToneLex,
		TurnEmb:      s.TurnEmb,
		NoneTurn:     s.NoneTurn,
	}
	m.ensureManualGrads()
	return m, nil
}

// TrainedLayers returns the names of the layers this model was
// trained with, in shelf order (for scorer reporting). A frame
// variant is named as it was selected ("frame-top"/"frame-slots";
// a soft full frame reports "frame-soft").
func (m *LayersModel) TrainedLayers() []string {
	var out []string
	for _, name := range []string{"char", "class", "frame", "frame-soft", "frame-top", "frame-slots", "copy", "turn", "memory"} {
		switch name {
		case "frame":
			if m.Config.Frame && !m.Config.FrameSoft && !m.Config.FrameTopOnly && !m.Config.FrameSlotsOnly {
				out = append(out, name)
			}
		case "frame-soft":
			if m.Config.FrameSoft && !m.Config.FrameTopOnly && !m.Config.FrameSlotsOnly {
				out = append(out, name)
			}
		case "frame-top":
			if m.Config.FrameTopOnly {
				out = append(out, name)
			}
		case "frame-slots":
			if m.Config.FrameSlotsOnly {
				out = append(out, name)
			}
		default:
			if m.Config.On(name) {
				out = append(out, name)
			}
		}
	}
	return out
}
