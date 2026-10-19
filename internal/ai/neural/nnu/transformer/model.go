package transformer

import (
	"encoding/gob"
	"fmt"
	"os"

	"github.com/golangast/gollemer/internal/ai/neural/nn"
	"github.com/golangast/gollemer/internal/ai/neural/tensor"
)

// TransformerConfig holds hyperparameters.
type TransformerConfig struct {
	DModel    int // Model dimension (512)
	NumHeads  int // Attention heads (8)
	NumLayers int // Encoder/decoder layers (6)
	DFF       int // Feed-forward hidden (2048)
	MaxLen    int // Max sequence length (512)
	VocabSize int // Vocabulary size
}

// DefaultConfig returns a standard Transformer config.
func DefaultConfig(vocabSize int) TransformerConfig {
	return TransformerConfig{
		DModel:    512,
		NumHeads:  8,
		NumLayers: 6,
		DFF:       2048,
		MaxLen:    512,
		VocabSize: vocabSize,
	}
}

// Transformer is a full seq2seq Transformer model.
type Transformer struct {
	Config TransformerConfig

	SrcEmbedding *nn.Embedding
	TgtEmbedding *nn.Embedding
	PosEncoding  *PositionalEncoding

	EncoderLayers []*TransformerEncoderLayer
	DecoderLayers []*TransformerDecoderLayer

	OutputProj *nn.Linear // d_model -> vocab_size
}

// NewTransformer creates a Transformer model.
func NewTransformer(cfg TransformerConfig) (*Transformer, error) {
	srcEmb := nn.NewEmbedding(cfg.VocabSize, cfg.DModel)
	tgtEmb := nn.NewEmbedding(cfg.VocabSize, cfg.DModel)
	posEnc := NewPositionalEncoding(cfg.DModel, cfg.MaxLen)

	encLayers := make([]*TransformerEncoderLayer, cfg.NumLayers)
	for i := range encLayers {
		l, err := NewTransformerEncoderLayer(cfg.DModel, cfg.NumHeads, cfg.DFF)
		if err != nil {
			return nil, fmt.Errorf("encoder layer %d: %w", i, err)
		}
		encLayers[i] = l
	}

	decLayers := make([]*TransformerDecoderLayer, cfg.NumLayers)
	for i := range decLayers {
		l, err := NewTransformerDecoderLayer(cfg.DModel, cfg.NumHeads, cfg.DFF)
		if err != nil {
			return nil, fmt.Errorf("decoder layer %d: %w", i, err)
		}
		decLayers[i] = l
	}

	outProj, err := nn.NewLinear(cfg.DModel, cfg.VocabSize)
	if err != nil {
		return nil, fmt.Errorf("output proj: %w", err)
	}

	return &Transformer{
		Config:       cfg,
		SrcEmbedding: srcEmb,
		TgtEmbedding: tgtEmb,
		PosEncoding:  posEnc,
		EncoderLayers: encLayers,
		DecoderLayers: decLayers,
		OutputProj:   outProj,
	}, nil
}

// Encode runs the encoder: src [batch, src_seq] -> memory [batch, src_seq, d_model]
func (t *Transformer) Encode(src *tensor.Tensor) (*tensor.Tensor, error) {
	// Embed + positional encoding
	embedded, err := t.SrcEmbedding.Forward(src)
	if err != nil {
		return nil, fmt.Errorf("src embed: %w", err)
	}
	x, err := t.PosEncoding.Forward(embedded)
	if err != nil {
		return nil, fmt.Errorf("pos enc: %w", err)
	}

	// Encoder layers
	for i, layer := range t.EncoderLayers {
		x, err = layer.Forward(x)
		if err != nil {
			return nil, fmt.Errorf("encoder layer %d: %w", i, err)
		}
	}
	return x, nil
}

// Decode runs the decoder: tgt [batch, tgt_seq], memory [batch, src_seq, d_model]
// Returns logits [batch, tgt_seq, vocab_size]
func (t *Transformer) Decode(tgt *tensor.Tensor, memory *tensor.Tensor) (*tensor.Tensor, error) {
	// Embed + positional encoding
	embedded, err := t.TgtEmbedding.Forward(tgt)
	if err != nil {
		return nil, fmt.Errorf("tgt embed: %w", err)
	}
	x, err := t.PosEncoding.Forward(embedded)
	if err != nil {
		return nil, fmt.Errorf("pos enc: %w", err)
	}

	// Decoder layers
	for i, layer := range t.DecoderLayers {
		x, err = layer.Forward(x, memory)
		if err != nil {
			return nil, fmt.Errorf("decoder layer %d: %w", i, err)
		}
	}

	// Output projection
	return t.OutputProj.Forward(x)
}

// Forward runs the full model: src [batch, src_seq], tgt [batch, tgt_seq]
// Returns logits [batch, tgt_seq, vocab_size]
func (t *Transformer) Forward(src, tgt *tensor.Tensor) (*tensor.Tensor, error) {
	memory, err := t.Encode(src)
	if err != nil {
		return nil, fmt.Errorf("encode: %w", err)
	}
	return t.Decode(tgt, memory)
}

// NumParams estimates the parameter count.
func (t *Transformer) NumParams() int {
	cfg := t.Config
	// Embeddings: 2 * vocab * d_model
	params := 2 * cfg.VocabSize * cfg.DModel
	// Each encoder layer: 4*d_model^2 (attn) + 2*d_model*d_ff (ffn) + norms
	encLayer := 4*cfg.DModel*cfg.DModel + 2*cfg.DModel*cfg.DFF + 4*cfg.DModel
	// Each decoder layer: 8*d_model^2 (2 attn) + 2*d_model*d_ff + norms
	decLayer := 8*cfg.DModel*cfg.DModel + 2*cfg.DModel*cfg.DFF + 6*cfg.DModel
	params += cfg.NumLayers * (encLayer + decLayer)
	// Output projection
	params += cfg.DModel * cfg.VocabSize + cfg.VocabSize
	return params
}

// Parameters collects all trainable tensors for the optimizer.
func (t *Transformer) Parameters() []*tensor.Tensor {
	var params []*tensor.Tensor
	params = append(params, t.SrcEmbedding.Parameters()...)
	params = append(params, t.TgtEmbedding.Parameters()...)
	for _, l := range t.EncoderLayers {
		params = append(params, l.SelfAttn.WQ.Parameters()...)
		params = append(params, l.SelfAttn.WK.Parameters()...)
		params = append(params, l.SelfAttn.WV.Parameters()...)
		params = append(params, l.SelfAttn.WO.Parameters()...)
		params = append(params, l.Norm1.Parameters()...)
		params = append(params, l.Norm2.Parameters()...)
		params = append(params, l.FF1.Parameters()...)
		params = append(params, l.FF2.Parameters()...)
	}
	for _, l := range t.DecoderLayers {
		params = append(params, l.SelfAttn.WQ.Parameters()...)
		params = append(params, l.SelfAttn.WK.Parameters()...)
		params = append(params, l.SelfAttn.WV.Parameters()...)
		params = append(params, l.SelfAttn.WO.Parameters()...)
		params = append(params, l.CrossAttn.WQ.Parameters()...)
		params = append(params, l.CrossAttn.WK.Parameters()...)
		params = append(params, l.CrossAttn.WV.Parameters()...)
		params = append(params, l.CrossAttn.WO.Parameters()...)
		params = append(params, l.Norm1.Parameters()...)
		params = append(params, l.Norm2.Parameters()...)
		params = append(params, l.Norm3.Parameters()...)
		params = append(params, l.FF1.Parameters()...)
		params = append(params, l.FF2.Parameters()...)
	}
	params = append(params, t.OutputProj.Parameters()...)
	return params
}

// SerializableTransformer is the gob-encodable form for checkpoints.
type SerializableTransformer struct {
	Config TransformerConfig
	Shapes [][]int
	Datas  [][]float32
}

// Save writes the model to a file via gob.
func (t *Transformer) Save(filePath string) error {
	params := t.Parameters()
	s := &SerializableTransformer{
		Config: t.Config,
		Shapes: make([][]int, len(params)),
		Datas:  make([][]float32, len(params)),
	}
	for i, p := range params {
		s.Shapes[i] = append([]int(nil), p.Shape...)
		s.Datas[i] = append([]float32(nil), p.Data...)
	}
	file, err := os.Create(filePath)
	if err != nil {
		return fmt.Errorf("create: %w", err)
	}
	defer file.Close()
	if err := gob.NewEncoder(file).Encode(s); err != nil {
		return fmt.Errorf("encode: %w", err)
	}
	return nil
}

// LoadTransformer loads a Transformer from a gob checkpoint file.
func LoadTransformer(filePath string) (*Transformer, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("open: %w", err)
	}
	defer file.Close()
	var s SerializableTransformer
	if err := gob.NewDecoder(file).Decode(&s); err != nil {
		return nil, fmt.Errorf("decode: %w", err)
	}
	model, err := NewTransformer(s.Config)
	if err != nil {
		return nil, fmt.Errorf("new: %w", err)
	}
	params := model.Parameters()
	if len(params) != len(s.Datas) {
		return nil, fmt.Errorf("param count mismatch: %d vs %d", len(params), len(s.Datas))
	}
	for i, p := range params {
		if len(p.Data) != len(s.Datas[i]) {
			return nil, fmt.Errorf("param %d size mismatch", i)
		}
		copy(p.Data, s.Datas[i])
	}
	return model, nil
}
