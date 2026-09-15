package nn

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"

	"github.com/fiber/ai/tensor"
)

// Parameter files: a small self-describing binary format so a trained
// model can be written to disk and read back into a freshly constructed
// module of the same architecture.
//
//	magic "FAIP", version uint32 (1), count uint32,
//	per parameter: dims uint32, shape [dims]uint32, data [size]float32,
//	all little-endian.
const (
	paramsMagic   = "FAIP"
	paramsVersion = 1
)

// Parameterised is anything with parameters to save: every Module, and
// also a model whose forward pass does not take a single tensor. A
// decoder takes token ids and a batch layout, so it is not a Module,
// and saving never needed one — only Params.
type Parameterised interface {
	Params() []*tensor.Tensor
}

// SaveParams writes the parameters of m in the order Params returns
// them. Gradients and optimizer state are not saved.
func SaveParams(w io.Writer, m Parameterised) error {
	params := m.Params()
	if _, err := io.WriteString(w, paramsMagic); err != nil {
		return err
	}
	hdr := []uint32{paramsVersion, uint32(len(params))}
	if err := binary.Write(w, binary.LittleEndian, hdr); err != nil {
		return err
	}
	for _, p := range params {
		shape := p.Shape()
		dims := make([]uint32, 0, len(shape)+1)
		dims = append(dims, uint32(len(shape)))
		for _, d := range shape {
			dims = append(dims, uint32(d))
		}
		if err := binary.Write(w, binary.LittleEndian, dims); err != nil {
			return err
		}
		data := p.Float32s()
		buf := make([]byte, 4*len(data))
		for i, v := range data {
			binary.LittleEndian.PutUint32(buf[4*i:], math.Float32bits(v))
		}
		if _, err := w.Write(buf); err != nil {
			return err
		}
	}
	return nil
}

// LoadParams reads parameters written by SaveParams into m, which must
// have the same architecture: the same number of parameters with the
// same shapes, in the same order. The tensors are updated in place, so
// optimizers already holding them keep working.
func LoadParams(r io.Reader, m Parameterised) error {
	params := m.Params()
	magic := make([]byte, 4)
	if _, err := io.ReadFull(r, magic); err != nil {
		return fmt.Errorf("nn: reading parameters: %w", err)
	}
	if string(magic) != paramsMagic {
		return fmt.Errorf("nn: not a parameter file (magic %q)", magic)
	}
	var hdr [2]uint32
	if err := binary.Read(r, binary.LittleEndian, &hdr); err != nil {
		return fmt.Errorf("nn: reading parameters: %w", err)
	}
	if hdr[0] != paramsVersion {
		return fmt.Errorf("nn: parameter file version %d, want %d", hdr[0], paramsVersion)
	}
	if int(hdr[1]) != len(params) {
		return fmt.Errorf("nn: file holds %d parameters, model has %d", hdr[1], len(params))
	}
	for i, p := range params {
		var nd uint32
		if err := binary.Read(r, binary.LittleEndian, &nd); err != nil {
			return fmt.Errorf("nn: parameter %d: %w", i, err)
		}
		shape := make([]uint32, nd)
		if err := binary.Read(r, binary.LittleEndian, shape); err != nil {
			return fmt.Errorf("nn: parameter %d: %w", i, err)
		}
		want := p.Shape()
		same := len(shape) == len(want)
		for j := range shape {
			same = same && int(shape[j]) == want[j]
		}
		if !same {
			return fmt.Errorf("nn: parameter %d has shape %v in the file, %v in the model", i, shape, want)
		}
		buf := make([]byte, 4*p.Size())
		if _, err := io.ReadFull(r, buf); err != nil {
			return fmt.Errorf("nn: parameter %d: %w", i, err)
		}
		data := make([]float32, p.Size())
		for j := range data {
			data[j] = math.Float32frombits(binary.LittleEndian.Uint32(buf[4*j:]))
		}
		src := tensor.New(data, want...)
		tensor.NoGrad(func() { p.CopyFrom(src) })
	}
	return nil
}

// Snapshot is an in-memory copy of a model's parameters, for the loop
// that keeps the best model rather than the last one. Early stopping
// needs exactly this: capture whenever the validation loss improves,
// restore at the end.
//
//	best := nn.NewSnapshot(model)
//	for epoch := range epochs {
//	    train(model)
//	    if l := validate(model); l < bestLoss {
//	        bestLoss = l
//	        best.Capture(model)
//	    }
//	}
//	best.Restore(model)
//
// Gradients and optimiser state are not part of a snapshot: restoring
// does not rewind Adam's moments, and training on after a restore
// continues with the optimiser state the discarded epochs produced.
type Snapshot struct {
	shapes [][]int
	data   [][]float32
}

// NewSnapshot allocates a snapshot of m and captures it.
func NewSnapshot(m Parameterised) *Snapshot {
	params := m.Params()
	s := &Snapshot{shapes: make([][]int, len(params)), data: make([][]float32, len(params))}
	for i, p := range params {
		s.shapes[i] = p.Shape()
		s.data[i] = make([]float32, p.Size())
	}
	// The shapes were just read from m, so this cannot fail.
	_ = s.Capture(m)
	return s
}

// Capture overwrites the snapshot with m's current parameters, copying
// into the buffers it already holds, so what it allocates does not grow
// with the model and it is cheap enough to call every epoch.
func (s *Snapshot) Capture(m Parameterised) error {
	params, err := s.check(m)
	if err != nil {
		return err
	}
	for i, p := range params {
		p.CopyTo(s.data[i])
	}
	return nil
}

// Restore copies the snapshot back into m's parameters in place, so an
// optimiser already holding them keeps working.
func (s *Snapshot) Restore(m Parameterised) error {
	params, err := s.check(m)
	if err != nil {
		return err
	}
	for i, p := range params {
		src := tensor.FromSlice(s.data[i], s.shapes[i]...)
		tensor.NoGrad(func() { p.CopyFrom(src) })
	}
	return nil
}

func (s *Snapshot) check(m Parameterised) ([]*tensor.Tensor, error) {
	params := m.Params()
	if len(params) != len(s.shapes) {
		return nil, fmt.Errorf("nn: snapshot holds %d parameters, model has %d", len(s.shapes), len(params))
	}
	for i, p := range params {
		got, want := p.Shape(), s.shapes[i]
		same := len(got) == len(want)
		for j := range got {
			same = same && got[j] == want[j]
		}
		if !same {
			return nil, fmt.Errorf("nn: parameter %d has shape %v in the model, %v in the snapshot", i, got, want)
		}
	}
	return params, nil
}
