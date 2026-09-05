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

// SaveParams writes the parameters of m in the order Params returns
// them. Gradients and optimizer state are not saved.
func SaveParams(w io.Writer, m Module) error {
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
func LoadParams(r io.Reader, m Module) error {
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
