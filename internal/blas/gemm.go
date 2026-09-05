// Package blas implements a cache-blocked, multi-threaded single-precision
// general matrix multiply (SGEMM) on top of the kernel micro-kernels.
//
// The algorithm is the classic Goto/BLIS structure:
//
//	for jc in N step NC:            // B column block
//	  for pc in K step KC:          // K block: pack B[pc:pc+KC, jc:jc+NC] into NR-wide panels
//	    for ic in M step MC:        // A row block:  pack A[ic:ic+MC, pc:pc+KC] into MR-wide panels
//	      for jr in NC step NR:     // micro-tile columns
//	        for ir in MC step MR:   // micro-tile rows
//	          C[MR×NR] += Apanel · Bpanel      (SIMD micro-kernel)
//
// The packed B block (KC×NC) stays in L2 while MR-row panels of A stream
// through L1. Packing removes all strides from the inner kernel, so
// transposed or otherwise strided inputs cost nothing extra.
//
// Per K block the driver packs B (into NR panels) and all of A (into MR
// panels) with every worker, then runs a 2-D grid of pure compute tasks
// (A row block × B panel range). Nothing is packed inside a task, and the
// grid keeps all cores busy both for tall (large M) and for wide/short
// (small M, e.g. batch-1 inference) products.
package blas

import (
	"fmt"
	"os"
	"strconv"
	"sync"

	"github.com/fiber/ai/internal/kernel"
	"github.com/fiber/ai/internal/parallel"
)

// Mat is a strided, row-major view of a float32 matrix: element (i, j) is
// Data[i*RS + j*CS]. A transposed matrix is simply a Mat with RS and CS
// swapped; no data is moved.
type Mat struct {
	Data       []float32
	Rows, Cols int
	RS, CS     int
}

// Contiguous returns a Mat over a dense row-major rows×cols matrix.
func Contiguous(data []float32, rows, cols int) Mat {
	return Mat{Data: data, Rows: rows, Cols: cols, RS: cols, CS: 1}
}

// T returns the transposed view.
func (m Mat) T() Mat {
	return Mat{Data: m.Data, Rows: m.Cols, Cols: m.Rows, RS: m.CS, CS: m.RS}
}

func (m Mat) at(i, j int) float32 { return m.Data[i*m.RS+j*m.CS] }

// row returns row i as a slice when the row is contiguous (CS == 1).
func (m Mat) row(i int) []float32 { return m.Data[i*m.RS : i*m.RS+m.Cols] }

// col returns column j as a slice when the column is contiguous (RS == 1).
func (m Mat) col(j int) []float32 { return m.Data[j*m.CS : j*m.CS+m.Rows] }

func (m Mat) check(name string) {
	if m.Rows < 0 || m.Cols < 0 {
		panic(fmt.Sprintf("blas: %s has negative dimensions %d×%d", name, m.Rows, m.Cols))
	}
	if m.Rows == 0 || m.Cols == 0 {
		return
	}
	if m.RS < 0 || m.CS < 0 {
		panic(fmt.Sprintf("blas: %s has negative strides (%d, %d)", name, m.RS, m.CS))
	}
	if need := (m.Rows-1)*m.RS + (m.Cols-1)*m.CS + 1; len(m.Data) < need {
		panic(fmt.Sprintf("blas: %s buffer too small: %d×%d strides (%d, %d) needs %d elements, have %d",
			name, m.Rows, m.Cols, m.RS, m.CS, need, len(m.Data)))
	}
}

// Cache blocking parameters in elements. The defaults are architecture
// specific (see params_*.go); they are variables so benchmarks can tune
// them. MC and NC are rounded down to multiples of MR and NR at use.
var (
	KC = defaultKC
	MC = defaultMC
	NC = defaultNC

	// ParallelThreshold is the number of multiply-adds (m·n·k) below which
	// Gemm runs on the calling goroutine only. Measured on Apple M2 Pro the
	// parallel path wins from roughly 160³.
	ParallelThreshold = 4 * 1024 * 1024
)

// Strategy selects how the compute phase is distributed:
//
//	StrategyShared: all workers pack every A panel of the K block into one
//	  shared buffer, then a grid of (row block × panel range) tasks reads it.
//	StrategyRows: one task per row block; the task packs its own A rows
//	  into a private buffer (L2-hot) and sweeps every B panel. B is shared
//	  packed in both.
//
// FIBERAI_BLAS_STRATEGY=shared|rows overrides the default for experiments.
var Strategy = StrategyShared

const (
	StrategyShared = iota
	StrategyRows
)

// The blocking parameters can be overridden for tuning runs without a
// rebuild: FIBERAI_BLAS_KC, FIBERAI_BLAS_MC and FIBERAI_BLAS_NC (elements).
func init() {
	switch os.Getenv("FIBERAI_BLAS_STRATEGY") {
	case "rows":
		Strategy = StrategyRows
	case "shared":
		Strategy = StrategyShared
	}
	for _, v := range []struct {
		name string
		dst  *int
	}{{"FIBERAI_BLAS_KC", &KC}, {"FIBERAI_BLAS_MC", &MC}, {"FIBERAI_BLAS_NC", &NC}} {
		if s := os.Getenv(v.name); s != "" {
			if n, err := strconv.Atoi(s); err == nil && n > 0 {
				*v.dst = n
			}
		}
	}
}

// Gemm computes C += A·B using up to parallel.Workers() goroutines.
//
// A is m×k, B is k×n and C is m×n; all three may be arbitrarily strided
// except that C must have unit column stride (CS == 1). C is accumulated
// into, not overwritten.
func Gemm(c, a, b Mat) { GemmWorkers(c, a, b, parallel.Workers()) }

// GemmWorkers is Gemm with an explicit upper bound on goroutines.
func GemmWorkers(c, a, b Mat, workers int) {
	a.check("A")
	b.check("B")
	c.check("C")
	if a.Cols != b.Rows || c.Rows != a.Rows || c.Cols != b.Cols {
		panic(fmt.Sprintf("blas: Gemm shape mismatch: C[%d×%d] += A[%d×%d] · B[%d×%d]",
			c.Rows, c.Cols, a.Rows, a.Cols, b.Rows, b.Cols))
	}
	if c.CS != 1 && c.Cols > 1 {
		panic("blas: Gemm requires C with unit column stride")
	}
	m, n, k := a.Rows, b.Cols, a.Cols
	if m == 0 || n == 0 || k == 0 {
		return
	}
	if float64(m)*float64(n)*float64(k) < float64(ParallelThreshold) {
		workers = 1
	}
	if workers < 1 {
		workers = 1
	}

	// Matrix-vector products get dedicated paths: the packed kernel would
	// waste MR-1 of MR rows (or NR-1 of NR columns).
	if m == 1 && gemvRow(c, a, b, workers) {
		return
	}
	if n == 1 && gemvCol(c, a, b, workers) {
		return
	}
	gemm(c, a, b, workers)
}

// gemvRow handles C[1×n] += a[1×k] · B[k×n]. Returns false if the strides
// do not allow a vectorised path.
func gemvRow(c, a, b Mat, workers int) bool {
	n, k := b.Cols, a.Cols
	y := c.Data[:n]
	switch {
	case b.CS == 1: // y += a[p] * B[p, :]
		parallel.RangeWorkers(n, 1024, workers, func(lo, hi int) {
			for p := 0; p < k; p++ {
				kernel.Axpy(a.at(0, p), b.row(p)[lo:hi], y[lo:hi])
			}
		})
		return true
	case b.RS == 1: // y[j] += <a, B[:, j]>
		ar := a.Data[:k]
		if a.CS != 1 {
			ar = make([]float32, k)
			for p := range ar {
				ar[p] = a.at(0, p)
			}
		}
		parallel.RangeWorkers(n, 16, workers, func(lo, hi int) {
			for j := lo; j < hi; j++ {
				y[j] += kernel.Dot(ar, b.col(j))
			}
		})
		return true
	}
	return false
}

// gemvCol handles C[m×1] += A[m×k] · b[k×1].
func gemvCol(c, a, b Mat, workers int) bool {
	m, k := a.Rows, a.Cols
	switch {
	case a.CS == 1: // y[i] += <A[i, :], b>
		bc := b.Data[:k]
		if b.RS != 1 {
			bc = make([]float32, k)
			for p := range bc {
				bc[p] = b.at(p, 0)
			}
		}
		parallel.RangeWorkers(m, 16, workers, func(lo, hi int) {
			for i := lo; i < hi; i++ {
				c.Data[i*c.RS] += kernel.Dot(a.row(i), bc)
			}
		})
		return true
	case a.RS == 1 && c.RS == 1: // y += b[p] * A[:, p]
		y := c.Data[:m]
		parallel.RangeWorkers(m, 1024, workers, func(lo, hi int) {
			for p := 0; p < k; p++ {
				kernel.Axpy(b.at(p, 0), a.col(p)[lo:hi], y[lo:hi])
			}
		})
		return true
	}
	return false
}

// bufPool recycles packing buffers between calls.
var bufPool sync.Pool

func getBuf(n int) []float32 {
	if v := bufPool.Get(); v != nil {
		s := *(v.(*[]float32))
		if cap(s) >= n {
			return s[:n]
		}
	}
	return make([]float32, n)
}

func putBuf(s []float32) { bufPool.Put(&s) }

func roundUp(x, m int) int { return (x + m - 1) / m * m }

func gemm(c, a, b Mat, workers int) {
	mr, nr := kernel.MR, kernel.NR
	m, n, k := a.Rows, b.Cols, a.Cols
	kc := max(1, KC)
	mc := max(mr, MC/mr*mr)
	nc := max(nr, NC/nr*nr)
	kcEff := min(kc, k)
	mPack := roundUp(m, mr)

	bp := getBuf(kcEff * min(nc, roundUp(n, nr)))
	defer putBuf(bp)
	ap := getBuf(mPack * kcEff) // all of A's rows for one K block, in MR panels
	defer putBuf(ap)

	nIc := (m + mc - 1) / mc
	for jc := 0; jc < n; jc += nc {
		jb := min(nc, n-jc)
		nPanels := (jb + nr - 1) / nr
		for pc := 0; pc < k; pc += kc {
			pb := min(kc, k-pc)

			if Strategy == StrategyRows {
				packB(bp, b, pc, jc, pb, jb, nr, workers)
				computeRows(c, a, bp, ic0(m, mc), jc, pc, pb, jb, mr, nr, workers)
				continue
			}
			// Phase 1: pack the B panels and all A panels of this K block in
			// one parallel round (the two packings are independent).
			packAB(ap, bp, a, b, pc, jc, m, pb, jb, mr, nr, workers)

			// Phase 3: pure compute over a grid of (row block × panel range)
			// tasks. A few tasks per worker keep dynamic scheduling effective
			// on uneven cores; nothing is packed inside a task, so the grid
			// can be fine-grained without redundant work.
			nJ := 1
			if workers > 1 {
				nJ = min(nPanels, max(1, (3*workers+nIc-1)/nIc))
			}
			panelsPerTask := (nPanels + nJ - 1) / nJ
			nJ = (nPanels + panelsPerTask - 1) / panelsPerTask

			parallel.ForWorkers(nIc*nJ, workers, func(task int) {
				ic := (task / nJ) * mc
				jt := task % nJ
				ib := min(mc, m-ic)
				var tmp []float32 // edge-tile scratch, allocated on demand
				p0 := jt * panelsPerTask
				p1 := min(p0+panelsPerTask, nPanels)
				for jr := p0; jr < p1; jr++ {
					j := jr * nr
					cols := min(nr, jb-j)
					bpanel := &bp[jr*nr*pb]
					for ir := 0; ir < ib; ir += mr {
						rows := min(mr, ib-ir)
						apanel := &ap[(ic+ir)*pb]
						if rows == mr && cols == nr {
							kernel.Gemm(pb, apanel, bpanel, &c.Data[(ic+ir)*c.RS+jc+j], c.RS)
							continue
						}
						if tmp == nil {
							tmp = make([]float32, mr*nr)
						} else {
							clear(tmp)
						}
						kernel.Gemm(pb, apanel, bpanel, &tmp[0], nr)
						for i := 0; i < rows; i++ {
							off := (ic+ir+i)*c.RS + jc + j
							crow := c.Data[off : off+cols]
							kernel.Add(crow, tmp[i*nr:i*nr+cols], crow)
						}
					}
				}
			})
		}
	}
}

// packAB packs the B panels of block (pc, jc) and all A panels of block
// pc in a single parallel round: items [0, nB) are B panels, [nB, nB+nA)
// are groups of A panels.
func packAB(ap, bp []float32, a, b Mat, pc, jc, m, pb, jb, mr, nr, workers int) {
	nB := (jb + nr - 1) / nr
	aPanels := (m + mr - 1) / mr
	const group = 4
	nA := (aPanels + group - 1) / group
	if workers <= 1 {
		packB(bp, b, pc, jc, pb, jb, nr, 1)
		packAAll(ap, a, pc, m, pb, mr, 1)
		return
	}
	parallel.ForWorkers(nB+nA, workers, func(i int) {
		if i < nB {
			packBPanel(bp, b, pc, jc, pb, jb, nr, i)
			return
		}
		p0 := (i - nB) * group
		for p := p0; p < min(p0+group, aPanels); p++ {
			i0 := p * mr
			packA(ap[i0*pb:(i0+mr)*pb], a, i0, pc, min(mr, m-i0), pb, mr)
		}
	})
}

// ic0 returns the row block starts for m rows in blocks of mc.
func ic0(m, mc int) []int {
	starts := make([]int, 0, (m+mc-1)/mc)
	for ic := 0; ic < m; ic += mc {
		starts = append(starts, ic)
	}
	return starts
}

// computeRows is the StrategyRows compute phase: one task per row block,
// each packing its own A rows into a private buffer and sweeping all
// panels of the shared packed B block.
func computeRows(c, a Mat, bp []float32, starts []int, jc, pc, pb, jb, mr, nr, workers int) {
	m := a.Rows
	nPanels := (jb + nr - 1) / nr
	parallel.ForWorkers(len(starts), workers, func(task int) {
		ic := starts[task]
		ib := ic0Block(starts, task, m)
		ap := getBuf(roundUp(ib, mr) * pb)
		defer putBuf(ap)
		packA(ap, a, ic, pc, ib, pb, mr)
		var tmp []float32
		for jr := 0; jr < nPanels; jr++ {
			j := jr * nr
			cols := min(nr, jb-j)
			bpanel := &bp[jr*nr*pb]
			for ir := 0; ir < ib; ir += mr {
				rows := min(mr, ib-ir)
				apanel := &ap[ir*pb]
				if rows == mr && cols == nr {
					kernel.Gemm(pb, apanel, bpanel, &c.Data[(ic+ir)*c.RS+jc+j], c.RS)
					continue
				}
				if tmp == nil {
					tmp = make([]float32, mr*nr)
				} else {
					clear(tmp)
				}
				kernel.Gemm(pb, apanel, bpanel, &tmp[0], nr)
				for i := 0; i < rows; i++ {
					off := (ic+ir+i)*c.RS + jc + j
					crow := c.Data[off : off+cols]
					kernel.Add(crow, tmp[i*nr:i*nr+cols], crow)
				}
			}
		}
	})
}

// ic0Block returns the size of row block task.
func ic0Block(starts []int, task, m int) int {
	if task+1 < len(starts) {
		return starts[task+1] - starts[task]
	}
	return m - starts[task]
}

// packAAll packs rows [0, m) of A for the K block at p0 into dst as
// consecutive MR panels (panel p holds rows p*mr .. p*mr+mr), in parallel
// over panels. The last panel is zero-padded.
func packAAll(dst []float32, a Mat, p0, m, pb, mr, workers int) {
	nPanels := (m + mr - 1) / mr
	parallel.RangeWorkers(nPanels, 4, workers, func(lo, hi int) {
		for p := lo; p < hi; p++ {
			i0 := p * mr
			packA(dst[i0*pb:(i0+mr)*pb], a, i0, p0, min(mr, m-i0), pb, mr)
		}
	})
}

// packA copies the ib×pb block of A at (i0, p0) into dst as consecutive
// MR-row panels, each stored k-major: panel[p*mr + i] = A[i0+ir+i, p0+p].
// Rows beyond ib in the last panel are zero-filled.
func packA(dst []float32, a Mat, i0, p0, ib, pb, mr int) {
	for ir := 0; ir < ib; ir += mr {
		rows := min(mr, ib-ir)
		panel := dst[ir*pb : (ir+mr)*pb]
		switch {
		case a.CS == 1: // rows of A are contiguous: stream each row
			for i := 0; i < rows; i++ {
				src := a.Data[(i0+ir+i)*a.RS+p0:][:pb]
				for p, v := range src {
					panel[p*mr+i] = v
				}
			}
		case a.RS == 1: // columns of A are contiguous (transposed input)
			for p := 0; p < pb; p++ {
				src := a.Data[(p0+p)*a.CS+i0+ir:][:rows]
				copy(panel[p*mr:p*mr+rows], src)
			}
		default:
			for i := 0; i < rows; i++ {
				base := (i0+ir+i)*a.RS + p0*a.CS
				for p := 0; p < pb; p++ {
					panel[p*mr+i] = a.Data[base+p*a.CS]
				}
			}
		}
		if rows < mr {
			for p := 0; p < pb; p++ {
				clear(panel[p*mr+rows : p*mr+mr])
			}
		}
	}
}

// packB copies the pb×jb block of B at (p0, j0) into dst as consecutive
// NR-column panels, each stored k-major: panel[p*nr + j] = B[p0+p, j0+jr+j].
// Columns beyond jb in the last panel are zero-filled.
func packB(dst []float32, b Mat, p0, j0, pb, jb, nr, workers int) {
	nPanels := (jb + nr - 1) / nr
	packOne := func(pi int) { packBPanel(dst, b, p0, j0, pb, jb, nr, pi) }
	if workers > 1 && nPanels >= 8 {
		parallel.ForWorkers(nPanels, workers, packOne)
		return
	}
	for pi := 0; pi < nPanels; pi++ {
		packOne(pi)
	}
}

// packBPanel packs panel pi of the B block at (p0, j0).
func packBPanel(dst []float32, b Mat, p0, j0, pb, jb, nr, pi int) {
	{
		jr := pi * nr
		cols := min(nr, jb-jr)
		panel := dst[jr*pb : (jr+nr)*pb]
		switch {
		case b.CS == 1: // rows of B are contiguous: one copy per k
			for p := 0; p < pb; p++ {
				src := b.Data[(p0+p)*b.RS+j0+jr:][:cols]
				copy(panel[p*nr:p*nr+cols], src)
			}
		case b.RS == 1: // columns of B are contiguous (transposed input)
			for j := 0; j < cols; j++ {
				src := b.Data[(j0+jr+j)*b.CS+p0:][:pb]
				for p, v := range src {
					panel[p*nr+j] = v
				}
			}
		default:
			for p := 0; p < pb; p++ {
				base := (p0+p)*b.RS + (j0+jr)*b.CS
				for j := 0; j < cols; j++ {
					panel[p*nr+j] = b.Data[base+j*b.CS]
				}
			}
		}
		if cols < nr {
			for p := 0; p < pb; p++ {
				clear(panel[p*nr+cols : p*nr+nr])
			}
		}
	}
}
