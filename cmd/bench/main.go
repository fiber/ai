// bench measures the throughput of the tensor stack and prints Markdown
// tables. The Python counterpart in benchmarks/python/bench.py runs the
// same workloads with NumPy and PyTorch.
//
//	go run ./cmd/bench            # all sections
//	go run ./cmd/bench -quick     # smaller sizes, shorter runs
package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"runtime/pprof"
	"strings"
	"time"

	"github.com/fiber/ai/models/gemma"
	"github.com/fiber/ai/nn"
	"github.com/fiber/ai/optim"
	"github.com/fiber/ai/tensor"
)

var (
	quick      = flag.Bool("quick", false, "smaller sizes and shorter runs")
	duration   = flag.Duration("d", 700*time.Millisecond, "measurement time per case")
	only       = flag.String("only", "", "run only these sections (comma-separated: gemm,elementwise,reductions,mlp)")
	cpuprofile = flag.String("cpuprofile", "", "write a CPU profile of the run to this file")
)

// timeIt runs fn repeatedly for the measurement duration (after one warm-up
// call) and returns the mean seconds per call.
func timeIt(fn func()) float64 {
	fn()
	iters := 0
	start := time.Now()
	for time.Since(start) < *duration {
		fn()
		iters++
	}
	return time.Since(start).Seconds() / float64(iters)
}

func main() {
	flag.Parse()
	fmt.Printf("## fiber/ai — %s/%s, %d CPUs, GOMAXPROCS %d, backend %s, Go %s\n\n",
		runtime.GOOS, runtime.GOARCH, runtime.NumCPU(), runtime.GOMAXPROCS(0), tensor.Backend(), runtime.Version())

	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		if err := pprof.StartCPUProfile(f); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		defer pprof.StopCPUProfile()
	}
	sections := map[string]func(){"gemm": benchGemm, "elementwise": benchElementwise, "reductions": benchReductions, "attention": benchAttention, "conv": benchConv, "mlp": benchMLP, "embed": benchEmbed}
	order := []string{"gemm", "elementwise", "reductions", "attention", "conv", "mlp", "embed"}
	if *only != "" {
		order = strings.Split(*only, ",")
	}
	for _, name := range order {
		fn, ok := sections[strings.TrimSpace(name)]
		if !ok {
			fmt.Fprintf(os.Stderr, "unknown section %q\n", name)
			os.Exit(2)
		}
		fn()
	}
	printAllocStats()
}

// printAllocStats shows how the off-heap result allocator behaved over
// the run: reuse rate, retained memory and the number of GC cycles.
func printAllocStats() {
	hits, misses, retained, pinned := tensor.MappedStats()
	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)
	fmt.Println()
	fmt.Printf("allocator: mapped hits %d, misses %d, retained %d MiB, pinned %d MiB; GC cycles %d, forced %d\n",
		hits, misses, retained>>20, pinned>>20, ms.NumGC, ms.NumForcedGC)
}

func benchGemm() {
	sizes := []int{128, 256, 512, 1024, 2048}
	if *quick {
		sizes = []int{128, 256, 512, 1024}
	}
	// Results are released after each call, as PyTorch's caching allocator
	// does implicitly: the output buffer then stays in cache between calls
	// instead of rotating through cold memory until the next collection.
	fmt.Println("### Matrix multiply (float32, GFLOPS; result released)")
	fmt.Println()
	fmt.Println("| n (n×n · n×n) | 1 thread | all threads |")
	fmt.Println("|---:|---:|---:|")
	all := tensor.Threads()
	for _, n := range sizes {
		a, b := tensor.Randn(n, n), tensor.Randn(n, n)
		flops := 2 * float64(n) * float64(n) * float64(n)
		tensor.SetThreads(1)
		t1 := timeIt(func() { a.MatMul(b).Release() })
		tensor.SetThreads(all)
		tn := timeIt(func() { a.MatMul(b).Release() })
		fmt.Printf("| %d | %.1f | %.1f |\n", n, flops/t1/1e9, flops/tn/1e9)
	}
	fmt.Println()
	fmt.Println("| shape | GFLOPS (all threads) |")
	fmt.Println("|---|---:|")
	for _, s := range [][3]int{{1, 4096, 4096}, {8, 4096, 4096}, {64, 1024, 1024}, {256, 768, 3072}} {
		m, k, n := s[0], s[1], s[2]
		a, b := tensor.Randn(m, k), tensor.Randn(k, n)
		flops := 2 * float64(m) * float64(n) * float64(k)
		t := timeIt(func() { a.MatMul(b).Release() })
		fmt.Printf("| [%d×%d]·[%d×%d] | %.1f |\n", m, k, k, n, flops/t/1e9)
	}
	// transposed operand: no copy is made
	a, b := tensor.Randn(1024, 1024), tensor.Randn(1024, 1024)
	bt := b.T()
	t := timeIt(func() { a.MatMul(bt).Release() })
	fmt.Printf("| [1024×1024]·[1024×1024]ᵀ (view) | %.1f |\n", 2*1024.0*1024*1024/t/1e9)
	fmt.Println()
}

func benchElementwise() {
	fmt.Println("### Element-wise (all threads)")
	fmt.Println()
	fmt.Println("| op | n | time | GB/s |")
	fmt.Println("|---|---:|---:|---:|")
	for _, n := range []int{1 << 16, 1 << 20, 1 << 24} {
		x, y := tensor.Randn(n), tensor.Randn(n)
		row := tensor.Randn(1024)
		x2 := x.Reshape(-1, 1024)
		cases := []struct {
			name  string
			bytes float64
			fn    func()
		}{
			{"x + y", 12 * float64(n), func() { x.Add(y) }},
			{"x + y, result released", 12 * float64(n), func() { x.Add(y).Release() }},
			{"x * y", 12 * float64(n), func() { x.Mul(y) }},
			{"x + row (broadcast)", 8 * float64(n), func() { x2.Add(row) }},
			{"x * 2.5", 8 * float64(n), func() { x.MulScalar(2.5) }},
			{"exp(x)", 8 * float64(n), func() { x.Exp() }},
			{"tanh(x)", 8 * float64(n), func() { x.Tanh() }},
			{"sigmoid(x)", 8 * float64(n), func() { x.Sigmoid() }},
			{"gelu(x)", 8 * float64(n), func() { x.GELU() }},
			{"relu(x)", 8 * float64(n), func() { x.ReLU() }},
		}
		for _, c := range cases {
			t := timeIt(c.fn)
			fmt.Printf("| %s | %s | %s | %.1f |\n", c.name, fmtN(n), fmtDur(t), c.bytes/t/1e9)
		}
	}
	fmt.Println()
}

func benchReductions() {
	fmt.Println("### Reductions (all threads)")
	fmt.Println()
	fmt.Println("| op | shape | time | GB/s |")
	fmt.Println("|---|---|---:|---:|")
	x := tensor.Randn(4096, 4096)
	n := float64(x.Size())
	cases := []struct {
		name  string
		bytes float64
		fn    func()
	}{
		{"sum()", 4 * n, func() { x.Sum() }},
		{"sum(dim=0)", 4 * n, func() { x.Sum(0) }},
		{"sum(dim=1)", 4 * n, func() { x.Sum(1) }},
		{"max(dim=1)", 4 * n, func() { x.Max(1) }},
		{"softmax(dim=1)", 8 * n, func() { x.Softmax(1) }},
		{"layernorm", 8 * n, func() { tensor.LayerNorm(x, tensor.Ones(4096), tensor.Zeros(4096), 1e-5) }},
		{"transpose+contiguous", 8 * n, func() { x.T().Contiguous() }},
	}
	for _, c := range cases {
		t := timeIt(c.fn)
		fmt.Printf("| %s | [4096×4096] | %s | %.1f |\n", c.name, fmtDur(t), c.bytes/t/1e9)
	}
	fmt.Println()
}

// benchAttention times scaled dot-product attention on a transformer-sized
// shape: batch 8, 8 heads, 512 tokens, head dimension 64.
func benchAttention() {
	fmt.Println("### Attention (all threads, NoGrad, result released)")
	fmt.Println()
	fmt.Println("| shape | time | GFLOPS |")
	fmt.Println("|---|---:|---:|")
	b, h, n, d := 8, 8, 512, 64
	q, k, v := tensor.Randn(b, h, n, d), tensor.Randn(b, h, n, d), tensor.Randn(b, h, n, d)
	mask := tensor.CausalMask(n)
	flops := 2.0 * 2 * float64(b*h) * float64(n) * float64(n) * float64(d) // two products
	tensor.NoGrad(func() {
		t := timeIt(func() { tensor.Attention(q, k, v, nil).Release() })
		fmt.Printf("| [%d×%d×%d×%d] q·kᵀ, softmax, ·v | %s | %.1f |\n", b, h, n, d, fmtDur(t), flops/t/1e9)
		t = timeIt(func() { tensor.Attention(q, k, v, mask).Release() })
		fmt.Printf("| same with causal mask | %s | %.1f |\n", fmtDur(t), flops/t/1e9)
	})
	fmt.Println()
}

// benchConv times a 3×3 convolution on a mid-network shape: 32 images,
// 64 → 64 channels, 56×56.
func benchConv() {
	fmt.Println("### Convolution (all threads, NoGrad, result released)")
	fmt.Println()
	fmt.Println("| shape | time | GFLOPS |")
	fmt.Println("|---|---:|---:|")
	n, c, h, w, o, k := 32, 64, 56, 56, 64, 3
	x := tensor.Randn(n, c, h, w)
	wt := tensor.Randn(o, c, k, k)
	b := tensor.Randn(o)
	flops := 2.0 * float64(n) * float64(o) * float64(h*w) * float64(c*k*k)
	tensor.NoGrad(func() {
		t := timeIt(func() { tensor.Conv2D(x, wt, b, 1, 1).Release() })
		fmt.Printf("| [%d×%d×%d×%d] · %d filters %d×%d, pad 1 | %s | %.1f |\n", n, c, h, w, o, k, k, fmtDur(t), flops/t/1e9)
	})
	fmt.Println()
}

// benchEmbed times EmbeddingGemma on 32 sentences of about 64 tokens, the
// shape the models manual quotes against sentence-transformers. It is
// skipped unless FIBERAI_MODELS names a directory holding embeddinggemma-300m.
func benchEmbed() {
	fmt.Println("### EmbeddingGemma (32 sentences ≈ 64 tokens, one batch)")
	fmt.Println()
	root := os.Getenv("FIBERAI_MODELS")
	if root == "" {
		fmt.Println("skipped: set FIBERAI_MODELS to the directory holding embeddinggemma-300m")
		fmt.Println()
		return
	}
	dir := filepath.Join(root, "embeddinggemma-300m")
	m, err := gemma.Load(dir)
	if err != nil {
		fmt.Printf("skipped: %v\n\n", err)
		return
	}
	defer m.Close()
	sentence := strings.TrimSpace(strings.Repeat("the quick brown fox jumps over the lazy dog ", 7))
	texts := make([]string, 32)
	for i := range texts {
		texts[i] = sentence
	}
	ntok := len(m.Tokenizer().Encode(sentence))
	m.Embed(texts) // warm
	t := timeIt(func() {
		if _, err := m.Embed(texts); err != nil {
			panic(err)
		}
	})
	fmt.Println("| shape | time / batch | sentences/s |")
	fmt.Println("|---|---:|---:|")
	fmt.Printf("| 32 × %d tokens, dim %d | %s | %.0f |\n", ntok, m.Dim(), fmtDur(t), 32/t)
	fmt.Println()
}

func benchMLP() {
	fmt.Println("### MLP 784→512→512→10, batch 256 (all threads)")
	fmt.Println()
	fmt.Println("| phase | time / batch | samples/s |")
	fmt.Println("|---|---:|---:|")
	const batch = 256
	model := nn.Sequential{
		nn.NewLinear(784, 512), nn.ReLU{},
		nn.NewLinear(512, 512), nn.ReLU{},
		nn.NewLinear(512, 10),
	}
	opt := optim.NewAdam(model.Params(), 1e-3)
	x := tensor.Randn(batch, 784)
	targets := make([]int, batch)
	for i := range targets {
		targets[i] = i % 10
	}
	tInf := timeIt(func() { tensor.NoGrad(func() { model.Forward(x) }) })
	fmt.Printf("| forward (NoGrad) | %s | %s |\n", fmtDur(tInf), fmtN(int(batch/tInf)))
	tFB := timeIt(func() {
		loss := tensor.CrossEntropy(model.Forward(x), targets)
		opt.ZeroGrad()
		loss.Backward()
	})
	fmt.Printf("| forward + backward | %s | %s |\n", fmtDur(tFB), fmtN(int(batch/tFB)))
	tStep := timeIt(func() {
		loss := tensor.CrossEntropy(model.Forward(x), targets)
		opt.ZeroGrad()
		loss.Backward()
		opt.Step()
	})
	fmt.Printf("| forward + backward + Adam step | %s | %s |\n", fmtDur(tStep), fmtN(int(batch/tStep)))
	fmt.Println()
}

func fmtDur(sec float64) string {
	switch {
	case sec < 1e-3:
		return fmt.Sprintf("%.1f µs", sec*1e6)
	case sec < 1:
		return fmt.Sprintf("%.2f ms", sec*1e3)
	}
	return fmt.Sprintf("%.2f s", sec)
}

func fmtN(n int) string {
	switch {
	case n >= 1<<20 && n%(1<<20) == 0:
		return fmt.Sprintf("%dM", n>>20)
	case n >= 1<<10 && n%(1<<10) == 0:
		return fmt.Sprintf("%dK", n>>10)
	case n >= 1000000:
		return fmt.Sprintf("%.2fM", float64(n)/1e6)
	case n >= 1000:
		return fmt.Sprintf("%.1fK", float64(n)/1e3)
	}
	return fmt.Sprintf("%d", n)
}
