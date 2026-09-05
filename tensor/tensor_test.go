package tensor

import (
	"math"
	"math/rand/v2"
	"strings"
	"testing"
)

func approx(a, b float32, tol float64) bool {
	x, y := float64(a), float64(b)
	return math.Abs(x-y) <= tol*(1+math.Abs(y))
}

func assertClose(t *testing.T, name string, got, want *Tensor, tol float64) {
	t.Helper()
	if !got.shape.Equal(want.shape) {
		t.Fatalf("%s: shape %v, want %v", name, got.shape, want.shape)
	}
	g, w := got.Data(), want.Data()
	for i := range g {
		if !approx(g[i], w[i], tol) {
			t.Fatalf("%s: element %d = %g, want %g\ngot:\n%v\nwant:\n%v", name, i, g[i], w[i], got, want)
		}
	}
}

func expectError(t *testing.T, name string, fn func()) {
	t.Helper()
	if err := Try(fn); err == nil {
		t.Errorf("%s: expected a tensor error", name)
	} else if _, ok := err.(*Error); !ok {
		t.Errorf("%s: expected *Error, got %T", name, err)
	}
}

func TestConstructors(t *testing.T) {
	x := New([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	if !x.Shape().Equal(Shape{2, 3}) || x.Size() != 6 || x.Dims() != 2 || x.Dim(-1) != 3 {
		t.Fatalf("bad shape info: %#v", x)
	}
	if x.At(1, 2) != 6 || x.At(-1, -1) != 6 || x.At(0, 1) != 2 {
		t.Fatal("At returned wrong values")
	}
	data := []float32{1, 2, 3}
	y := FromSlice(data, 3)
	data[0] = 42
	if y.At(0) != 42 {
		t.Fatal("FromSlice must share storage")
	}
	z := New(data, 3)
	data[1] = 99
	if z.At(1) != 2 {
		t.Fatal("New must copy")
	}
	if s := Scalar(3.5); s.Dims() != 0 || s.Item() != 3.5 || s.String() != "3.5" {
		t.Fatalf("scalar: %v", s)
	}
	if !Ones(2, 2).Equal(Full(1, 2, 2)) || !ZerosLike(x).Equal(Zeros(2, 3)) {
		t.Fatal("Ones/Full/ZerosLike mismatch")
	}
	if e := Eye(3); e.At(1, 1) != 1 || e.At(0, 1) != 0 || e.Sum().Item() != 3 {
		t.Fatal("Eye")
	}
	if a := Arange(0, 5, 1); !a.Equal(New([]float32{0, 1, 2, 3, 4}, 5)) {
		t.Fatalf("Arange: %v", a)
	}
	if a := Arange(1, 2, 0.25); a.Size() != 4 {
		t.Fatalf("Arange fractional: %v", a)
	}
	if l := Linspace(0, 1, 5); !l.Equal(New([]float32{0, 0.25, 0.5, 0.75, 1}, 5)) {
		t.Fatalf("Linspace: %v", l)
	}
	if o := OneHot([]int{2, 0}, 3); !o.Equal(New([]float32{0, 0, 1, 1, 0, 0}, 2, 3)) {
		t.Fatalf("OneHot: %v", o)
	}
	Seed(1)
	r1 := Randn(4, 4)
	Seed(1)
	r2 := Randn(4, 4)
	if !r1.Equal(r2) {
		t.Fatal("Seed must make Randn reproducible")
	}
	rng := rand.New(rand.NewPCG(1, 2))
	u := UniformFrom(rng, -1, 1, 1000)
	if u.Max().Item() > 1 || u.Min().Item() < -1 {
		t.Fatal("Uniform out of range")
	}
	if m := Rand(10000).Mean().Item(); m < 0.45 || m > 0.55 {
		t.Fatalf("Rand mean %v", m)
	}
	expectError(t, "New size", func() { New([]float32{1, 2}, 3) })
	expectError(t, "negative shape", func() { Zeros(-1) })
	expectError(t, "At rank", func() { x.At(1) })
	expectError(t, "At range", func() { x.At(2, 0) })
	expectError(t, "Item", func() { x.Item() })
	expectError(t, "OneHot range", func() { OneHot([]int{3}, 3) })
}

func TestViewsAndContiguity(t *testing.T) {
	x := Arange(0, 24, 1).Reshape(2, 3, 4)
	if !x.IsContiguous() {
		t.Fatal("reshape of contiguous must be contiguous")
	}
	tr := x.Transpose(0, 2)
	if tr.IsContiguous() || !tr.Shape().Equal(Shape{4, 3, 2}) || tr.At(3, 1, 1) != x.At(1, 1, 3) {
		t.Fatalf("Transpose: %#v", tr)
	}
	c := tr.Contiguous()
	if !c.IsContiguous() || !c.Equal(tr) {
		t.Fatal("Contiguous copy differs")
	}
	// Data on a view must be a copy, on a contiguous tensor the storage
	d := tr.Data()
	d[0] = -1
	if tr.At(0, 0, 0) == -1 {
		t.Fatal("Data of a strided view must copy")
	}
	x.Data()[0] = -7
	if x.At(0, 0, 0) != -7 {
		t.Fatal("Data of a contiguous tensor must be the storage")
	}
	x.Data()[0] = 0

	p := x.Permute(2, 0, 1)
	if !p.Shape().Equal(Shape{4, 2, 3}) || p.At(3, 1, 2) != x.At(1, 2, 3) {
		t.Fatalf("Permute: %v", p.Shape())
	}
	m := New([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	if !m.T().Equal(New([]float32{1, 4, 2, 5, 3, 6}, 3, 2)) {
		t.Fatalf("T: %v", m.T())
	}
	if !m.T().Reshape(6).Equal(New([]float32{1, 4, 2, 5, 3, 6}, 6)) {
		t.Fatal("Reshape of a view must copy in row-major order")
	}
	if r := x.Reshape(-1, 4); !r.Shape().Equal(Shape{6, 4}) {
		t.Fatalf("Reshape -1: %v", r.Shape())
	}
	if !x.Flatten().Shape().Equal(Shape{24}) {
		t.Fatal("Flatten")
	}
	s := x.Narrow(1, 1, 2)
	if !s.Shape().Equal(Shape{2, 2, 4}) || s.At(1, 0, 0) != x.At(1, 1, 0) || s.IsContiguous() {
		t.Fatalf("Narrow: %v", s)
	}
	if !x.Slice(2, 1, 3).Equal(x.Narrow(2, 1, 2)) {
		t.Fatal("Slice")
	}
	row := x.Row(1)
	if !row.Shape().Equal(Shape{3, 4}) || !row.IsContiguous() || row.At(2, 3) != 23 {
		t.Fatalf("Row: %v", row)
	}
	if !x.Select(2, -1).Equal(New([]float32{3, 7, 11, 15, 19, 23}, 2, 3)) {
		t.Fatalf("Select: %v", x.Select(2, -1))
	}
	e := New([]float32{1, 2, 3}, 3).Expand(2, 3)
	if !e.Equal(New([]float32{1, 2, 3, 1, 2, 3}, 2, 3)) || e.IsContiguous() {
		t.Fatalf("Expand: %v", e)
	}
	col := New([]float32{1, 2}, 2, 1).Expand(-1, 3)
	if !col.Equal(New([]float32{1, 1, 1, 2, 2, 2}, 2, 3)) {
		t.Fatalf("Expand -1: %v", col)
	}
	sq := Zeros(1, 3, 1, 2).Squeeze()
	if !sq.Shape().Equal(Shape{3, 2}) {
		t.Fatalf("Squeeze: %v", sq.Shape())
	}
	if !Zeros(1, 3, 1).Squeeze(0).Shape().Equal(Shape{3, 1}) {
		t.Fatal("Squeeze dim")
	}
	if !Zeros(3, 2).Unsqueeze(1).Shape().Equal(Shape{3, 1, 2}) || !Zeros(3).Unsqueeze(-1).Shape().Equal(Shape{3, 1}) {
		t.Fatal("Unsqueeze")
	}
	a, b := New([]float32{1, 2}, 1, 2), New([]float32{3, 4, 5, 6}, 2, 2)
	if !Cat(0, a, b).Equal(New([]float32{1, 2, 3, 4, 5, 6}, 3, 2)) {
		t.Fatalf("Cat 0: %v", Cat(0, a, b))
	}
	if !Cat(1, b, b.T()).Equal(New([]float32{3, 4, 3, 5, 5, 6, 4, 6}, 2, 4)) {
		t.Fatalf("Cat 1: %v", Cat(1, b, b.T()))
	}
	if st := Stack(0, b, b); !st.Shape().Equal(Shape{2, 2, 2}) || st.At(1, 1, 0) != 5 {
		t.Fatalf("Stack: %v", st)
	}
	if st := Stack(-1, New([]float32{1, 2}, 2), New([]float32{3, 4}, 2)); !st.Equal(New([]float32{1, 3, 2, 4}, 2, 2)) {
		t.Fatalf("Stack -1: %v", st)
	}
	expectError(t, "reshape size", func() { x.Reshape(5, 5) })
	expectError(t, "reshape two -1", func() { x.Reshape(-1, -1) })
	expectError(t, "T 3D", func() { x.T() })
	expectError(t, "permute dup", func() { x.Permute(0, 0, 1) })
	expectError(t, "squeeze non-1", func() { x.Squeeze(0) })
	expectError(t, "expand", func() { m.Expand(3, 3) })
	expectError(t, "narrow", func() { x.Narrow(0, 1, 5) })
	expectError(t, "cat", func() { Cat(0, a, Zeros(2, 3)) })
}

// naiveBinary applies f with NumPy broadcasting using only At.
func naiveBinary(x, y *Tensor, f func(a, b float32) float32) *Tensor {
	shape := broadcastShapes("test", x.shape, y.shape)
	out := Zeros(shape...)
	idx := make([]int, len(shape))
	pick := func(t *Tensor) float32 {
		ti := make([]int, len(t.shape))
		off := len(shape) - len(t.shape)
		for d := range ti {
			if t.shape[d] == 1 {
				ti[d] = 0
			} else {
				ti[d] = idx[off+d]
			}
		}
		return t.At(ti...)
	}
	for i := 0; i < out.size; i++ {
		rem := i
		for d := len(shape) - 1; d >= 0; d-- {
			idx[d] = rem % shape[d]
			rem /= shape[d]
		}
		out.data[i] = f(pick(x), pick(y))
	}
	return out
}

func TestBroadcastBinaryOps(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	cases := [][2]Shape{
		{{}, {}}, {{3}, {}}, {{}, {3}}, {{4, 5}, {4, 5}}, {{4, 5}, {5}}, {{4, 5}, {1, 5}}, {{4, 5}, {4, 1}},
		{{5}, {4, 5}}, {{4, 1}, {1, 5}}, {{2, 3, 4}, {3, 1}}, {{2, 1, 4}, {3, 1}}, {{1}, {2, 3}}, {{2, 3}, {1}},
		{{6, 7, 8}, {6, 7, 8}}, {{1, 1, 1}, {2, 2, 2}}, {{0, 3}, {3}}, {{300}, {300}}, {{20, 300}, {300}},
	}
	ops := []struct {
		name string
		op   func(a, b *Tensor) *Tensor
		f    func(a, b float32) float32
	}{
		{"Add", (*Tensor).Add, func(a, b float32) float32 { return a + b }},
		{"Sub", (*Tensor).Sub, func(a, b float32) float32 { return a - b }},
		{"Mul", (*Tensor).Mul, func(a, b float32) float32 { return a * b }},
		{"Div", (*Tensor).Div, func(a, b float32) float32 { return a / b }},
		{"Maximum", (*Tensor).Maximum, func(a, b float32) float32 { return max(a, b) }},
		{"Minimum", (*Tensor).Minimum, func(a, b float32) float32 { return min(a, b) }},
	}
	for _, c := range cases {
		x := UniformFrom(rng, 1, 2, c[0]...)
		y := UniformFrom(rng, 1, 2, c[1]...)
		for _, transposed := range []bool{false, true} {
			xx, yy := x, y
			if transposed && x.Dims() >= 2 {
				xx = x.Transpose(0, 1).Contiguous().Transpose(0, 1) // same values, strided storage
			}
			if transposed && y.Dims() >= 2 {
				yy = y.Transpose(0, 1).Contiguous().Transpose(0, 1)
			}
			for _, op := range ops {
				got := op.op(xx, yy)
				want := naiveBinary(x, y, op.f)
				assertClose(t, op.name+" "+c[0].String()+" "+c[1].String(), got, want, 1e-6)
			}
		}
	}
	expectError(t, "broadcast", func() { Zeros(2, 3).Add(Zeros(4)) })
}

func TestScalarAndUnaryOps(t *testing.T) {
	x := New([]float32{-2, -0.5, 0, 0.5, 1, 2}, 2, 3)
	check := func(name string, got *Tensor, f func(v float64) float64) {
		t.Helper()
		want := Zeros(2, 3)
		for i, v := range x.data {
			want.data[i] = float32(f(float64(v)))
		}
		assertClose(t, name, got, want, 1e-6)
	}
	check("AddScalar", x.AddScalar(1.5), func(v float64) float64 { return v + 1.5 })
	check("SubScalar", x.SubScalar(1.5), func(v float64) float64 { return v - 1.5 })
	check("MulScalar", x.MulScalar(-3), func(v float64) float64 { return v * -3 })
	check("DivScalar", x.DivScalar(4), func(v float64) float64 { return v / 4 })
	check("Neg", x.Neg(), func(v float64) float64 { return -v })
	check("Exp", x.Exp(), math.Exp)
	check("Square", x.Square(), func(v float64) float64 { return v * v })
	check("Abs", x.Abs(), math.Abs)
	check("Tanh", x.Tanh(), math.Tanh)
	check("Sigmoid", x.Sigmoid(), func(v float64) float64 { return 1 / (1 + math.Exp(-v)) })
	check("ReLU", x.ReLU(), func(v float64) float64 { return math.Max(v, 0) })
	check("Clamp", x.Clamp(-1, 0.75), func(v float64) float64 { return math.Min(math.Max(v, -1), 0.75) })
	check("Pow3", x.Pow(3), func(v float64) float64 { return v * v * v })
	check("GELU", x.GELU(), func(v float64) float64 {
		return 0.5 * v * (1 + math.Tanh(0.7978845608028654*(v+0.044715*v*v*v)))
	})
	pos := x.Abs().AddScalar(0.5)
	posCheck := func(name string, got *Tensor, f func(v float64) float64) {
		t.Helper()
		want := Zeros(2, 3)
		for i, v := range pos.data {
			want.data[i] = float32(f(float64(v)))
		}
		assertClose(t, name, got, want, 1e-6)
	}
	posCheck("Log", pos.Log(), math.Log)
	posCheck("Sqrt", pos.Sqrt(), math.Sqrt)
	posCheck("Pow0.5", pos.Pow(0.5), math.Sqrt)
	posCheck("Pow-1", pos.Pow(-1), func(v float64) float64 { return 1 / v })
	// unary ops on strided views
	xt := x.T()
	assertClose(t, "Exp strided", xt.Exp(), x.Exp().T(), 1e-6)
}

func TestReductions(t *testing.T) {
	x := Arange(0, 24, 1).Reshape(2, 3, 4)
	if s := x.Sum(); s.Dims() != 0 || s.Item() != 276 {
		t.Fatalf("Sum all: %v", s)
	}
	if m := x.Mean(); m.Item() != 11.5 {
		t.Fatalf("Mean all: %v", m)
	}
	assertClose(t, "Sum 0", x.Sum(0), New([]float32{12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34}, 3, 4), 0)
	assertClose(t, "Sum 1", x.Sum(1), New([]float32{12, 15, 18, 21, 48, 51, 54, 57}, 2, 4), 0)
	assertClose(t, "Sum -1", x.Sum(-1), New([]float32{6, 22, 38, 54, 70, 86}, 2, 3), 0)
	assertClose(t, "Sum 0,2", x.Sum(0, 2), New([]float32{60, 92, 124}, 3), 0)
	assertClose(t, "Sum 2,0", x.Sum(2, 0), New([]float32{60, 92, 124}, 3), 0)
	assertClose(t, "Mean 1", x.Mean(1), New([]float32{4, 5, 6, 7, 16, 17, 18, 19}, 2, 4), 1e-6)
	assertClose(t, "Max 2", x.Max(2), New([]float32{3, 7, 11, 15, 19, 23}, 2, 3), 0)
	assertClose(t, "Max 0", x.Max(0), Arange(12, 24, 1).Reshape(3, 4), 0)
	assertClose(t, "Min 1", x.Min(1), New([]float32{0, 1, 2, 3, 12, 13, 14, 15}, 2, 4), 0)
	if x.Max().Item() != 23 || x.Min().Item() != 0 {
		t.Fatal("Max/Min all")
	}
	// strided input
	assertClose(t, "Sum transposed", x.Transpose(0, 2).Sum(2), x.Sum(0).T(), 0)
	// large reductions exercise the parallel paths
	big := Ones(1000, 1000)
	if s := big.Sum().Item(); s != 1e6 {
		t.Fatalf("big Sum: %v", s)
	}
	assertClose(t, "big Sum 0", big.Sum(0), Full(1000, 1000), 0)
	assertClose(t, "big Sum 1", big.Sum(1), Full(1000, 1000), 0)
	assertClose(t, "big Max 0", Arange(0, 1e6, 1).Reshape(1000, 1000).Max(0), Arange(999000, 1e6, 1), 0)
	v := New([]float32{1, 2, 3, 4}, 2, 2)
	assertClose(t, "Var", v.Var(), Scalar(1.25), 1e-6)
	assertClose(t, "Std 1", v.Std(1), Full(0.5, 2), 1e-6)
	am := New([]float32{1, 5, 3, 9, 2, 4}, 2, 3).Argmax(1)
	if am[0] != 1 || am[1] != 0 {
		t.Fatalf("Argmax: %v", am)
	}
	am = New([]float32{1, 5, 3, 9, 2, 4}, 2, 3).Argmax(0)
	if am[0] != 1 || am[1] != 0 || am[2] != 1 {
		t.Fatalf("Argmax 0: %v", am)
	}
	expectError(t, "dim range", func() { x.Sum(3) })
	expectError(t, "dup dims", func() { x.Sum(1, 1) })
	expectError(t, "empty max", func() { Zeros(0, 3).Max(0) })
}

func naiveMatMul(a, b *Tensor) *Tensor {
	m, k, n := a.shape[0], a.shape[1], b.shape[1]
	out := Zeros(m, n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var s float64
			for p := 0; p < k; p++ {
				s += float64(a.At(i, p)) * float64(b.At(p, j))
			}
			out.data[i*n+j] = float32(s)
		}
	}
	return out
}

func TestMatMul(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 6))
	for _, s := range [][3]int{{1, 1, 1}, {2, 3, 4}, {7, 5, 9}, {64, 64, 64}, {33, 17, 65}, {1, 40, 50}, {50, 40, 1}, {130, 70, 90}} {
		m, k, n := s[0], s[1], s[2]
		a := RandnFrom(rng, m, k)
		b := RandnFrom(rng, k, n)
		want := naiveMatMul(a, b)
		assertClose(t, "MatMul", a.MatMul(b), want, 1e-4)
		// transposed views are consumed without copying
		at := RandnFrom(rng, k, m).T()
		bt := RandnFrom(rng, n, k).T()
		assertClose(t, "MatMul AᵀB", at.MatMul(b), naiveMatMul(at, b), 1e-4)
		assertClose(t, "MatMul ABᵀ", a.MatMul(bt), naiveMatMul(a, bt), 1e-4)
		assertClose(t, "MatMul AᵀBᵀ", at.MatMul(bt), naiveMatMul(at, bt), 1e-4)
	}
	// vector forms
	v := New([]float32{1, 2, 3}, 3)
	m := New([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	if d := v.Dot(v); d.Dims() != 0 || d.Item() != 14 {
		t.Fatalf("Dot: %v", d)
	}
	assertClose(t, "M·v", m.MatMul(v), New([]float32{14, 32}, 2), 0)
	assertClose(t, "v·Mᵀ", v.MatMul(m.T()), New([]float32{14, 32}, 2), 0)
	assertClose(t, "Outer", New([]float32{1, 2}, 2).Outer(v), New([]float32{1, 2, 3, 2, 4, 6}, 2, 3), 0)
	// batched with broadcasting
	a := RandnFrom(rng, 3, 2, 4, 5)
	b := RandnFrom(rng, 2, 5, 6)
	got := a.MatMul(b)
	if !got.Shape().Equal(Shape{3, 2, 4, 6}) {
		t.Fatalf("batched shape %v", got.Shape())
	}
	for i := 0; i < 3; i++ {
		for j := 0; j < 2; j++ {
			want := naiveMatMul(a.Select(0, i).Select(0, j), b.Select(0, j))
			assertClose(t, "batched", got.Select(0, i).Select(0, j), want, 1e-4)
		}
	}
	// batched × 2-D folds the batch into M
	c := RandnFrom(rng, 5, 7)
	got = a.MatMul(c)
	for i := 0; i < 3; i++ {
		for j := 0; j < 2; j++ {
			assertClose(t, "batched 2D", got.Select(0, i).Select(0, j), naiveMatMul(a.Select(0, i).Select(0, j), c), 1e-4)
		}
	}
	// batched with strided (transposed) batch operands
	bt := RandnFrom(rng, 2, 6, 5).Transpose(1, 2)
	got = a.MatMul(bt)
	for i := 0; i < 3; i++ {
		for j := 0; j < 2; j++ {
			assertClose(t, "batched transposed", got.Select(0, i).Select(0, j), naiveMatMul(a.Select(0, i).Select(0, j), bt.Select(0, j)), 1e-4)
		}
	}
	// many small products run in parallel over the batch
	x := RandnFrom(rng, 64, 8, 8)
	y := RandnFrom(rng, 64, 8, 8)
	got = x.MatMul(y)
	for i := 0; i < 64; i += 7 {
		assertClose(t, "small batch", got.Select(0, i), naiveMatMul(x.Select(0, i), y.Select(0, i)), 1e-4)
	}
	expectError(t, "shape", func() { Zeros(2, 3).MatMul(Zeros(2, 3)) })
	expectError(t, "scalar", func() { Scalar(1).MatMul(Zeros(2)) })
	expectError(t, "batch broadcast", func() { Zeros(2, 3, 4).MatMul(Zeros(3, 4, 5)) })
}

func TestSoftmaxFamily(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 8))
	x := RandnFrom(rng, 5, 7).MulScalar(3)
	sm := x.Softmax(1)
	assertClose(t, "softmax rows sum to 1", sm.Sum(1), Ones(5), 1e-5)
	assertClose(t, "logsoftmax", x.LogSoftmax(1), sm.Log(), 1e-4)
	assertClose(t, "softmax dim 0", x.Softmax(0).Sum(0), Ones(7), 1e-5)
	assertClose(t, "softmax -1", x.Softmax(-1), sm, 0)
	// composite reference
	e := x.Sub(x.Max(1).Unsqueeze(1)).Exp()
	assertClose(t, "softmax composite", e.Div(e.Sum(1).Unsqueeze(1)), sm, 1e-5)
	// cross entropy equals -mean(logsoftmax[target])
	targets := []int{0, 3, 6, 1, 1}
	ls := x.LogSoftmax(1)
	var want float32
	for i, c := range targets {
		want -= ls.At(i, c)
	}
	want /= 5
	if got := CrossEntropy(x, targets).Item(); !approx(got, want, 1e-5) {
		t.Fatalf("CrossEntropy %v want %v", got, want)
	}
	p, q := RandnFrom(rng, 3, 4), RandnFrom(rng, 3, 4)
	if got, w := MSELoss(p, q).Item(), p.Sub(q).Square().Mean().Item(); !approx(got, w, 1e-5) {
		t.Fatalf("MSELoss %v want %v", got, w)
	}
	// layer norm rows have mean 0, var 1 with identity affine
	ln := LayerNorm(x, Ones(7), Zeros(7), 1e-5)
	assertClose(t, "layernorm mean", ln.Mean(1), Zeros(5), 1e-5)
	assertClose(t, "layernorm var", ln.Var(1), Ones(5), 1e-3)
	expectError(t, "CE targets", func() { CrossEntropy(x, []int{0}) })
	expectError(t, "CE range", func() { CrossEntropy(x, []int{0, 1, 2, 3, 7}) })
	expectError(t, "LN size", func() { LayerNorm(x, Ones(3), Zeros(3), 1e-5) })
}

func TestInPlace(t *testing.T) {
	x := Ones(2, 3)
	x.AddInPlace(New([]float32{1, 2, 3}, 3)).MulScalarInPlace(2)
	assertClose(t, "AddInPlace+MulScalar", x, New([]float32{4, 6, 8, 4, 6, 8}, 2, 3), 0)
	x.SubInPlace(Scalar(1)).DivInPlace(Full(2, 2, 3)).MulInPlace(New([]float32{1, 0, 1}, 3))
	assertClose(t, "Sub/Div/Mul", x, New([]float32{1.5, 0, 3.5, 1.5, 0, 3.5}, 2, 3), 0)
	x.AddScaledInPlace(Ones(2, 3), -0.5)
	assertClose(t, "axpy", x, New([]float32{1, -0.5, 3, 1, -0.5, 3}, 2, 3), 0)
	x.Fill(2).CopyFrom(Arange(0, 6, 1).Reshape(2, 3))
	assertClose(t, "Fill+CopyFrom", x, Arange(0, 6, 1).Reshape(2, 3), 0)
	x.T().Fill(9) // strided view fill
	assertClose(t, "strided Fill", x, Full(9, 2, 3), 0)
	x.Set(1, 1, 2)
	if x.At(1, 2) != 1 {
		t.Fatal("Set")
	}
	x.Narrow(1, 0, 1).Zero()
	assertClose(t, "Zero view", x, New([]float32{0, 9, 9, 0, 9, 1}, 2, 3), 0)
	w := Ones(3).SetRequiresGrad(true)
	expectError(t, "in-place on grad tensor", func() { w.AddInPlace(Ones(3)) })
	NoGrad(func() { w.AddInPlace(Ones(3)) })
	if w.At(0) != 2 {
		t.Fatal("in-place under NoGrad")
	}
	expectError(t, "in-place broadcast", func() { Ones(3).AddInPlace(Ones(2, 3)) })
}

func TestThreadsGiveSameResults(t *testing.T) {
	rng := rand.New(rand.NewPCG(9, 10))
	a := RandnFrom(rng, 257, 129)
	b := RandnFrom(rng, 129, 301)
	c := RandnFrom(rng, 301)
	compute := func() *Tensor {
		return a.MatMul(b).Add(c).Tanh().Softmax(1).Sum(0)
	}
	saved := Threads()
	defer SetThreads(saved)
	SetThreads(1)
	want := compute()
	for _, n := range []int{2, 3, 8, 0} {
		SetThreads(n)
		assertClose(t, "threads", compute(), want, 1e-5)
	}
	if Backend() == "" {
		t.Fatal("Backend empty")
	}
}

func TestFormat(t *testing.T) {
	m := New([]float32{1, 2.5, -3, 4, 5, 6}, 2, 3)
	want := "[[  1 2.5  -3]\n [  4   5   6]]"
	if got := m.String(); got != want {
		t.Fatalf("String:\n%s\nwant:\n%s", got, want)
	}
	if got := New([]float32{1, 2, 3}, 3).String(); got != "[1 2 3]" {
		t.Fatalf("1-D: %q", got)
	}
	big := Arange(0, 100, 1)
	if s := big.String(); !strings.Contains(s, "...") || !strings.HasPrefix(s, "[ 0  1  2 ...") {
		t.Fatalf("elision: %q", s)
	}
	if s := Zeros(2, 2, 2).String(); strings.Count(s, "\n") != 4 { // blank line between outer blocks
		t.Fatalf("3-D layout: %q", s)
	}
	if s := Zeros(0, 3).String(); !strings.Contains(s, "shape=[0 3]") {
		t.Fatalf("empty: %q", s)
	}
	if s := m.T().String(); !strings.HasPrefix(s, "[[  1   4]") {
		t.Fatalf("strided: %q", s)
	}
	gs := m.SetRequiresGrad(true).GoString()
	if !strings.Contains(gs, "requiresGrad") || !strings.Contains(gs, "[2 3]") {
		t.Fatalf("GoString: %q", gs)
	}
}

func TestTry(t *testing.T) {
	err := Try(func() { Zeros(2).Add(Zeros(3)) })
	if err == nil || !strings.Contains(err.Error(), "tensor: Add:") {
		t.Fatalf("Try: %v", err)
	}
	if Try(func() {}) != nil {
		t.Fatal("Try without panic")
	}
	defer func() {
		if r := recover(); r != "other" {
			t.Fatalf("foreign panic must propagate, got %v", r)
		}
	}()
	_ = Try(func() { panic("other") })
}

func TestRowsGatherAndGradient(t *testing.T) {
	x := Arange(0, 12, 1).Reshape(4, 3).SetRequiresGrad(true)
	r := x.Rows([]int{3, 0, 3})
	if !r.Shape().Equal(Shape{3, 3}) || r.At(0, 1) != 10 || r.At(1, 2) != 2 {
		t.Fatalf("Rows gathered %v", r)
	}
	r.Sum().Backward()
	// row 3 taken twice, row 0 once, rows 1 and 2 never
	want := []float32{1, 1, 1, 0, 0, 0, 0, 0, 0, 2, 2, 2}
	if got := x.Grad().Float32s(); !Equalf(got, want) {
		t.Fatalf("Rows gradient %v, want %v", got, want)
	}
	if err := Try(func() { x.Rows([]int{4}) }); err == nil {
		t.Fatal("out-of-range index accepted")
	}
}
