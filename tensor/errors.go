package tensor

import "fmt"

// Error is the panic value raised for invalid shapes or arguments.
type Error struct {
	Op  string // the operation that failed, e.g. "MatMul"
	Msg string
}

func (e *Error) Error() string { return "tensor: " + e.Op + ": " + e.Msg }

func fail(op, format string, args ...any) {
	panic(&Error{Op: op, Msg: fmt.Sprintf(format, args...)})
}

// Try runs fn and converts a tensor *Error panic into a returned error.
// Other panics propagate unchanged.
func Try(fn func()) (err error) {
	defer func() {
		if r := recover(); r != nil {
			if e, ok := r.(*Error); ok {
				err = e
				return
			}
			panic(r)
		}
	}()
	fn()
	return nil
}
