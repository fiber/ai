//go:build !amd64 && !arm64

package kernel

func candidates() []*impl { return nil }
func allImpls() []*impl   { return nil }
