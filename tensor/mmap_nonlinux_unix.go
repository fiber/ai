//go:build unix && !linux

package tensor

func adviseHuge([]byte) {}
