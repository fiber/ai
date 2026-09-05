//go:build arm64 && !darwin

package kernel

// Apple's AMX coprocessor exists only under macOS.
var amxImpl *impl
