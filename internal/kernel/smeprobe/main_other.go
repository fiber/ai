//go:build !(darwin && arm64)

// Command smeprobe is an Apple Silicon (M4) experiment; on other platforms
// it only says so, so that go build ./... and go test ./... stay green.
package main

import "fmt"

func main() {
	fmt.Println("smeprobe: the SME probe runs on Apple Silicon (darwin/arm64) only")
}
