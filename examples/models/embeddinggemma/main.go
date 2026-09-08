// Command embeddinggemma embeds a set of syslog templates and free-text
// queries with EmbeddingGemma and prints, for each query, the nearest
// templates by cosine similarity. It is the worked example for the models
// manual page.
//
// It needs the model files in a directory (config.json, tokenizer.json, the
// weight safetensors and the pooling/Dense configs), downloaded once from
// Hugging Face:
//
//	huggingface-cli download google/embeddinggemma-300m --local-dir embeddinggemma-300m
//
// Run with:
//
//	go run ./examples/models/embeddinggemma -model path/to/embeddinggemma-300m
package main

import (
	"flag"
	"fmt"
	"log"
	"sort"
	"time"

	"github.com/fiber/ai/cluster"
	"github.com/fiber/ai/models/gemma"
)

func main() {
	model := flag.String("model", "", "path to the embeddinggemma-300m directory")
	flag.Parse()
	if *model == "" {
		log.Fatal("give -model path/to/embeddinggemma-300m")
	}

	start := time.Now()
	m, err := gemma.Load(*model)
	if err != nil {
		log.Fatal(err)
	}
	defer m.Close()
	fmt.Printf("loaded in %v (hidden %d, %d layers, embedding dim %d)\n\n",
		time.Since(start).Round(time.Millisecond),
		m.Config().HiddenSize, m.Config().NumHiddenLayers, m.Dim())

	// The corpus: syslog message templates an operator might store.
	templates := []string{
		"interface GigabitEthernet0/1 changed state to down",
		"interface GigabitEthernet0/1 changed state to up",
		"%BGP-5-ADJCHANGE: neighbor 10.0.0.1 Up",
		"%BGP-5-ADJCHANGE: neighbor 10.0.0.1 Down",
		"authentication failure for user admin from 203.0.113.7",
		"accepted password for user backup from 10.0.0.9",
		"CPU utilization exceeded 90 percent on router core-1",
		"memory usage at 95 percent on firewall fw-2",
		"DHCP lease assigned 10.0.0.55 to aa:bb:cc:dd:ee:ff",
		"disk /var reached 92 percent on host web-3",
		"NTP time synchronization lost with server 10.0.0.2",
		"fan speed sensor reading abnormal on chassis 1",
	}
	// Documents get the retrieval "document" prompt; queries the "query"
	// prompt. Prompts are part of how EmbeddingGemma was trained.
	docs, err := m.Embed(templates, gemma.Prompt("document"))
	if err != nil {
		log.Fatal(err)
	}

	queries := []string{
		"a network link went down",
		"someone failed to log in",
		"the device is running out of resources",
		"routing neighbor relationship changed",
	}
	qemb, err := m.Embed(queries, gemma.Prompt("query"))
	if err != nil {
		log.Fatal(err)
	}

	for qi, q := range queries {
		type hit struct {
			text string
			sim  float64
		}
		hits := make([]hit, len(templates))
		for ti := range templates {
			hits[ti] = hit{templates[ti], float64(cluster.Cosine(qemb[qi], docs[ti]))}
		}
		sort.Slice(hits, func(a, b int) bool { return hits[a].sim > hits[b].sim })
		fmt.Printf("query: %s\n", q)
		for _, h := range hits[:3] {
			fmt.Printf("  %.3f  %s\n", h.sim, h.text)
		}
		fmt.Println()
	}
}
