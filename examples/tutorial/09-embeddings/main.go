// Tutorial chapter 9: embeddings — a model that turns text into geometry.
// No training here: a pretrained encoder, a ruler, and what the ruler says.
package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"time"

	"github.com/fiber/ai/cluster"
	"github.com/fiber/ai/models/gemma"
)

// Sixteen syslog templates as an operator would store them after the
// variable parts (addresses, names, numbers) have been recognised.
var templates = []string{
	"interface GigabitEthernet0/1 changed state to down",
	"interface GigabitEthernet0/1 changed state to up",
	"%BGP-5-ADJCHANGE: neighbor 10.0.0.1 Down",
	"%BGP-5-ADJCHANGE: neighbor 10.0.0.1 Up",
	"OSPF neighbor 10.0.0.3 on Vlan10 went from FULL to DOWN",
	"authentication failure for user admin from 203.0.113.7",
	"accepted password for user backup from 10.0.0.9",
	"session opened for user root by cron",
	"CPU utilization exceeded 90 percent on router core-1",
	"memory usage at 95 percent on firewall fw-2",
	"disk /var reached 92 percent on host web-3",
	"DHCP lease assigned 10.0.0.55 to aa:bb:cc:dd:ee:ff",
	"NTP time synchronization lost with server 10.0.0.2",
	"fan speed sensor reading abnormal on chassis 1",
	"power supply 2 failed on switch access-7",
	"backup job nightly-full completed successfully",
}

func main() {
	model := flag.String("model", "", "path to the embeddinggemma-300m directory (default $FIBERAI_MODELS/embeddinggemma-300m)")
	flag.Parse()
	dir := *model
	if dir == "" {
		if root := os.Getenv("FIBERAI_MODELS"); root != "" {
			dir = filepath.Join(root, "embeddinggemma-300m")
		}
	}
	if dir == "" {
		fmt.Println("this chapter needs the EmbeddingGemma weights: see docs/manual/models.md, then")
		fmt.Println("set FIBERAI_MODELS to the directory holding embeddinggemma-300m, or pass -model")
		return
	}

	start := time.Now()
	m, err := gemma.Load(dir)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	defer m.Close()
	fmt.Printf("EmbeddingGemma loaded in %v: %d layers, vectors of %d numbers\n", time.Since(start).Round(time.Millisecond), m.Config().NumHiddenLayers, m.Dim())

	// One vector per template. The "document" prompt is part of how the
	// model was trained; queries get the "query" prompt below.
	start = time.Now()
	vecs, err := m.Embed(templates, gemma.Prompt("document"))
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Printf("%d templates embedded in %v\n\n", len(templates), time.Since(start).Round(time.Millisecond))

	// A vector is just numbers. The first five of the first template:
	fmt.Printf("%q\n  starts %.3f %.3f %.3f %.3f %.3f ... (%d numbers, length 1)\n\n", templates[0], vecs[0][0], vecs[0][1], vecs[0][2], vecs[0][3], vecs[0][4], len(vecs[0]))

	// The ruler: cosine similarity, 1 for the same direction, 0 for
	// unrelated. Every pair of templates, the closest and the farthest.
	type pair struct {
		a, b int
		sim  float64
	}
	var pairs []pair
	for i := range templates {
		for j := i + 1; j < len(templates); j++ {
			pairs = append(pairs, pair{i, j, float64(cluster.Cosine(vecs[i], vecs[j]))})
		}
	}
	sort.Slice(pairs, func(x, y int) bool { return pairs[x].sim > pairs[y].sim })
	fmt.Println("closest pairs:")
	for _, p := range pairs[:3] {
		fmt.Printf("  %.3f  %s\n         %s\n", p.sim, templates[p.a], templates[p.b])
	}
	fmt.Println("farthest pairs:")
	for _, p := range pairs[len(pairs)-2:] {
		fmt.Printf("  %.3f  %s\n         %s\n", p.sim, templates[p.a], templates[p.b])
	}

	// Topic, not polarity: "down" and "up" of the same interface are
	// nearer to each other than to anything else.
	fmt.Printf("\ninterface down vs interface up: %.3f\ninterface down vs BGP neighbor down: %.3f\ninterface down vs backup completed: %.3f\n",
		cluster.Cosine(vecs[0], vecs[1]), cluster.Cosine(vecs[0], vecs[2]), cluster.Cosine(vecs[0], vecs[15]))

	// Search: a question in plain words against the templates.
	queries := []string{"a network link went down", "someone failed to log in", "the device is running out of resources"}
	qv, err := m.Embed(queries, gemma.Prompt("query"))
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	for qi, q := range queries {
		idx := make([]int, len(templates))
		for i := range idx {
			idx[i] = i
		}
		sort.Slice(idx, func(a, b int) bool {
			return cluster.Cosine(qv[qi], vecs[idx[a]]) > cluster.Cosine(qv[qi], vecs[idx[b]])
		})
		fmt.Printf("\n%q\n", q)
		for _, i := range idx[:3] {
			fmt.Printf("  %.3f  %s\n", cluster.Cosine(qv[qi], vecs[i]), templates[i])
		}
	}
}
