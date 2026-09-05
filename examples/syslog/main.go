// Example: the syslog project of the network workbook on a generated day
// of logs. Lines become templates (logtemplate), templates get embedding
// vectors (faked here by a deterministic random projection of their
// tokens; in production EmbeddingGemma via Ollama), the vectors are
// clustered (cluster), and the result is the dashboard: clusters with
// counts, plus the templates that fit no cluster.
package main

import (
	"fmt"
	"hash/fnv"
	"math/rand/v2"
	"time"

	"github.com/fiber/ai/cluster"
	"github.com/fiber/ai/logtemplate"
	"github.com/fiber/ai/tensor"
)

const dim = 768

// embed fakes an embedding: every token contributes a fixed pseudo-random
// direction, so templates sharing words land near each other. A real
// system calls the embedding model once per template here.
func embed(tokens []string) []float32 {
	v := make([]float32, dim)
	for _, tok := range tokens {
		if tok == logtemplate.Wildcard || tok == logtemplate.Num || tok == logtemplate.IP {
			continue
		}
		h := fnv.New64a()
		h.Write([]byte(tok))
		r := rand.New(rand.NewPCG(h.Sum64(), 1))
		for i := range v {
			v[i] += float32(r.NormFloat64())
		}
	}
	return v
}

func main() {
	r := rand.New(rand.NewPCG(42, 0))
	lines := generate(r, 2_000_000)

	// 1. templates
	start := time.Now()
	m := logtemplate.NewMiner()
	ids := make([]int32, len(lines))
	for i, l := range lines {
		ids[i] = int32(m.Add(l))
	}
	mined := time.Since(start)
	fmt.Printf("%d lines → %d templates in %v (%.0f lines/s)\n\n", len(lines), m.Len(), mined.Round(time.Millisecond), float64(len(lines))/mined.Seconds())

	// 2. one vector per template
	tmpls := m.Templates()
	data := make([]float32, 0, len(tmpls)*dim)
	for _, t := range tmpls {
		data = append(data, embed(t.Tokens)...)
	}
	x := cluster.Normalize(tensor.New(data, len(tmpls), dim))

	// 3. clusters
	k := 8
	res := cluster.KMeans(x, k, 30, 1)
	fmt.Printf("k-means: %d templates into %d clusters, %d iterations, mean distance %.3f\n\n", len(tmpls), k, res.Iterations, res.Inertia())

	// 4. the dashboard: per cluster its volume and its most frequent template
	type row struct {
		cluster, lines int
		example        string
	}
	rows := make([]row, k)
	for i, t := range tmpls {
		c := res.Assignment[i]
		rows[c].cluster = c
		rows[c].lines += t.Count
		if rows[c].example == "" {
			rows[c].example = t.String() // templates are sorted by count
		}
	}
	fmt.Println("cluster  lines/day  most frequent template")
	for _, rw := range rows {
		ex := rw.example
		if len(ex) > 60 {
			ex = ex[:57] + "..."
		}
		fmt.Printf("%7d  %9d  %s\n", rw.cluster, rw.lines, ex)
	}

	// 5. novelty: a template that fits no cluster well is what a person
	// should look at. Here we fake a new kind of message arriving.
	novel := []string{
		"kernel: BUG: soft lockup - CPU#3 stuck for 22s! [kworker/3:1:9821]",
		"sshd[9912]: Accepted publickey for sven from 10.9.9.9 port 5555 ssh2",
	}
	fmt.Println("\nnew lines against the day's clusters:")
	for _, l := range novel {
		id := m.Add(l)
		t := m.Template(id)
		v := cluster.Normalize(tensor.New(embed(t.Tokens), 1, dim))
		idx, sim := cluster.Nearest(v, res.Centres)
		verdict := "known kind"
		if t.Count == 1 {
			verdict = "new template"
		}
		if sim[0] < 0.5 {
			verdict += ", fits no cluster: show to a person"
		}
		fmt.Printf("  %-52.52s → cluster %d, similarity %.2f: %s\n", l, idx[0], sim[0], verdict)
	}
}

// generate fakes a day of syslog from a dozen message shapes.
func generate(r *rand.Rand, n int) []string {
	users := []string{"sven", "anna", "root", "backup", "www-data", "monitor"}
	hosts := []string{"fw-01", "core-sw-1", "core-sw-2", "web-03", "db-01", "vpn-01"}
	ip := func() string { return fmt.Sprintf("10.%d.%d.%d", r.IntN(255), r.IntN(255), r.IntN(255)) }
	out := make([]string, n)
	for i := range out {
		h := hosts[r.IntN(len(hosts))]
		switch r.IntN(14) {
		case 0, 1, 2:
			out[i] = fmt.Sprintf("%s sshd[%d]: Accepted publickey for %s from %s port %d ssh2", h, r.IntN(60000), users[r.IntN(len(users))], ip(), r.IntN(65535))
		case 3:
			out[i] = fmt.Sprintf("%s sshd[%d]: Failed password for invalid user %s from %s port %d ssh2", h, r.IntN(60000), users[r.IntN(len(users))], ip(), r.IntN(65535))
		case 4, 5, 6:
			out[i] = fmt.Sprintf("%s firewall: DROP IN=eth%d OUT= SRC=%s DST=%s PROTO=TCP SPT=%d DPT=%d", h, r.IntN(4), ip(), ip(), r.IntN(65535), r.IntN(65535))
		case 7:
			out[i] = fmt.Sprintf("%s dhcpd: DHCPACK on %s to 3c:22:fb:%02x:%02x:%02x via eth%d", h, ip(), r.IntN(256), r.IntN(256), r.IntN(256), r.IntN(4))
		case 8, 9:
			out[i] = fmt.Sprintf("%s named[%d]: client %s#%d: query: host%d.example.com IN A + (%s)", h, r.IntN(60000), ip(), r.IntN(65535), r.IntN(500), ip())
		case 10:
			out[i] = fmt.Sprintf("%s %%LINK-3-UPDOWN: Interface GigabitEthernet%d/0/%d, changed state to %s", h, r.IntN(3)+1, r.IntN(48), []string{"up", "down"}[r.IntN(2)])
		case 11:
			out[i] = fmt.Sprintf("%s systemd[1]: Started session %d of user %s.", h, r.IntN(9000), users[r.IntN(len(users))])
		case 12:
			out[i] = fmt.Sprintf("%s nginx: %s - - GET /api/v%d/items/%d HTTP/1.1 200 %d", h, ip(), r.IntN(3)+1, r.IntN(100000), r.IntN(20000))
		default:
			out[i] = fmt.Sprintf("%s CRON[%d]: (%s) CMD (/usr/bin/backup --job %d)", h, r.IntN(60000), users[r.IntN(len(users))], r.IntN(40))
		}
	}
	return out
}
