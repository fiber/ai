// Tutorial chapter 18: a text classifier on frozen embeddings — the
// pattern most text goes through in a Go service. EmbeddingGemma turns
// each line into a vector once; a head of a few thousand numbers
// trained in seconds turns the vector into a decision. Measured against
// a bag of words on wordings neither has seen.
package main

import (
	"flag"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/fiber/ai/cluster"
	"github.com/fiber/ai/data"
	"github.com/fiber/ai/metrics"
	"github.com/fiber/ai/models/gemma"
	"github.com/fiber/ai/nn"
	"github.com/fiber/ai/optim"
	"github.com/fiber/ai/tensor"
)

// The categories a line can be routed to, and the shapes each is written
// in. The last two shapes of every category are held out entirely: the
// classifiers never see a single line of them, so those lines test
// whether a classifier understood the category or memorised its words.
var categories = []string{"access", "network", "storage", "jobs", "hardware"}

const heldOut = 2 // shapes per category kept out of training

var shapes = [][]string{
	{ // access
		"sshd[%P]: Accepted publickey for %U from %I port %P ssh2",
		"sshd[%P]: Failed password for invalid user %U from %I port %P ssh2",
		"sudo: %U : TTY=pts/%D ; PWD=/home/%U ; COMMAND=/usr/bin/systemctl restart %S",
		"systemd-logind[%P]: New session %D of user %U.",
		"login[%P]: pam_unix(login:session): session opened for user %U by (uid=0)",
		"sshd[%P]: Received disconnect from %I port %P: 11: disconnected by user",
		"pam_tally2[%P]: account %U locked after %D bad attempts from %I", // held out
		"vsftpd[%P]: CONNECT: Client %I, anonymous login refused",         // held out
	},
	{ // network
		"kernel: eth%D: link is up, 1000 Mbps full duplex",
		"bgpd[%P]: neighbor %I Down: BGP Notification received (hold timer expired)",
		"dhcpd: DHCPACK on %I to %M via eth%D",
		"named[%P]: client %I#%P: query: %H.example.com IN A + (%I)",
		"switch-%D: %%LINK-3-UPDOWN: Interface GigabitEthernet%D/0/%D, changed state to down",
		"ospfd[%P]: adjacency with %I on eth%D changed from Full to Down",
		"ntpd[%P]: no servers reachable, %I unreachable for %D minutes",      // held out
		"firewall: DROP IN=eth%D SRC=%I DST=%I PROTO=TCP SPT=%P DPT=%P",      // held out
	},
	{ // storage
		"kernel: EXT4-fs (sd%L%D): remounting filesystem read-only after error",
		"smartd[%P]: Device /dev/sd%L, %D Currently unreadable (pending) sectors",
		"mdadm[%P]: DegradedArray event detected on md device /dev/md%D",
		"monit[%P]: filesystem /var usage %D%% matches resource limit [usage > 85%%]",
		"nfs: server %H not responding, still trying (mount /export/%S)",
		"kernel: sd %D:0:0:0: [sd%L] I/O error, dev sd%L, sector %P",
		"lvm[%P]: thin pool vg0/pool0 is %D%% full, snapshot %S will be dropped", // held out
		"rsyslogd: no space left on device writing /var/log/%S.log",             // held out
	},
	{ // jobs
		"CRON[%P]: (%U) CMD (/usr/local/bin/backup --job %S)",
		"systemd[1]: Started Daily %S cleanup timer.",
		"backup[%P]: job %S finished: %D files, %D MB, %D seconds",
		"anacron[%P]: Job `cron.daily' terminated (exit status: %D)",
		"logrotate[%P]: rotating /var/log/%S.log, %D files kept",
		"report-gen[%P]: report %S for %D rows written to /srv/reports/%S.pdf",
		"batch[%P]: queue %S drained, %D tasks in %D s, %D retried",    // held out
		"puppet-agent[%P]: Applied catalog in %D.%D seconds (%S run)", // held out
	},
	{ // hardware
		"ipmi: fan %D speed %D RPM below threshold on chassis %D",
		"kernel: CPU%D: Core temperature above threshold, cpu clock throttled",
		"kernel: EDAC MC%D: %D CE memory read error on DIMM_%L%D",
		"ipmi: power supply %D failed on %H, redundancy lost",
		"acpid: battery %D at %D%%, discharging on %H",
		"kernel: pcieport 0000:%D:00.0: AER: Corrected error received",
		"snmpd[%P]: chassis %D intrusion sensor asserted on %H",          // held out
		"kernel: thermal thermal_zone%D: critical temperature %D C reached", // held out
	},
}

// foreign is a sixth kind of line the classifiers are never trained on,
// application errors, to see what a classifier says about something
// from outside its world.
var foreign = []string{
	"orders[%P]: NullPointerException in OrderService.submit at line %D",
	"api[%P]: HTTP 500 on POST /checkout/%S after %D ms",
	"db-pool[%P]: connection pool exhausted, %D waiters, timeout %D ms",
	"postgres[%P]: deadlock detected, process %P waits for ShareLock on transaction %P",
	"cache[%P]: miss rate %D%% over the last %D s on %S",
	"ingest[%P]: JSON parse error at offset %D in message from %H",
	"grpc[%P]: rpc error: code = DeadlineExceeded desc = %S.Get after %D ms",
	"web[%P]: unhandled promise rejection in %S handler: TypeError",
}

// render fills a shape's placeholders: %P a pid or port, %D a small
// number, %I an address, %M a MAC, %U a user, %H a host, %S a service
// or job name, %L a drive letter.
func render(shape string, r *rand.Rand) string {
	users := []string{"sven", "anna", "root", "backup", "www-data", "deploy"}
	hosts := []string{"web-3", "db-1", "edge-7", "nas-2", "build-4"}
	names := []string{"nightly", "metrics", "ledger", "weekly-full", "auth", "billing", "search"}
	var sb strings.Builder
	for i := 0; i < len(shape); i++ {
		if shape[i] != '%' || i+1 == len(shape) {
			sb.WriteByte(shape[i])
			continue
		}
		i++
		switch shape[i] {
		case 'P':
			fmt.Fprint(&sb, 1024+r.IntN(60000))
		case 'D':
			fmt.Fprint(&sb, r.IntN(100))
		case 'I':
			fmt.Fprintf(&sb, "10.%d.%d.%d", r.IntN(255), r.IntN(255), r.IntN(255))
		case 'M':
			fmt.Fprintf(&sb, "3c:22:fb:%02x:%02x:%02x", r.IntN(256), r.IntN(256), r.IntN(256))
		case 'U':
			sb.WriteString(users[r.IntN(len(users))])
		case 'H':
			sb.WriteString(hosts[r.IntN(len(hosts))])
		case 'S':
			sb.WriteString(names[r.IntN(len(names))])
		case 'L':
			sb.WriteByte("abcdef"[r.IntN(6)])
		case '%':
			sb.WriteByte('%')
		}
	}
	return sb.String()
}

// set is a labelled collection of lines.
type set struct {
	lines  []string
	labels []int
}

func (s *set) add(shape string, label, n int, r *rand.Rand) {
	for range n {
		s.lines = append(s.lines, render(shape, r))
		s.labels = append(s.labels, label)
	}
}

// generate builds the three sets: training lines and test lines from
// the seen shapes, and test lines from the held-out shapes.
func generate(r *rand.Rand, perShape int) (train, seen, unseen set) {
	for c, sh := range shapes {
		for i, s := range sh {
			if i >= len(sh)-heldOut {
				unseen.add(s, c, perShape, r)
			} else {
				train.add(s, c, perShape, r)
				seen.add(s, c, perShape/4, r)
			}
		}
	}
	return
}

// Bag of words: which words of the training vocabulary a line contains.
// Words with digits are dropped, which removes the pids, addresses and
// counts that would otherwise be one-off features.
func words(line string) []string {
	f := func(c rune) bool { return !(c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9') }
	var out []string
	for _, w := range strings.FieldsFunc(strings.ToLower(line), f) {
		if strings.IndexAny(w, "0123456789") < 0 {
			out = append(out, w)
		}
	}
	return out
}

func vocabulary(lines []string) map[string]int {
	v := map[string]int{}
	for _, l := range lines {
		for _, w := range words(l) {
			if _, ok := v[w]; !ok {
				v[w] = len(v)
			}
		}
	}
	return v
}

func bag(lines []string, v map[string]int) *tensor.Tensor {
	x := make([]float32, len(lines)*len(v))
	for i, l := range lines {
		for _, w := range words(l) {
			if j, ok := v[w]; ok {
				x[i*len(v)+j] = 1
			}
		}
	}
	return tensor.New(x, len(lines), len(v))
}

func matrix(vecs [][]float32) *tensor.Tensor {
	d := len(vecs[0])
	x := make([]float32, 0, len(vecs)*d)
	for _, v := range vecs {
		x = append(x, v...)
	}
	return tensor.New(x, len(vecs), d)
}

// trainHead fits one linear layer on the features: a few seconds of
// AdamW, the same loop as chapter 7 with a cross-entropy loss.
func trainHead(x *tensor.Tensor, labels []int, epochs int, r *rand.Rand) *nn.Linear {
	head := nn.NewLinear(x.Dim(1), len(categories))
	opt := optim.NewAdamW(head.Params(), 1e-2, 0.01)
	for range epochs {
		for idx := range data.Batches(len(labels), 64, r) {
			yb := make([]int, len(idx))
			for i, j := range idx {
				yb[i] = labels[j]
			}
			loss := tensor.CrossEntropy(head.Forward(x.Rows(idx)), yb)
			opt.ZeroGrad()
			loss.Backward()
			opt.Step()
		}
	}
	return head
}

func predict(head *nn.Linear, x *tensor.Tensor) (pred []int, conf []float32) {
	tensor.NoGrad(func() {
		p := head.Forward(x).Softmax(1)
		pred = p.Argmax(1)
		probs := p.Float32s()
		k := p.Dim(1)
		for i, c := range pred {
			conf = append(conf, probs[i*k+c])
		}
	})
	return
}

func accuracy(pred, truth []int) float64 {
	n := 0
	for i := range pred {
		if pred[i] == truth[i] {
			n++
		}
	}
	return 100 * float64(n) / float64(len(pred))
}

// centroids is the no-training classifier: the mean vector of every
// category, and the nearest one by cosine wins.
func centroids(vecs [][]float32, labels []int) [][]float32 {
	c := make([][]float32, len(categories))
	for i := range c {
		c[i] = make([]float32, len(vecs[0]))
	}
	for i, v := range vecs {
		for j, x := range v {
			c[labels[i]][j] += x
		}
	}
	return c
}

func nearest(cs [][]float32, vecs [][]float32) []int {
	pred := make([]int, len(vecs))
	for i, v := range vecs {
		best := float32(math.Inf(-1))
		for c, cv := range cs {
			if s := cluster.Cosine(v, cv); s > best {
				best, pred[i] = s, c
			}
		}
	}
	return pred
}

func quantile(xs []float32, q float64) float32 {
	s := append([]float32(nil), xs...)
	sort.Slice(s, func(i, j int) bool { return s[i] < s[j] })
	return s[int(q*float64(len(s)-1))]
}

func main() {
	model := flag.String("model", "", "path to the embeddinggemma-300m directory (default $FIBERAI_MODELS/embeddinggemma-300m)")
	perShape := flag.Int("per-shape", 40, "generated lines per shape")
	epochs := flag.Int("epochs", 50, "training epochs for each head")
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

	r := rand.New(rand.NewPCG(18, 0))
	tensor.Seed(18)
	train, seen, unseen := generate(r, *perShape)
	var other set
	for _, s := range foreign {
		other.add(s, -1, *perShape/2, r)
	}
	fmt.Printf("%d categories, %d shapes each, %d held out of training\n", len(categories), len(shapes[0]), heldOut)
	fmt.Printf("training %d lines, test on seen wordings %d lines, on held-out wordings %d lines, foreign %d lines\n\n",
		len(train.lines), len(seen.lines), len(unseen.lines), len(other.lines))
	for c := range categories {
		fmt.Printf("  %-9s seen      %s\n", categories[c], render(shapes[c][0], r))
		fmt.Printf("  %-9s held out  %s\n", "", render(shapes[c][len(shapes[c])-1], r))
	}

	// Bag of words first: the vocabulary comes from the training lines
	// and nothing else, so a word that only occurs in a held-out shape
	// has no column.
	vocab := vocabulary(train.lines)
	fmt.Printf("\nbag of words: %d distinct words in the training lines\n", len(vocab))
	start := time.Now()
	bow := trainHead(bag(train.lines, vocab), train.labels, *epochs, r)
	fmt.Printf("  head trained in %.1fs\n", time.Since(start).Seconds())
	bowSeen, _ := predict(bow, bag(seen.lines, vocab))
	bowUnseen, _ := predict(bow, bag(unseen.lines, vocab))
	fmt.Printf("  accuracy on seen wordings %.1f%%, on held-out wordings %.1f%%\n", accuracy(bowSeen, seen.labels), accuracy(bowUnseen, unseen.labels))

	// Then the embeddings. Every line goes through the encoder once;
	// this is the expensive step and the only one.
	start = time.Now()
	m, err := gemma.Load(dir)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	defer m.Close()
	fmt.Printf("\nEmbeddingGemma loaded in %v\n", time.Since(start).Round(time.Millisecond))
	embed := func(lines []string) [][]float32 {
		v, err := m.Embed(lines, gemma.Prompt("document"))
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		return v
	}
	start = time.Now()
	trainV := embed(train.lines)
	took := time.Since(start)
	fmt.Printf("  %d lines embedded in %.1fs: %.0f lines/s, %d numbers each\n", len(train.lines), took.Seconds(), float64(len(train.lines))/took.Seconds(), len(trainV[0]))
	seenV, unseenV, otherV := embed(seen.lines), embed(unseen.lines), embed(other.lines)

	start = time.Now()
	head := trainHead(matrix(trainV), train.labels, *epochs, r)
	fmt.Printf("  head of %d numbers trained in %.1fs\n", len(trainV[0])*len(categories)+len(categories), time.Since(start).Seconds())
	headSeen, confSeen := predict(head, matrix(seenV))
	headUnseen, confUnseen := predict(head, matrix(unseenV))
	fmt.Printf("  accuracy on seen wordings %.1f%%, on held-out wordings %.1f%%\n", accuracy(headSeen, seen.labels), accuracy(headUnseen, unseen.labels))

	cs := centroids(trainV, train.labels)
	fmt.Printf("\nnearest centroid, no training at all: seen %.1f%%, held out %.1f%%\n",
		accuracy(nearest(cs, seenV), seen.labels), accuracy(nearest(cs, unseenV), unseen.labels))

	fmt.Println("\nheld-out wordings, bag of words:")
	cm := metrics.Confusion(bowUnseen, unseen.labels, len(categories))
	cm.Labels = categories
	fmt.Print(cm)
	fmt.Println("\nheld-out wordings, embedding head:")
	cm = metrics.Confusion(headUnseen, unseen.labels, len(categories))
	cm.Labels = categories
	fmt.Print(cm)
	fmt.Println("  wrong, one line per kind of mistake:")
	shown := map[[2]int]bool{}
	for i, p := range headUnseen {
		if k := [2]int{unseen.labels[i], p}; p != unseen.labels[i] && !shown[k] {
			shown[k] = true
			fmt.Printf("    %-9s for %-9s %.2f  %s\n", categories[p], categories[unseen.labels[i]], confUnseen[i], unseen.lines[i])
		}
	}

	// Lines from outside the five categories. The head has to answer
	// with one of the five; the only thing it can signal is how sure it
	// is, and a threshold on that is the chapter-12 question again.
	otherPred, confOther := predict(head, matrix(otherV))
	fmt.Println("\nforeign lines (application errors), embedding head:")
	for i := 0; i < len(other.lines); i += *perShape / 2 {
		fmt.Printf("  %-9s %.2f  %s\n", categories[otherPred[i]], confOther[i], other.lines[i])
	}
	fmt.Printf("\nconfidence               median    10th pct   worst\n")
	fmt.Printf("  seen wordings          %.3f     %.3f     %.3f\n", quantile(confSeen, 0.5), quantile(confSeen, 0.1), quantile(confSeen, 0))
	fmt.Printf("  held-out wordings      %.3f     %.3f     %.3f\n", quantile(confUnseen, 0.5), quantile(confUnseen, 0.1), quantile(confUnseen, 0))
	fmt.Printf("  foreign lines          %.3f     %.3f     %.3f\n", quantile(confOther, 0.5), quantile(confOther, 0.1), quantile(confOther, 0))
	fmt.Println("\nthreshold   sent to a human: seen   held out    foreign lines caught")
	for _, t := range []float32{0.9, 0.95, 0.98, 0.99} {
		below := func(c []float32) float64 {
			n := 0
			for _, v := range c {
				if v < t {
					n++
				}
			}
			return 100 * float64(n) / float64(len(c))
		}
		fmt.Printf("  %.2f                       %5.1f%%   %5.1f%%    %5.1f%%\n", t, below(confSeen), below(confUnseen), below(confOther))
	}
}
