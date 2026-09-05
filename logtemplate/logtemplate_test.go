package logtemplate

import (
	"fmt"
	"math/rand/v2"
	"testing"
)

func TestMask(t *testing.T) {
	cases := map[string]string{
		"sshd[41233]: Accepted publickey for sven from 10.20.30.40 port 51234 ssh2":         "sshd[<NUM>]: Accepted publickey for sven from <IP> port <NUM> ssh<NUM>",
		"link down on ge-0/0/3 after 41233 s":                                               "link down on ge-<NUM>/<NUM>/<NUM> after <NUM> s",
		"DHCPACK on 192.168.1.77 to 3c:22:fb:9a:01:44 via eth0":                             "DHCPACK on <IP> to <MAC> via eth<NUM>",
		"session 0x7f3a9c2e opened for user 'root' from fe80::1c2d:3e4f:5a6b:7c8d":          "session <HEX> opened for user <STR> from <IP>",
		`kernel: [ 1234.567890] usb 1-1: new high-speed USB device number 7 using xhci_hcd`: "kernel: [ <NUM>] usb <NUM>-<NUM>: new high-speed USB device number <NUM> using xhci_hcd",
		"temperature 42.5C exceeds threshold -3.0 on sensor deadbeef":                       "temperature <NUM>C exceeds threshold <NUM> on sensor deadbeef",
	}
	for in, want := range cases {
		if got := Mask(in); got != want {
			t.Errorf("Mask(%q)\n got %q\nwant %q", in, got, want)
		}
	}
}

func TestMinerMergesAndSeparates(t *testing.T) {
	m := NewMiner()
	lines := []string{
		"interface ge-0/0/1 changed state to down",
		"interface ge-0/0/7 changed state to down",
		"interface xe-1/2/0 changed state to up",
		"Accepted publickey for sven from 10.1.1.1 port 22 ssh2",
		"Accepted publickey for anna from 10.1.1.9 port 22 ssh2",
		"Failed password for root from 203.0.113.7 port 4411 ssh2",
		"interface ge-0/0/1 changed state to down",
	}
	ids := make([]int, len(lines))
	for i, l := range lines {
		ids[i] = m.Add(l)
	}
	if ids[0] != ids[1] || ids[0] != ids[2] || ids[0] != ids[6] {
		t.Fatalf("interface lines not merged: %v", ids)
	}
	if ids[3] != ids[4] {
		t.Fatalf("login lines not merged: %v", ids)
	}
	if ids[3] == ids[5] {
		t.Fatalf("accepted and failed logins merged: %v", ids)
	}
	if ids[0] == ids[3] {
		t.Fatalf("interface and login lines merged: %v", ids)
	}
	tm := m.Template(ids[0])
	if tm.Count != 4 || tm.Tokens[1] != Wildcard || tm.Tokens[5] != Wildcard {
		t.Fatalf("interface template %v count %d", tm, tm.Count)
	}
	total := 0
	for _, tp := range m.Templates() {
		total += tp.Count
	}
	if total != len(lines) {
		t.Fatalf("counts sum to %d, want %d", total, len(lines))
	}
	if m.Templates()[0].ID != ids[0] {
		t.Fatal("Templates not sorted by count")
	}
}

// genLines fakes a day of syslog: a few hundred message shapes with
// varying addresses, numbers and names.
func genLines(r *rand.Rand, n int) []string {
	shapes := []string{
		"sshd[%d]: Accepted publickey for %s from %s port %d ssh2",
		"sshd[%d]: Failed password for invalid user %s from %s port %d ssh2",
		"kernel: eth%d: link is up, 1000 Mbps full duplex",
		"dhcpd: DHCPACK on %s to %s via eth%d",
		"named[%d]: client %s#%d: query: %s.example.com IN A + (%s)",
		"systemd[1]: Started session %d of user %s.",
		"CRON[%d]: (%s) CMD (/usr/bin/backup --job %d)",
		"nginx: %s - - GET /api/v%d/items/%d HTTP/1.1 200 %d",
		"firewall: DROP IN=eth%d OUT= SRC=%s DST=%s PROTO=TCP SPT=%d DPT=%d",
		"switch-%d: %%LINK-3-UPDOWN: Interface GigabitEthernet%d/0/%d, changed state to %s",
	}
	users := []string{"sven", "anna", "root", "backup", "www-data"}
	out := make([]string, n)
	for i := range out {
		ip := func() string { return fmt.Sprintf("10.%d.%d.%d", r.IntN(255), r.IntN(255), r.IntN(255)) }
		mac := func() string { return fmt.Sprintf("3c:22:fb:%02x:%02x:%02x", r.IntN(256), r.IntN(256), r.IntN(256)) }
		state := []string{"up", "down"}[r.IntN(2)]
		switch s := shapes[r.IntN(len(shapes))]; s {
		case shapes[0], shapes[1]:
			out[i] = fmt.Sprintf(s, r.IntN(60000), users[r.IntN(len(users))], ip(), r.IntN(65535))
		case shapes[2]:
			out[i] = fmt.Sprintf(s, r.IntN(4))
		case shapes[3]:
			out[i] = fmt.Sprintf(s, ip(), mac(), r.IntN(4))
		case shapes[4]:
			out[i] = fmt.Sprintf(s, r.IntN(60000), ip(), r.IntN(65535), users[r.IntN(len(users))], ip())
		case shapes[5]:
			out[i] = fmt.Sprintf(s, r.IntN(9000), users[r.IntN(len(users))])
		case shapes[6]:
			out[i] = fmt.Sprintf(s, r.IntN(60000), users[r.IntN(len(users))], r.IntN(40))
		case shapes[7]:
			out[i] = fmt.Sprintf(s, ip(), r.IntN(3)+1, r.IntN(100000), r.IntN(20000))
		case shapes[8]:
			out[i] = fmt.Sprintf(s, r.IntN(4), ip(), ip(), r.IntN(65535), r.IntN(65535))
		default:
			out[i] = fmt.Sprintf(s, r.IntN(20), r.IntN(3)+1, r.IntN(48), state)
		}
	}
	return out
}

func TestMinerOnGeneratedDay(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2))
	lines := genLines(r, 20000)
	m := NewMiner()
	for _, l := range lines {
		m.Add(l)
	}
	if n := m.Len(); n < 10 || n > 40 {
		t.Fatalf("expected about 10-20 templates for 10 shapes, got %d:\n%v", n, m.Templates())
	}
}

func BenchmarkMinerAdd(b *testing.B) {
	r := rand.New(rand.NewPCG(3, 4))
	lines := genLines(r, 100000)
	m := NewMiner()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.Add(lines[i%len(lines)])
	}
}

func BenchmarkMask(b *testing.B) {
	r := rand.New(rand.NewPCG(3, 4))
	lines := genLines(r, 10000)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Mask(lines[i%len(lines)])
	}
}
