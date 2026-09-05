// Package logtemplate reduces a stream of log lines to templates: the
// variable parts (addresses, numbers, identifiers) are masked and lines
// that differ only in a few tokens are merged, so that fifty million
// syslog lines a day become a few thousand templates with counts. Only
// the templates need to be embedded or looked at by a person.
package logtemplate

import "strings"

// Placeholders written by Mask.
const (
	IP  = "<IP>"
	MAC = "<MAC>"
	Hex = "<HEX>"
	Num = "<NUM>"
	Str = "<STR>"
)

// Mask replaces the variable parts of a line by placeholders: IPv4 and
// IPv6 addresses (with an optional port or prefix length), MAC
// addresses, single-token quoted strings, hexadecimal identifiers (four
// or more hex characters mixing digits and letters, or a 0x prefix) and
// numbers, including digit runs inside words such as eth0 or ge-0/0/3.
// Words made only of hex letters (deadbeef, cafe) are left alone; they
// cannot be told from English. Runs of whitespace collapse to one space.
//
// It is a single pass over the bytes with no regular expressions: about
// a microsecond per line.
func Mask(line string) string {
	var b strings.Builder
	b.Grow(len(line))
	i, n := 0, len(line)
	space := false
	for i < n {
		c := line[i]
		if c == ' ' || c == '\t' || c == '\n' || c == '\r' {
			space = true
			i++
			continue
		}
		if space && b.Len() > 0 {
			b.WriteByte(' ')
		}
		space = false
		if c == '"' || c == '\'' { // quoted string within one token
			if j := strings.IndexByte(line[i+1:], c); j >= 0 && !strings.ContainsAny(line[i+1:i+1+j], " \t") {
				b.WriteString(Str)
				i += j + 2
				continue
			}
		}
		if !isWordByte(c) { // punctuation passes through
			b.WriteByte(c)
			i++
			continue
		}
		j := i
		for j < n && isWordByte(line[j]) {
			j++
		}
		maskRun(&b, line[i:j])
		i = j
	}
	return b.String()
}

// isWordByte: the characters that can be part of an address, number or
// identifier run.
func isWordByte(c byte) bool {
	return c >= '0' && c <= '9' || c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c == '.' || c == ':' || c == '/' || c == '-' || c == '_' || c == '%'
}

func isDigit(c byte) bool { return c >= '0' && c <= '9' }
func isHex(c byte) bool {
	return isDigit(c) || c >= 'a' && c <= 'f' || c >= 'A' && c <= 'F'
}

// maskRun classifies one run of word bytes and writes its masked form.
func maskRun(b *strings.Builder, s string) {
	// trailing '.' ',' ':' belong to the sentence, not the token
	trail := ""
	for len(s) > 0 && (s[len(s)-1] == '.' || s[len(s)-1] == ':' || s[len(s)-1] == '-') {
		trail = s[len(s)-1:] + trail
		s = s[:len(s)-1]
	}
	switch {
	case s == "":
	case isIPv4(s):
		b.WriteString(IP)
	case isMAC(s):
		b.WriteString(MAC)
	case isIPv6(s):
		b.WriteString(IP)
	case isNumber(s):
		b.WriteString(Num)
	case isHexID(s):
		b.WriteString(Hex)
	default:
		// digit runs (with inner dots) inside a word: eth0, ge-0/0/3, 42.5C
		i := 0
		for i < len(s) {
			if isDigit(s[i]) {
				j := i
				for j < len(s) && (isDigit(s[j]) || s[j] == '.' && j+1 < len(s) && isDigit(s[j+1])) {
					j++
				}
				b.WriteString(Num)
				i = j
				continue
			}
			b.WriteByte(s[i])
			i++
		}
	}
	b.WriteString(trail)
}

func isNumber(s string) bool {
	i := 0
	if i < len(s) && s[i] == '-' {
		i++
	}
	digits, dots := 0, 0
	for ; i < len(s); i++ {
		switch {
		case isDigit(s[i]):
			digits++
		case s[i] == '.':
			dots++
		default:
			return false
		}
	}
	return digits > 0 && dots <= 1
}

// isIPv4: d.d.d.d with 1–3 digits per part, optional :port or /prefix.
func isIPv4(s string) bool {
	if k := strings.IndexAny(s, ":/"); k >= 0 {
		port := s[k+1:]
		if port == "" || !allDigits(port) {
			return false
		}
		s = s[:k]
	}
	parts := 0
	i := 0
	for i < len(s) {
		j := i
		for j < len(s) && isDigit(s[j]) {
			j++
		}
		if j == i || j-i > 3 {
			return false
		}
		parts++
		if j == len(s) {
			break
		}
		if s[j] != '.' {
			return false
		}
		i = j + 1
		if i == len(s) {
			return false
		}
	}
	return parts == 4
}

func allDigits(s string) bool {
	for i := 0; i < len(s); i++ {
		if !isDigit(s[i]) {
			return false
		}
	}
	return len(s) > 0
}

// isMAC: six hex pairs separated by ':' or '-', or Cisco's three groups
// of four separated by '.'.
func isMAC(s string) bool {
	if len(s) == 17 {
		sep := s[2]
		if sep != ':' && sep != '-' {
			return false
		}
		for i := 0; i < 17; i++ {
			if i%3 == 2 {
				if s[i] != sep {
					return false
				}
			} else if !isHex(s[i]) {
				return false
			}
		}
		return true
	}
	if len(s) == 14 && s[4] == '.' && s[9] == '.' {
		for i := 0; i < 14; i++ {
			if i == 4 || i == 9 {
				continue
			}
			if !isHex(s[i]) {
				return false
			}
		}
		return true
	}
	return false
}

// isIPv6: at least two colons, only hex digits and colons (an embedded
// IPv4 tail or a /prefix is allowed).
func isIPv6(s string) bool {
	if k := strings.IndexByte(s, '/'); k >= 0 {
		if !allDigits(s[k+1:]) {
			return false
		}
		s = s[:k]
	}
	colons, hexes := 0, 0
	for i := 0; i < len(s); i++ {
		switch {
		case s[i] == ':':
			colons++
		case isHex(s[i]):
			hexes++
		case s[i] == '.':
		default:
			return false
		}
	}
	return colons >= 2 && hexes > 0
}

// isHexID: 0x-prefixed hex, or four or more hex characters that mix
// digits and letters.
func isHexID(s string) bool {
	if len(s) > 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X') {
		s = s[2:]
		if s == "" {
			return false
		}
		for i := 0; i < len(s); i++ {
			if !isHex(s[i]) {
				return false
			}
		}
		return true
	}
	if len(s) < 4 {
		return false
	}
	digits, letters := 0, 0
	for i := 0; i < len(s); i++ {
		switch {
		case isDigit(s[i]):
			digits++
		case isHex(s[i]):
			letters++
		default:
			return false
		}
	}
	return digits > 0 && letters > 0
}
