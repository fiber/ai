import json
from tokenizers import Tokenizer

tok = Tokenizer.from_file("$FIBERAI_MODELS/embeddinggemma-300m/tokenizer.json")

strings = [
    "", " ", "  ", "\t", "\n", "hello", "Hello, world!", " leading space",
    "trailing space ", "multiple   spaces", "a\tb\tc", "line1\nline2",
    "The quick brown fox jumps over the lazy dog.",
    "Der schnelle braune Fuchs springt über den faulen Hund.",
    "Größe: 42µm, ±0.5°C, café, naïve, Ölmenge",
    "192.168.1.1", "2001:db8::1", "00:1a:2b:3c:4d:5e",
    "2026-09-08T09:15:42.123Z", "Sep  8 09:15:42",
    "%BGP-5-ADJCHANGE: neighbor 10.0.0.1 Up",
    "kernel: [12345.678901] eth0: link down",
    '{"level":"error","msg":"connection refused","code":500}',
    "https://example.com/path?q=1&r=2#frag",
    "GET /api/v1/users HTTP/1.1", "SELECT * FROM logs WHERE id=42;",
    "0xDEADBEEF 0b1010 1_000_000", "emoji: \U0001f680\U0001f525\U0001f600 mixed 文字 テスト",
    "中文测试字符串", "日本語のテスト", "한국어 테스트", "العربية اختبار",
    "Здравствуй мир", "tab\tand\nnewline",
    "CamelCaseIdentifier snake_case_name kebab-case-name",
    "a" * 100, "▁already has underscore", "<bos>literal", "<eos>", "<pad>",
    "<start_of_turn>user", "<unused42>", "price is $19.99 or €15,50",
    "café", "é", "ﬁle (ligature)", "ＦＵＬＬＷＩＤＴＨ",
    "SFTP transfer 4096 bytes from 172.16.0.5",
    "Authentication failure for user admin from 203.0.113.7 port 22",
    "CPU load 0.95 mem 87% disk /var 92%",
    "task: search result | query: what is the mtu",
    "title: none | text: interface GigabitEthernet0/1 is down",
]
for w in "the of and to in is it you that he was for on are with as I his they be at one have this from or had by word but not what all were we when your can said there use an each which she do how their if".split():
    strings.append(w)

out = []
for s in strings:
    ids = tok.encode(s).ids
    ids_raw = tok.encode(s, add_special_tokens=False).ids
    out.append({"s": s, "ids": ids, "raw": ids_raw})

json.dump(out, open("tokenizer/testdata/tokens.json", "w"), ensure_ascii=False)
print("wrote", len(out), "cases")
