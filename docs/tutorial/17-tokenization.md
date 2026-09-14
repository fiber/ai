# 17. Tokens: what a model actually reads

Chapter 15 fed its model characters and gave a paragraph of reasons.
Chapter 10 loaded a model that reads *pieces* — `▁question`, `ing`,
`▁the` — and never said where the pieces come from. Every real language
model has a layer in front of it that is not a neural network at all: the
tokenizer, which decides what one symbol is. This chapter builds one,
trains chapter 15's model on its output, and measures what changed.

```
go run ./examples/tutorial/17-tokenization
```

About eight minutes: two training runs of four minutes each, the same
model twice. Everything else takes a second.

## Three views of one sentence

```
"To be, or not to be, that is the question:"
   42 characters  [32 53 1 40 43 6 1 53 56 1 52 53]
   10 words       ["To" "be," "or" "not"]
   13 Gemma pieces ["To" "▁be" "," "▁or" "▁not" "▁to" "▁be" "," "▁that" "▁is" "▁the" "▁question" ":"]  (vocabulary 262144)
```

Characters are the smallest choice: 65 symbols in this corpus, every
one of them seen tens of thousands of times, and a model that has to
spend its first layers learning that `t`, `h`, `e` is a thing. Words are
the largest: no spelling to learn, but `be,` and `be` are different
symbols, the vocabulary of a real corpus runs to hundreds of thousands,
and the first word not in it — a name, a typo, a hostname — has no
symbol at all.

Gemma's line is the compromise every current model makes. `▁be` is one
piece, the `▁` standing for the space in front of it, and so is
`▁question`; a word the vocabulary does not hold breaks into pieces it
does, down to single bytes if it has to. The Gemma line only appears
when `FIBERAI_MODELS` points at the model from chapter 10; the rest of
the chapter does not need it.

## Byte-pair encoding, in a loop

The algorithm that produces such a vocabulary is short enough to be in
the example rather than in the library. Start with the bytes. Count
every adjacent pair of symbols in the corpus. Merge the most frequent
pair into a new symbol. Repeat.

```go
for done := 1; done <= merges; done++ {
    counts := pairCounts(words)          // (a, b) -> occurrences
    best := mostFrequent(counts)
    id := len(b.pieces)
    b.pieces = append(b.pieces, b.pieces[best[0]]+b.pieces[best[1]])
    b.rank[best], b.merged[best] = done, id
    for wi := range words {
        words[wi].syms = replacePair(words[wi].syms, best, id)
    }
}
```

That is *byte-pair encoding*, BPE, and the merges in the order they were
made are the whole tokenizer: to encode new text, apply them in the
same order. One detail matters, and every production tokenizer has it.
Before counting, the text is cut into chunks — a run of letters with
the space in front of it, or a single other byte — and merges never
cross a chunk. Without that rule the most frequent pairs are things
like `e,` and `, ` and the vocabulary fills with fragments that mean
nothing; with it, a piece is at most one word and its leading space.

```
learning 1024 merges on 1115394 characters
  merges   vocabulary    corpus tokens   chars/token   last piece
       0           65         1115394          1.00   ""
       1           66         1091557          1.02   " t"
       2           67         1073354          1.04   "he"
       3           68         1059813          1.05   " a"
       4           69         1047083          1.07   "ou"
       8           73         1002858          1.11   " w"
      16           81          936079          1.19   " f"
      64          129          747778          1.49   "ith"
     256          321          578006          1.93   "ate"
     512          577          503345          2.22   " VINCENTIO"
    1024         1089          438337          2.54   "pose"
  learned in 1.0s
  Gemma's 262144 pieces, trained on other text: 321210 tokens, 3.47 chars/token
```

The first merge is ` t`, the second `he`; ` the` is the twelfth, made
from those two, and ` you`, ` and`, ` of` and ` my` all arrive before
the sixtieth. Every merge shortens the corpus by exactly the
number of times its pair occurred, so the token count falls fastest at
the start and the 1024th merge (`pose`, as in *suppose* and *purpose*)
buys almost nothing. A vocabulary of 1089 symbols encodes the play in
438 000 tokens, two and a half characters each.

The Gemma line is the same corpus through a vocabulary 240 times
larger, learned from the web rather than from Shakespeare: 3.47
characters per token. Bigger helps, and it helps less than you would
think — most of the gain is in the first few hundred merges, which is
why a 1024-merge vocabulary is a reasonable thing to train on a
megabyte.

Since the corpus is a few thousand distinct chunks with frequencies,
the loop counts pairs over those rather than over the text, and the
whole thing takes a second. Encoding is the same merges applied per
chunk, lowest rank first, with the result cached by chunk, and decoding
is string concatenation; the example checks that the corpus comes back
byte for byte.

## The same model on two vocabularies

Now the question the chapter exists for. Chapter 15's decoder — four
blocks, width 256, 128 tokens of context, batches of 32 — is trained
twice for 1000 steps from the same seed, once on characters and once on
the 1089 pieces. The held-out tenth of the corpus is cut at a line
boundary so both models are scored on exactly the same text:

```
validation text: 111539 characters = 111539 character tokens = 45996 piece tokens
```

```
training on characters: 1000 steps of 32 x 128 tokens
  step  250  train 1.822  57s
  step 1000  train 1.438  224s

training on 1089 pieces: 1000 steps of 32 x 128 tokens
  step  250  train 3.832  59s
  step 1000  train 2.920  235s

                      ms/step     loss/token    bits/char      chars/s
characters                224          1.634        2.357          400
1089 pieces               235          3.749        2.230          885
```

Read the table carefully, because its second column is a trap. The
per-token loss is what every framework prints and it says the piece
model is *worse*: 3.749 nats per token against 1.634. It is
predicting one of 1089 symbols instead of one of 65, and each of its
guesses is worth two and a half characters. A loss per token cannot be
compared across vocabularies at all.

Bits per character can. Total the loss over the held-out text in nats,
divide by the characters those tokens spell, convert to bits, and the
two models are on one scale: 2.230 bits per character for the piece
model against 2.357 for the character model. The piece model is the
better one, by five percent, and the per-token loss said the opposite
by a factor of two.

The other two columns are the practical side. A step costs about the
same in either vocabulary — the model is the same, only the output layer
grew — but a step of the piece model covers two and a half times more
text, so in the same 1000 steps it read 2.5× more Shakespeare. And when
generating, every forward pass produces a whole piece: 885 characters
per second against 400, from the same forward pass at the same cost.
That multiplier stacks with chapter 16's KV cache, which is per token as
well: 400 characters of context are 400 rows in the cache for the
character model and about 160 for the piece model.

## What it writes

```
sample from the characters model:
or Warwick, muster if thou hast fear the sole.
The hand you belo! my lord spice that the sake
pray their can her sun me, unlainted reath;
I will be again, good will he worship,

sample from the 1089 pieces model:
Second Servingman:
Come, I am a Peter'd balder. What, boot!

BISTRESS Offords! Come boot! my good lord; I am a cup
his sister of Herefords, and alived the business,
To be our princely and lies all with some ill
To make her enemies; and ineest have the Signery
```

Both still invent words, and the difference is in how. The character
model's inventions are letter salad — `unlainted`, `RICHMustiation` a
few lines further down. The piece model's are assembled from pieces
that are themselves real syllables and words, so `Herefords`,
`Signery` and `injurives` look like English a copy-editor missed. It is
also better at the shape of the play — speaker, colon, line — because a
speaker's name is one or two of its symbols rather than a dozen.

What a piece model cannot do cheaply is spell something it never saw as
a unit. The 65 single bytes stay in the vocabulary for that, which is
why Gemma's tokenizer is called *byte-level*: a name from outside the
training text comes out of the byte pieces, one character per token,
at the character model's price.

## What to remember

- A tokenizer is a vocabulary plus an ordered list of merges. Training
  it is a loop; the library's `tokenizer` package loads the ones that
  ship with models.
- Merges stay inside words. The chunking rule is a design decision, not
  an implementation detail.
- Per-token loss is not comparable across vocabularies. Bits per
  character is.
- A larger vocabulary buys shorter sequences, which is faster to train
  on and faster to generate from, with diminishing returns after a few
  hundred merges.

## What to try

`-merges 256` and `-merges 4096`: the table shows the token count, the
comparison shows whether the model kept up. With 4096 merges the
vocabulary holds pieces that occur a handful of times in the corpus and
their rows in the embedding barely train. Then look at the last-piece
column at 512: ` VINCENTIO` is a character's name that appears often
enough to earn a merge — a vocabulary learned on one corpus carries that
corpus with it, which is what the Gemma line above was measuring in the
other direction.

Next: [18. A text classifier on frozen embeddings](18-text-classifier.md).
