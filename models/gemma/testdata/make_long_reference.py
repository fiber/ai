"""Reference embeddings for inputs longer than the 512-token sliding window,
where Gemma's sliding and full attention layers differ. Writes
models/gemma/testdata/reference_long.bin: int32 n, int32 dim, n*dim float32,
and long_sentences.json with the texts."""
import json
import struct
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("$FIBERAI_MODELS/embeddinggemma-300m", device="cpu").float()

base = ("Sep  8 09:15:42 fw01 %ASA-6-302013: Built outbound TCP connection 12345 "
        "for outside:203.0.113.7/443 to inside:10.0.0.5/52918. ")
prose = ("The network operations team reviewed the overnight incident in which the "
         "core router lost its BGP session to the upstream provider for eleven minutes; "
         "traffic failed over to the secondary uplink as designed, but the monitoring "
         "alert arrived four minutes late because the syslog collector was under load. ")
texts = [
    base * 12,      # about 700 tokens: crosses the 512 window once
    prose * 22,     # about 1300 tokens
    (base + prose) * 14,  # about 2000 tokens, near the 2048 cap
]
enc = model.tokenize(texts)
lengths = [int((row != 0).sum()) for row in enc["input_ids"]]
print("token lengths", lengths)

emb = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=4)
emb = np.asarray(emb, dtype=np.float32)
n, dim = emb.shape
with open("models/gemma/testdata/reference_long.bin", "wb") as f:
    f.write(struct.pack("<ii", n, dim))
    f.write(emb.tobytes())
json.dump({"texts": texts, "tokens": lengths},
          open("models/gemma/testdata/long_sentences.json", "w"), ensure_ascii=False)
print("wrote", n, "long references")
