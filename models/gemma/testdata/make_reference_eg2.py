"""fp32 reference embeddings for google/embeddinggemma-2 (text only), in the
format of make_reference.py and make_long_reference.py: int32 n, int32 dim,
n*dim float32. Reuses the 300m test inputs so the two models are checked on
the same text. Output goes to the directory given as argv[1]."""
import json, os, struct, sys
import numpy as np
from sentence_transformers import SentenceTransformer

out = sys.argv[1]
testdata = "/Users/sven/ai/models/gemma/testdata"
model = SentenceTransformer("/Users/sven/models/embeddinggemma-2", device="cpu").float()

sc = json.load(open(os.path.join(testdata, "sentences.json")))
emb = np.asarray(model.encode(sc["sentences"], convert_to_numpy=True, normalize_embeddings=True, batch_size=16), dtype=np.float32)
qd = np.asarray(model.encode([sc["query_prompted"], sc["document_prompted"]], convert_to_numpy=True, normalize_embeddings=True), dtype=np.float32)
n, dim = emb.shape
with open(os.path.join(out, "eg2_reference.bin"), "wb") as f:
    f.write(struct.pack("<ii", n, dim)); f.write(emb.tobytes())
    f.write(struct.pack("<ii", 2, dim)); f.write(qd.tobytes())

long = json.load(open(os.path.join(testdata, "long_sentences.json")))
enc = model.tokenize(long["texts"])
print("long token lengths", [int(m.sum()) for m in enc["attention_mask"]])
le = np.asarray(model.encode(long["texts"], convert_to_numpy=True, normalize_embeddings=True, batch_size=4), dtype=np.float32)
with open(os.path.join(out, "eg2_reference_long.bin"), "wb") as f:
    f.write(struct.pack("<ii", *le.shape)); f.write(le.tobytes())
print("wrote", n, "short and", le.shape[0], "long references, dim", dim)
print("first sentence tokens:", model.tokenize([sc["sentences"][0]])["input_ids"][0].tolist())
