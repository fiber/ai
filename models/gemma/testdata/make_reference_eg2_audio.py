"""fp32 reference embeddings for EmbeddingGemma 2 audio input.

The clips are built from the WAVs in models/gemma/testdata/audio exactly as
the Go test builds them. Writes to argv[1]:
  eg2_audio.json   clip names, the mixed input's text
  eg2_audio.bin    int32 n, int32 dim, n*dim float32, one per clip, then the mixed input
and to argv[2], for development only, the log-mel features and frame mask
of two clips (int32 frames, int32 bins, float32 data, then uint8 mask).
"""
import json, os, struct, sys, wave
import numpy as np
from sentence_transformers import SentenceTransformer

out, dev = sys.argv[1], sys.argv[2]
adir = "/Users/sven/ai/models/gemma/testdata/audio"


def read(name):
    w = wave.open(os.path.join(adir, name))
    return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0


speech = read("librivox_20s.wav")
clips = {
    "speech_3s": speech[: 3 * 16000],
    "speech_20s": speech,
    "speech_40s_truncated": np.concatenate([speech, speech]),
    "tone_440hz": read("tone_440hz.wav"),
    "noise_2s": read("noise_2s.wav"),
    "quiet_1s": read("quiet_1s.wav"),
    "speech_50ms": speech[:800],
}
names = list(clips)
m = SentenceTransformer("/Users/sven/models/embeddinggemma-2", device="cpu").float()
inputs = [{"array": clips[n], "sampling_rate": 16000} for n in names]
emb = np.asarray(m.encode(inputs, batch_size=1, convert_to_numpy=True, normalize_embeddings=True), dtype=np.float32)
mixed_text = "Recording of the opening: <|audio|> It introduces the book."
mixed = np.asarray(m.encode([{"text": mixed_text, "audio": {"array": clips["speech_3s"], "sampling_rate": 16000}}],
                            convert_to_numpy=True, normalize_embeddings=True), dtype=np.float32)
allv = np.concatenate([emb, mixed])
with open(os.path.join(out, "eg2_audio.bin"), "wb") as f:
    f.write(struct.pack("<ii", *allv.shape))
    f.write(allv.tobytes())
json.dump({"clips": names, "mixed_text": mixed_text}, open(os.path.join(out, "eg2_audio.json"), "w"))

for n in ("speech_3s", "tone_440hz", "speech_50ms"):
    f = m.preprocess([{"array": clips[n], "sampling_rate": 16000}])
    feats = f["input_features"][0].numpy().astype(np.float32)
    mask = f["input_features_mask"][0].numpy().astype(np.uint8)
    with open(os.path.join(dev, "mel_" + n + ".bin"), "wb") as fh:
        fh.write(struct.pack("<ii", *feats.shape))
        fh.write(feats.tobytes())
        fh.write(mask.tobytes())
for n, inp in zip(names, inputs):
    ids = m.preprocess([inp])["input_ids"][0].tolist()
    print(f"{n:22s} {len(clips[n]) / 16000:6.2f} s -> {ids.count(258881)} audio tokens")
