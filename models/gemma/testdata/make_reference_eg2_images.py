"""fp32 reference embeddings for EmbeddingGemma 2 image input.

Writes to argv[1]:
  eg2_images.json      image names, the mixed input's text, prompt
  eg2_images.bin       int32 n, int32 dim, n*dim float32 (one per image, in order)
  eg2_mixed.bin        the mixed text+image input, and the prompted image
and, for development only, to argv[2] the preprocessed patches per image
(pixel_values rounded back to 8 bits, int32 n, int32 patch_len, uint8 data;
then int32 positions n*2) so the Go resize can be compared pixel by pixel.
"""
import json, os, struct, sys
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

out, dev = sys.argv[1], sys.argv[2]
d = "/Users/sven/models/embeddinggemma-2"
images_dir = "/Users/sven/ai/models/gemma/testdata/images"
names = ["red_circle.png", "blue_square.png", "green_triangle.png", "gray_gradient.png",
         "tiny_noise.png", "wide_stripes.png", "alpha_ramp.png", "earth.png"]
model = SentenceTransformer(d, device="cpu").float()
imgs = [Image.open(os.path.join(images_dir, n)) for n in names]

emb = np.asarray(model.encode(imgs, convert_to_numpy=True, normalize_embeddings=True, batch_size=1), dtype=np.float32)
with open(os.path.join(out, "eg2_images.bin"), "wb") as f:
    f.write(struct.pack("<ii", *emb.shape))
    f.write(emb.tobytes())

mixed_text = "Product photo: <|image|> A round red warning sign on a white background."
prompt = "task: search result | query: "
mixed = np.asarray(model.encode([{"text": mixed_text, "image": imgs[0]}], convert_to_numpy=True, normalize_embeddings=True), dtype=np.float32)
prompted = np.asarray(model.encode([imgs[1]], prompt=prompt, convert_to_numpy=True, normalize_embeddings=True), dtype=np.float32)
both = np.concatenate([mixed, prompted])
with open(os.path.join(out, "eg2_mixed.bin"), "wb") as f:
    f.write(struct.pack("<ii", *both.shape))
    f.write(both.tobytes())
json.dump({"images": names, "mixed_text": mixed_text, "prompt": prompt},
          open(os.path.join(out, "eg2_images.json"), "w"))

# What the processor sees, for the pixel comparison.
for name, img in zip(names, imgs):
    feats = model.preprocess([img])
    pv = feats["pixel_values"][0].numpy()
    pos = feats["image_position_ids"][0].numpy()
    valid = (pos >= 0).all(axis=1)
    pv, pos = pv[valid], pos[valid]
    u8 = np.rint(pv * 255).astype(np.uint8)
    with open(os.path.join(dev, "pixels_" + name + ".bin"), "wb") as f:
        f.write(struct.pack("<ii", *u8.shape))
        f.write(u8.tobytes())
        f.write(pos.astype(np.int32).tobytes())
    ids = feats["input_ids"][0].tolist()
    print(f"{name:20s} {img.size[0]}x{img.size[1]} {img.mode:5s} -> {u8.shape[0]} patches, {len(ids)} tokens")
mf = model.preprocess([{"text": mixed_text, "image": imgs[0]}])
print("mixed ids head", mf["input_ids"][0][:12].tolist(), "len", mf["input_ids"].shape[1])
pf = model.preprocess([imgs[1]], prompt=prompt)
print("prompted ids head", pf["input_ids"][0][:14].tolist(), "len", pf["input_ids"].shape[1])
