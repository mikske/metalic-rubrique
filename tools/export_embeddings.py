import torch, json

src = '/Volumes/Extreme SSD/vkr_rusage/embeddings/fasttext_embeddings.pt'
dst = '/Volumes/Extreme SSD/vkr_rusage/embeddings/fasttext_embeddings.f32'

m = torch.load(src, map_location="cpu").float().contiguous()
m.numpy().tofile(dst)

meta = {"rows": m.shape[0], "cols": m.shape[1], "dtype": "float32"}
json.dump(meta, open(dst + ".json", "w"))
print(meta)