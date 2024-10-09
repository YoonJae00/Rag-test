import requests
from PIL import Image
from transformers import ViTFeatureExtractor, ViTModel

feature_extractor = ViTFeatureExtractor.from_pretrained('facebook/dino-vits16')
model = ViTModel.from_pretrained('facebook/dino-vits16')

test_img = Image.open(requests.get("https://i.ibb.co/7Ksr5mw/yNp6qTS.png", stream=True).raw).convert("RGB")



test_img_tensor = feature_extractor(images=test_img, return_tensors="pt")
test_outputs = model(**test_img_tensor)

test_embedding = test_outputs.pooler_output.detach().cpu().numpy().squeeze().tolist()

len(test_embedding)

import chromadb

# ChromaDB 클라이언트 초기화
client = chromadb.PersistentClient(path="./chroma_db")

# 기존 'food' 컬렉션 불러오기
collection = client.get_collection("foods")

query_result = collection.query(
    query_embeddings=[test_embedding],
    n_results=3,
)

print(query_result)

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(16, 10))

for i, metadata in enumerate(query_result["metadatas"][0]):
    distance = query_result["distances"][0][i]

    axes[i].imshow(Image.open(metadata["uri"]))
    axes[i].set_title(f"{metadata['name']}: {distance:.2f}")
    axes[i].axis("off")

plt.show()