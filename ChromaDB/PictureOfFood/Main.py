from PIL import Image

# img = Image.open("test/Bread/0.jpg")

# Image Vectorizer 모델 로드
# https://huggingface.co/facebook/dino-vits16
from transformers import ViTFeatureExtractor, ViTModel

feature_extractor = ViTFeatureExtractor.from_pretrained('facebook/dino-vits16')
model = ViTModel.from_pretrained('facebook/dino-vits16')

print("Models loaded!")

# 임베딩
# img_tensor = feature_extractor(images=img, return_tensors="pt")
# outputs = model(**img_tensor)

# embedding = outputs.pooler_output.detach().cpu().numpy().squeeze()

# print(embedding)

# Chroma DB 시작
import chromadb
from chromadb.config import Settings

 # 영구 저장소 설정

# 구버전 코드임 사용 X
# client = chromadb.Client(Settings(
#        chroma_db_impl="duckdb+parquet",
#        persist_directory="./chroma_db"  # 저장할 디렉토리 지정
#    ))
# document 생성
# collection = client.create_collection("foods")

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection("foods")


# 모든 이미지 벡터화
from glob import glob
# glob 라이브러리는 파일 시스템에서 특정 패턴과 일치하는 파일명을 찾을 때 사용하는 Python 내장 모듈입니다.
# 주로 다음과 같은 상황에서 사용됩니다
img_list = sorted(glob("test/*/*.jpg"))

len(img_list)

# 110 개 파일들 백터화
from tqdm import tqdm

embeddings = []
metadatas = []
ids = []

for i, img_path in enumerate(tqdm(img_list)):
    img = Image.open(img_path)
    cls = img_path.split("/")[1]

    img_tensor = feature_extractor(images=img, return_tensors="pt")
    outputs = model(**img_tensor)

    embedding = outputs.pooler_output.detach().cpu().numpy().squeeze().tolist()

    embeddings.append(embedding)

    metadatas.append({
        "uri": img_path,
        "name": cls
    })

    ids.append(str(i))

print("Done!")

# 임베딩을 데이터베이스에 저장
# 즉 food 라는 컬렉션에 저장
collection.add(
    embeddings=embeddings,
    metadatas=metadatas,
    ids=ids, # str 로 넣어야함
)