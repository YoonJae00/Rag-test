from dotenv import load_dotenv
load_dotenv()

from langchain_teddynote import logging

# 프로젝트 이름을 입력합니다.
logging.langsmith("gen-image")

# DALL-E API 래퍼 가져오기
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from IPython.display import Image

# DALL-E API 래퍼 초기화
# model: 사용할 DALL-E 모델 버전
# size: 생성할 이미지 크기
# quality: 이미지 품질
# n: 생성할 이미지 수
dalle = DallEAPIWrapper(model="dall-e-3", size="1024x1024", quality="standard", n=1)

# 질문
query = "스마트폰을 바라보는 사람들을 풍자한 neo-classicism painting"

# 이미지 생성 및 URL 받기
# chain.invoke()를 사용하여 이미지 설명을 DALL-E 프롬프트로 변환
# dalle.run()을 사용하여 실제 이미지 생성
image_url = dalle.run(chain.invoke({"image_desc": query}))

# 생성된 이미지를 표시합니다.
Image(url=image_url, width=500)

