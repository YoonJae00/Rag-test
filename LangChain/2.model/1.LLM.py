from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAI

llm = OpenAI()

result = llm.invoke("한국의 대표적인 관광지 3군데를 추천해주세요.")
print(result)