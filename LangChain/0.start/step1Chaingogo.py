from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. 컴포넌트 정의
prompt = ChatPromptTemplate.from_template("지구과학에서 {topic}에 대해 간단히 설명해주세요.")
model = ChatOpenAI(model="gpt-4o-mini")
output_parser = StrOutputParser()

# 2. 체인 생성
chain = prompt | model | output_parser

# 3. invoke 메소드 사용
# 일반적인 결과 출력
# result = chain.invoke({"topic": "지구 자전"})
# print("invoke 결과:", result)

# batch 메소드 사용
# 한가지 프롬포트로 여러개의 인풋을 처리
# topics = ["지구 공전", "화산 활동", "대륙 이동"]
# results = chain.batch([{"topic": t} for t in topics])
# for topic, result in zip(topics, results):
#     print(f"{topic} 설명: {result[:50]}...")  # 결과의 처음 50자만 출력

# stream 메소드 사용
# 실시간으로 결과를 출력
stream = chain.stream({"topic": "지진"})
print("stream 결과:")
for chunk in stream:
    print(chunk, end="", flush=True)
print()

