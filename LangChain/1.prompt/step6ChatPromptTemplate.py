from dotenv import load_dotenv

load_dotenv()

from langchain_teddynote import logging

# 프로젝트 이름을 입력합니다.
logging.langsmith("personas-chat")

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")

from langchain_core.prompts import ChatPromptTemplate

chat_prompt = ChatPromptTemplate.from_template("{country}의 수도는 어디인가요?")

from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate.from_messages(
       [
           ("system", "당신은 친절하지만 비격식적인 AI 어시스턴트입니다. 당신의 이름은 {name}이며, '알빠노!'와 같은 표현을 사용합니다."),
           ("human", "안녕!"),
           ("ai", "알빠노! 무엇을 도와드려?"),
           ("human", "{user_input}"),
       ]
   )

chain = chat_template | llm
result = chain.invoke({"name": "Teddy", "user_input": "당신의 이름은 무엇입니까?"}).content
print(result)