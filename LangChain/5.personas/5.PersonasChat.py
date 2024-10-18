from dotenv import load_dotenv
load_dotenv()

from langchain_teddynote import logging

# 프로젝트 이름을 입력합니다.
logging.langsmith("personas-chat")

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", temperature=0)

from langchain_core.prompts import PromptTemplate

from datetime import datetime
# 날짜를 반환하는 함수 정의
def get_today():
    return datetime.now().strftime("%B %d")

prompt = PromptTemplate(
    template="오늘의 날짜는 {today} 입니다. 오늘이 생일인 유명인 {n}명을 나열해 주세요. 생년월일을 표기해주세요.",
    input_variables=["n"],
    partial_variables={
        "today": get_today  # dictionary 형태로 partial_variables를 전달
    },
)
