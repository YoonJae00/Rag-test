# API 키를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv

# API 키 정보 로드
load_dotenv()

# LangSmith 추적을 설정합니다. https://smith.langchain.com
# !pip install -qU langchain-teddynote
from langchain_teddynote import logging

# 프로젝트 이름을 입력합니다.
logging.langsmith("CH15-Tools")
import warnings

from langchain.tools import tool


# 데코레이터를 사용하여 함수를 도구로 변환합니다.
@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


@tool
def multiply_numbers(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

from langchain_teddynote.tools import GoogleNews

# 도구 생성
news_tool = GoogleNews()
# 키워드로 뉴스 검색
print(news_tool.search_by_keyword("삼성전자 주가", k=5))