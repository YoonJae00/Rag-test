# API KEY를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv

# API KEY 정보로드
load_dotenv()

import re
import requests
from bs4 import BeautifulSoup
from langchain.agents import tool
from langchain_teddynote.tools import GoogleNews

# 도구 생성
news_tool = GoogleNews()

@tool
def google_news(query: str) -> str:
    """Search Google News for the given query."""
    return news_tool.search_by_keyword(query)


# 도구를 정의합니다.
@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)


@tool
def add_function(a: float, b: float) -> float:
    """Adds two numbers together."""
    return a + b


# @tool
# def naver_news_crawl(news_url: str) -> str:
#     """Crawls a 네이버 (naver.com) news article and returns the body content."""
#     # HTTP GET 요청 보내기
#     response = requests.get(news_url)

#     # 요청이 성공했는지 확인
#     if response.status_code == 200:
#         # BeautifulSoup을 사용하여 HTML 파싱
#         soup = BeautifulSoup(response.text, "html.parser")

#         # 원하는 정보 추출
#         title = soup.find("h2", id="title_area").get_text()
#         content = soup.find("div", id="contents").get_text()
#         cleaned_title = re.sub(r"\n{2,}", "\n", title)
#         cleaned_content = re.sub(r"\n{2,}", "\n", content)
#     else:
#         print(f"HTTP 요청 실패. 응답 코드: {response.status_code}")

#     return f"{cleaned_title}\n{cleaned_content}"

@tool
def summarize_news(news_content: str) -> str:
    """Summarizes a news article."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant that summarizes news articles."),
            ("user", "{input}"),
        ]
    )
    result = llm.invoke(prompt.format(input=news_content))
    return result


from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

# Agent프롬프트 생성
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are very powerful assistant, but don't know current events",
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# 모델 생성
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor

# 이전에 정의한 도구 사용
tools = [get_word_length, add_function, google_news, summarize_news]

# Agent 생성
agent = create_tool_calling_agent(llm, tools, prompt)

# AgentExecutor 생성
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
)
# Agent 실행
# result = agent_executor.invoke({"input": "How many letters in the word `teddynote`?"})

# 결과 확인
# print(result["output"])
# Agent 실행
# result = agent_executor.invoke({"input": "114.5 + 121.2 의 계산 결과는?"})

# 결과 확인
# print(result["output"])
# Agent 실행
# result = agent_executor.invoke(
    # {"input": "114.5 + 121.2 + 34.2 + 110.1 의 계산 결과는?"}
# )

# 결과 확인
# print(result["output"])
print("==========\n")
# print(114.5 + 121.2 + 34.2 + 110.1)
# result = agent_executor.invoke(
#     {
#         "input": "뉴스 기사를 요약해 줘: https://n.news.naver.com/mnews/hotissue/article/092/0002347672?type=series&cid=2000065"
#     }
# )
# print(result["output"])

result = agent_executor.invoke(
    {"input": "비트코인의 현재 가격은?"}
)
print(result["output"])