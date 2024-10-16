from dotenv import load_dotenv
load_dotenv()

from langchain_teddynote import logging

# 프로젝트 이름을 입력합니다.
logging.langsmith("langchain-test")

from langchain_community.tools import TavilySearchResults

query = "10월 15일 낢씨 알려줘."

web_search = TavilySearchResults(max_results=1)

search_results = web_search.invoke(query)

for result in search_results:
    print(result)
    print("-" * 100)
