from dotenv import load_dotenv

load_dotenv()

from langchain_teddynote import logging

# 프로젝트 이름을 입력합니다.
logging.langsmith("personas-chat")

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_community.tools import TavilySearchResults

web_search = TavilySearchResults(max_results=1)


chat_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "당신은 노래 제목 알아내기 전문 AI 어시스턴트입니다. 당신의 임무는 노래 제목을 알아내서 '노래제목 가사' 형식으로 출력합니다",
        ),
        MessagesPlaceholder(variable_name="conversation2"),
        ("human", "'노래제목 가사' 형식으로 출력합니다."),
    ]
)

chat_prompt2 = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "당신은 노래 가사를 해석하는 전문 AI 어시스턴트입니다. 당신의 임무는 노래의 감정을 해석하는 것입니다.",
        ),
        MessagesPlaceholder(variable_name="conversation3"),
        ("human", "노래의 감정을 해석해줘"),
    ]
)
def format_search_result(result):
    # 검색 결과를 적절한 메시지 형식으로 변환
    return {"conversation3": [{"role": "system", "content": f"검색 결과: {result[0]['content']}"}]}

from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
llm = ChatOpenAI(model="gpt-4o")
# chain 생성
# chain = chat_prompt | llm | StrOutputParser() | web_search | RunnablePassthrough() | chat_prompt2
chain = (
    chat_prompt 
    | llm 
    | StrOutputParser() 
    | web_search 
    | format_search_result  # 검색 결과를 적절한 형식으로 변환
    | RunnablePassthrough.assign(conversation3=lambda x: x)  # 변환된 결과를 conversation3에 할당
    | chat_prompt2 
    | llm
)

# chain 실행 및 결과확인
rusult =chain.invoke(
    {
        "word_count": 2,
        "conversation2": [
            (
                "human",
                "잠깐시갈될까? 그 한마디보다 지금 당장 널 보러 가고 싶은데",
            ),
        ],
    }
)

print(rusult)