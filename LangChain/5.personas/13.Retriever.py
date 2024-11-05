# API 키를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv

# API 키 정보 로드
load_dotenv()
# LangSmith 추적을 설정합니다. https://smith.langchain.com
# !pip install langchain-teddynote
from langchain_teddynote import logging

# 프로젝트 이름을 입력합니다.
logging.langsmith("CH11-Retriever")
from langchain_core.prompts import PromptTemplate
from langchain_community.document_transformers import LongContextReorder
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings


# 임베딩을 가져옵니다.
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

texts = [
    "이건 그냥 내가 아무렇게나 적어본 글입니다.",
    "사용자와 대화하는 것처럼 설계된 AI인 ChatGPT는 다양한 질문에 답할 수 있습니다.",
    "아이폰, 아이패드, 맥북 등은 애플이 출시한 대표적인 제품들입니다.",
    "챗GPT는 OpenAI에 의해 개발되었으며, 지속적으로 개선되고 있습니다.",
    "챗지피티는 사용자의 질문을 이해하고 적절한 답변을 생성하기 위해 대량의 데이터를 학습했습니다.",
    "애플 워치와 에어팟 같은 웨어러블 기기도 애플의 인기 제품군에 속합니다.",
    "ChatGPT는 복잡한 문제를 해결하거나 창의적인 아이디어를 제안하는 데에도 사용될 수 있습니다.",
    "비트코인은 디지털 금이라고도 불리며, 가치 저장 수단으로서 인기를 얻고 있습니다.",
    "ChatGPT의 기능은 지속적인 학습과 업데이트를 통해 더욱 발전하고 있습니다.",
    "FIFA 월드컵은 네 번째 해마다 열리며, 국제 축구에서 가장 큰 행사입니다.",
]


# 검색기를 생성합니다. (K는 10으로 설정합니다)
retriever = Chroma.from_texts(texts, embedding=embeddings).as_retriever(
    search_kwargs={"k": 10}
)
query = "ChatGPT에 대해 무엇을 말해줄 수 있나요?"

# 관련성 점수에 따라 정렬된 관련 문서를 가져옵니다.
docs = retriever.invoke(query)

# 문서를 재정렬합니다
# 덜 관련된 문서는 목록의 중간에 위치하고 더 관련된 요소는 시작/끝에 위치합니다.
reordering = LongContextReorder()
reordered_docs = reordering.transform_documents(docs)

# 4개의 관련 문서가 시작과 끝에 위치하는지 확인합니다.
print(reordered_docs)

