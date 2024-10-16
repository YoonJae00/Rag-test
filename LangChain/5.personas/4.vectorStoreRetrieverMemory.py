# API KEY를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv

# API KEY 정보로드
load_dotenv()

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.memory import VectorStoreRetrieverMemory
from langchain_core.documents import Document


# OpenAI 임베딩 모델 초기화
embedding_model = OpenAIEmbeddings()

# Chroma 벡터 스토어 초기화
vectorstore = Chroma(
    embedding_function=embedding_model,
    persist_directory="./chroma_memory_db"
)

# VectorStoreRetrieverMemory 초기화
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
memory = VectorStoreRetrieverMemory(retriever=retriever)

# 대화 내용을 저장하는 함수
def save_conversation(human_input, ai_output):
    memory.save_context(
        {"human": human_input},
        {"ai": ai_output}
    )
    # Chroma에 직접 저장 (메타데이터 포함)
    vectorstore.add_documents([
        Document(
            page_content=f"Human: {human_input}\nAI: {ai_output}",
            metadata={"type": "conversation"}
        )
    ])

# 예시 대화 저장
save_conversation(
    "안녕하세요, 오늘 면접에 참석해주셔서 감사합니다. 자기소개 부탁드립니다.",
    "안녕하세요. 저는 컴퓨터 과학을 전공한 신입 개발자입니다. 대학에서는 주로 자바와 파이썬을 사용했으며, 최근에는 웹 개발 프로젝트에 참여하여 실제 사용자를 위한 서비스를 개발하는 경험을 했습니다."
)

save_conversation(
    "프로젝트에서 어떤 역할을 맡았나요?",
    "제가 맡은 역할은 백엔드 개발자였습니다. 사용자 데이터 처리와 서버 로직 개발을 담당했으며, RESTful API를 구현하여 프론트엔드와의 통신을 담당했습니다. 또한, 데이터베이스 설계에도 참여했습니다."
)

save_conversation(
    "팀 프로젝트에서 어려움을 겪었던 경험이 있다면 어떻게 해결했나요?",
    "프로젝트 초기에 의사소통 문제로 몇 가지 어려움이 있었습니다. 이를 해결하기 위해 저희 팀은 정기적인 미팅을 갖고 각자의 진행 상황을 공유했습니다. 또한, 문제가 발생했을 때는 적극적으로 의견을 나누고, 합리적인 해결책을 찾기 위해 노력했습니다."
)

save_conversation(
    "개발자로서 자신의 강점은 무엇이라고 생각하나요?",
    "제 강점은 빠른 학습 능력과 문제 해결 능력입니다. 새로운 기술이나 도구를 빠르게 습득할 수 있으며, 복잡한 문제에 직면했을 때 창의적인 해결책을 제시할 수 있습니다. 또한, 팀워크를 중시하며 동료들과 협력하는 것을 중요하게 생각합니다."
)

# 저장된 대화 검색 예시
query = "개발자의 강점은 무엇인가요?"
results = vectorstore.similarity_search(query, k=1)

for doc in results:
    print(f"검색 결과: {doc.page_content}")
    print(f"메타데이터: {doc.metadata}")
    print("---")

# 메모리에서 관련 정보 검색
print(memory.load_memory_variables({"prompt": query}))

# 변경사항을 디스크에 저장
vectorstore.persist()
