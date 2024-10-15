from dotenv import load_dotenv
load_dotenv()

# 필요한 모듈 가져오기
from langchain_experimental.tools import PythonAstREPLTool
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI

# Python 코드 실행을 위한 REPL 도구 초기화
python_repl = PythonAstREPLTool()

# OpenAI의 GPT-4 모델 초기화
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 에이전트 초기화
# ZERO_SHOT_REACT_DESCRIPTION 타입의 에이전트를 사용하며, verbose=True로 설정하여 상세 출력을 얻습니다.
agent = initialize_agent(
    [python_repl],  # 사용할 도구 리스트
    llm,  # 사용할 언어 모델
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # 에이전트 타입
    verbose=True  # 상세 출력 설정
)

# 에이전트에게 작업 지시 및 실행
result = agent.run("""
Firebase Admin SDK를 사용하여 Firebase 데이터베이스의 'feeds' 컬렉션에서 'UN9gP1sOc3Q2iDN3qRgw' 문서에 댓글을 달아주세요.
serviceAccountKey.json 경로는 /Users/yoonjae/Desktop/AI-X/RAG/LangChain/mirrorgram-20713-firebase-adminsdk-u9pdx-c3e12134b4.json 입니다. 
""")

# 결과 출력
print(result)
