# API 키를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv

# API 키 정보 로드
load_dotenv()

# API 키를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv

# API 키 정보 로드
load_dotenv()
import warnings

# 경고 메시지 무시
warnings.filterwarnings("ignore")
from langchain_experimental.tools import PythonREPLTool

# 파이썬 코드를 실행하는 도구를 생성합니다.
python_tool = PythonREPLTool()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda


# 파이썬 코드를 실행하고 중간 과정을 출력하고 도구 실행 결과를 반환하는 함수
def print_and_execute(code, debug=True):
    if debug:
        print("CODE:")
        print(code)
    return python_tool.invoke(code)


# 파이썬 코드를 작성하도록 요청하는 프롬프트
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are Raymond Hetting, an expert python programmer, well versed in meta-programming and elegant, concise and short but well documented code. You follow the PEP8 style guide. "
            "Return only the code, no intro, no explanation, no chatty, no markdown, no code block, no nothing. Just the code.",
        ),
        ("human", "{input}"),
    ]
)
# LLM 모델 생성
llm = ChatOpenAI(model="gpt-4o", temperature=0)

from langchain_community.tools import TavilySearchResults

# 도구 생성
tool = TavilySearchResults(
    max_results=6,
    include_answer=True,
    include_raw_content=True,
    # include_images=True,
    # search_depth="advanced", # or "basic"
    include_domains=["github.io", "wikidocs.net"],
    # exclude_domains = []
)

# 도구 실행
print(tool.invoke({"query": "LangChain Tools 에 대해서 알려주세요"}))
