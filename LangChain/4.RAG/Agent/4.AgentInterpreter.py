from dotenv import load_dotenv
load_dotenv()

# 필요한 모듈 가져오기
from langchain_experimental.tools import PythonAstREPLTool
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from langchain_community.tools import TavilySearchResults

# Python 코드 실행을 위한 REPL 도구 초기화
python_repl = PythonAstREPLTool()

# 웹 검색을 위한 Tavily 도구 초기화
web_search = TavilySearchResults(max_results=3)

# OpenAI의 GPT-4 모델 초기화
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 에이전트 초기화
agent = initialize_agent(
    [python_repl, web_search],  # 사용할 도구 리스트에 web_search 추가
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

from datetime import datetime
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 에이전트에 대한 지시사항 업데이트
agent_instructions = """
당신은 다양한 작업을 수행할 수 있는 AI 어시스턴트입니다. 다음 도구들을 사용할 수 있습니다:

1. Python REPL: Python 코드를 실행하여 복잡한 계산이나 데이터 처리를 수행할 수 있습니다.
   사용법: Python 코드를 직접 작성하여 실행할 수 있습니다.

2. Web Search: Tavily 검색 엔진을 사용하여 웹에서 정보를 검색할 수 있습니다.
   사용법: 검색하고자 하는 쿼리를 입력하면 관련 웹 페이지의 정보를 반환합니다.

현재 날짜: {current_time}

Firebase Admin SDK를 사용하여 Firebase 데이터베이스 데이터를 조회할 수 있습니다.
serviceAccountKey.json 경로는 /Users/yoonjae/Desktop/AI-X/RAG/LangChain/mirrorgram-20713-firebase-adminsdk-u9pdx-c3e12134b4.json 입니다.

서버 엔드포인트 정보:
1. /chat (POST): 사용자와 선택한 페르소나 간의 1:1 대화를 처리합니다.
   파라미터: ChatRequest (persona_name, user_input, user)

2. /personas (GET): 사용 가능한 모든 페르소나의 목록을 반환합니다.
   파라미터: 없음

3. /feed (POST): 새로운 피드 포스트를 생성하고, 이미지 분석을 수행한 후 결과를 저장합니다.
   파라미터: FeedPost (id, image, caption, likes, comments, createdAt, userId, nick, subCommentId)

4. /persona-chat (POST): 두 페르소나 간의 대화를 생성합니다. 지정된 라운드 수만큼 대화를 진행합니다.
   파라미터: PersonaChatRequest (uid, topic, persona1, persona2, rounds)

5. /execute-task (POST): 특정 페르소나의 작업(다른 페르소나와의 상호작용)을 백그라운드에서 실행합니다.
   파라미터: TaskRequest (uid, persona_name, interaction_target, topic, conversation_rounds, time)

6. /generate-user-schedule/{uid} (POST): 특정 사용자의 일일 스케줄을 생성하고 저장합니다. 생성된 스케줄에 따라 페르소나 간 상호작용 작업을 예약합니다.
   파라미터: uid (경로 파라미터)

7. /user-schedule/{uid} (GET): 특정 사용자의 저장된 스케줄을 조회합니다.
   파라미터: uid (경로 파라미터)

작업을 수행할 때 다음 지침을 엄격히 따르세요:
1. 각 작업을 단계별로 수행하고, 각 단계를 명확하게 설명하세요.
2. 각 단계가 완료되면 반드시 다음 형식으로 보고하세요:
   "단계 완료: [완료된 작업 설명]"
3. 추가 작업이 필요한 경우 반드시 다음 형식으로 명시하세요:
   "추가 작업 필요: [다음 작업 설명]"
4. 모든 작업이 완전히 완료되었을 때만 다음과 같이 명확하게 표시하세요:
   "작업 완료: 모든 요청된 작업이 성공적으로 수행되었습니다."
5. 작업 중 오류가 발생하거나 추가 정보가 필요한 경우 다음과 같이 명확하게 표시하세요:
   "오류 발생: [오류 설명]" 또는 "추가 정보 필요: [필요한 정보 설명]"

각 응답에서 위의 형식 중 하나를 반드시 사용하여 현재 상태를 명확히 표시하세요.
사용자의 요청을 정확히 이해하고 수행했는지 확인하세요.
불확실한 점이 있으면 반드시 질문하여 명확히 하세요.

시작하겠습니다. 어떤 작업을 수행해야 할까요?
"""

def run_agent_until_complete(initial_task):
    task_queue = [initial_task]
    results = []

    while task_queue:
        current_task = task_queue.pop(0)
        print(f"현재 작업: {current_task}")

        try:
            result = agent.run(agent_instructions + "\n\n" + current_task)
            results.append(result)
            print(f"작업 결과: {result}")

            # 결과에 '작업 완료'가 포함되어 있는지 확인
            if "작업 완료" in result.lower():
                print("에이전트가 작업 완료를 보고했습니다.")
                break  # 작업 완료 시 루프 종료

            # '추가 작업 필요'가 없고 사용자 입력도 요구하지 않는 경우
            if "추가 작업 필요" not in result:
                user_input = input("작업이 완료되었습니까? 추가 작업이 필요하면 입력해주세요 (완료되었다면 '완료' 입력): ")
                if user_input.lower() == '완료':
                    print("사용자가 작업 완료를 확인했습니다.")
                    break  # 사용자가 완료를 확인하면 루프 종료
                elif user_input.strip():
                    task_queue.append(user_input)
                else:
                    print("추가 작업이 없으므로 프로그램을 종료합니다.")
                    break
            else:
                new_task = result.split("추가 작업 필요: ")[1]
                task_queue.append(new_task)
                print(f"새로운 작업이 추가되었습니다: {new_task}")

        except Exception as e:
            print(f"오류 발생: {e}")
            user_input = input("오류가 발생했습니다. 다시 시도하시겠습니까? (예/아니오): ")
            if user_input.lower() == '예':
                task_queue.append(current_task)
            else:
                print("작업을 중단합니다.")
                break

    return results

# 사용자로부터 초기 작업 입력 받기
initial_task = input("수행할 작업을 입력해주세요: ")

# 에이전트 실행
final_results = run_agent_until_complete(initial_task)

print("프로그램이 종료되었습니다.")
for i, result in enumerate(final_results, 1):
    print(f"작업 {i} 결과: {result}")
