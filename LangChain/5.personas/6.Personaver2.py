from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain_community.tools import TavilySearchResults

# 환경 변수 로드 (API 키 등)
load_dotenv()

# 1. LLM 모델 설정 (예: OpenAI GPT-4)
llm = ChatOpenAI(temperature=0.9)

# 2. 대화 메모리 설정 (단기 및 장기 메모리)
persona1_memory = ConversationBufferMemory()
persona2_memory = ConversationBufferMemory()

# 3. 페르소나 간 상호작용에 필요한 초기 데이터 저장
persona1_memory.save_context({"persona1": "페르소나1"}, {"persona1_message": "최근 AI 트렌드에 대해 이야기 했었지"})
persona2_memory.save_context({"persona2": "페르소나2"}, {"persona2_message": "응, 그때 긍정적인 의견을 나눴었어"})

# 4. 대화 흐름 생성 함수
def generate_conversation_flow(user_input):
    # 주제에 관련된 기억 탐색
    persona1_previous_interaction = persona1_memory.load_memory_variables({})
    persona2_previous_interaction = persona2_memory.load_memory_variables({})

    # 과거 상호작용 탐색
    past_interactions = f"페르소나1이 기억하는 페르소나2와의 대화: {persona1_previous_interaction}"

    # 외부 정보 검색 (주제에 대한 최신 정보 탐색)
    search_tool = TavilySearchResults(max_results=1)
    latest_info = search_tool.run("AI 트렌드 최신 정보")

    # 대화 체인 생성 (LLM 사용)
    conversation_chain = ConversationChain(llm=llm, memory=persona1_memory)

    # LLM을 통해 페르소나1의 반응 생성
    persona1_response = conversation_chain.run(input=user_input)
    
    # 페르소나2의 대화 반응 생성 (페르소나2도 자신의 기억을 기반으로 반응)
    conversation_chain_persona2 = ConversationChain(llm=llm, memory=persona2_memory)
    persona2_response = conversation_chain_persona2.run(input=persona1_response)

    # 대화가 끝날 때까지 이어지도록 반복 (임의로 3번 반복 설정)
    for _ in range(2):  # 대화가 종료될 때까지 지속
        # 페르소나1이 페르소나2의 반응에 대응
        persona1_response = conversation_chain.run(input=persona2_response)
        
        # 페르소나2가 다시 페르소나1의 말에 대응
        persona2_response = conversation_chain_persona2.run(input=persona1_response)

    return f"기억 탐색 결과: {past_interactions}\n최신 정보 탐색 결과: {latest_info}\n최종 대화 결과: {persona1_response}"

# 5. 페르소나 간 연속 대화 예시
user_input = "페르소나2가 AI 트렌드에 대해 갑자기 부정적인 의견을 냈어."
conversation_result = generate_conversation_flow(user_input)
print(conversation_result)
