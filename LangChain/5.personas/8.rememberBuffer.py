from dotenv import load_dotenv
load_dotenv()

from langchain_teddynote import logging

# 프로젝트 이름을 입력합니다.
logging.langsmith("remember-buffer")

from langchain_openai import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory

llm = ChatOpenAI()

memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=200,  # 요약의 기준이 되는 토큰 길이를 설정합니다.
    return_messages=True,
)
# 여행사 직원과 고객들 간의 대화 시뮬레이션
memory.save_context(
    inputs={"김여행": "안녕하세요, 유럽 여행 패키지에 대해 문의드리고 싶습니다. 가격이 어떻게 되나요?"},
    outputs={"여행사직원": "안녕하세요 김여행 님, 유럽 14박 15일 패키지의 기본 가격은 3,500유로입니다. 이 가격에는 항공료, 호텔 숙박비, 지정된 관광지 입장료가 포함되어 있습니다. 추가 비용은 선택하신 옵션 투어나 개인 경비에 따라 달라질 수 있습니다."}
)

memory.save_context(
    inputs={"이관광": "저는 주요 관광지에 대해 궁금합니다. 어떤 곳들을 방문하나요?"},
    outputs={"여행사직원": "이관광 님, 이 여행에서는 파리의 에펠탑, 로마의 콜로세움, 베를린의 브란덴부르크 문, 취리히의 라이네폴 등 유럽의 유명한 관광지들을 방문합니다. 각 도시의 대표적인 명소들을 포괄적으로 경험하실 수 있습니다."}
)

memory.save_context(
    inputs={"박안전": "여행자 보험은 따로 들어야 하나요?"},
    outputs={"여행사직원": "박안전 님, 걱정 마세요. 모든 여행자에게 기본 여행자 보험을 제공합니다. 이 보험은 의료비 지원, 긴급 상황 발생 시 지원 등을 포함합니다. 추가적인 보험 보장을 원하시면 상향 조정도 가능합니다."}
)

memory.save_context(
    inputs={"최호화": "항공편을 비즈니스 클래스로 업그레이드하고 싶은데 가능할까요? 비용은 어떻게 되나요?"},
    outputs={"여행사직원": "최호화 님, 네 가능합니다. 항공편 좌석을 비즈니스 클래스로 업그레이드하는 비용은 왕복 기준으로 약 1,200유로가 추가됩니다. 비즈니스 클래스에서는 더 넓은 좌석, 우수한 기내식, 그리고 추가 수하물 허용량 등의 혜택을 누리실 수 있습니다."}
)

memory.save_context(
    inputs={"정숙박": "호텔은 어떤 등급인가요? 좋은 곳에서 묵고 싶어요."},
    outputs={"여행사직원": "정숙박 님, 안심하세요. 이 패키지에는 4성급 호텔 숙박이 포함되어 있습니다. 각 호텔은 편안함과 편의성을 제공하며, 중심지에 위치해 관광지와의 접근성이 좋습니다. 모든 호텔은 우수한 서비스와 편의 시설을 갖추고 있어 만족스러운 숙박을 경험하실 수 있습니다."}
)

memory.save_context(
    inputs={"김여행": "죄송하지만 제가 아까 문의드린 가격에 대해 다시 한 번 확인하고 싶어요. 3,500유로가 맞나요?"},
    outputs={"여행사직원": "네, 김여행 님. 맞습니다. 기본 패키지 가격은 3,500유로입니다. 이 가격에는 앞서 말씀드린 대로 항공료, 호텔 숙박비, 주요 관광지 입장료가 모두 포함되어 있습니다. 추가 옵션이나 개인 경비는 별도입니다."}
)

# 메모리에 저장된 대화내용 확인
print(memory.load_memory_variables({})["history"])