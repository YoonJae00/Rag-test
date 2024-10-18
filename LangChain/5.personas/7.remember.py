from dotenv import load_dotenv
load_dotenv()

from langchain_teddynote import logging

# 프로젝트 이름을 입력합니다.
logging.langsmith("gen-image")

from langchain.memory import ConversationSummaryMemory
from langchain_openai import ChatOpenAI

memory = ConversationSummaryMemory(
    llm=ChatOpenAI(temperature=0), return_messages=True)

memory.save_context(
    inputs={"human": "유럽 여행 패키지의 가격은 얼마인가요?"},
    outputs={
        "ai": "유럽 14박 15일 패키지의 기본 가격은 3,500유로입니다. 이 가격에는 항공료, 호텔 숙박비, 지정된 관광지 입장료가 포함되어 있습니다. 추가 비용은 선택하신 옵션 투어나 개인 경비에 따라 달라집니다."
    },
)
memory.save_context(
    inputs={"human": "여행 중에 방문할 주요 관광지는 어디인가요?"},
    outputs={
        "ai": "이 여행에서는 파리의 에펠탑, 로마의 콜로세움, 베를린의 브란덴부르크 문, 취리히의 라이네폴 등 유럽의 유명한 관광지들을 방문합니다. 각 도시의 대표적인 명소들을 포괄적으로 경험하실 수 있습니다."
    },
)
memory.save_context(
    inputs={"human": "여행자 보험은 포함되어 있나요?"},
    outputs={
        "ai": "네, 모든 여행자에게 기본 여행자 보험을 제공합니다. 이 보험은 의료비 지원, 긴급 상황 발생 시 지원 등을 포함합니다. 추가적인 보험 보장을 원하시면 상향 조정이 가능합니다."
    },
)
memory.save_context(
    inputs={
        "human": "항공편 좌석을 비즈니스 클래스로 업그레이드할 수 있나요? 비용은 어떻게 되나요?"
    },
    outputs={
        "ai": "항공편 좌석을 비즈니스 클래스로 업그레이드하는 것이 가능합니다. 업그레이드 비용은 왕복 기준으로 약 1,200유로 추가됩니다. 비즈니스 클래스에서는 더 넓은 좌석, 우수한 기내식, 그리고 추가 수하물 허용량 등의 혜택을 제공합니다."
    },
)
memory.save_context(
    inputs={"human": "패키지에 포함된 호텔의 등급은 어떻게 되나요?"},
    outputs={
        "ai": "이 패키지에는 4성급 호텔 숙박이 포함되어 있습니다. 각 호텔은 편안함과 편의성을 제공하며, 중심지에 위치해 관광지와의 접근성이 좋습니다. 모든 호텔은 우수한 서비스와 편의 시설을 갖추고 있습니다."
    },
)
memory.save_context(
    inputs={"human": "식사 옵션에 대해 더 자세히 알려주실 수 있나요?"},
    outputs={
        "ai": "이 여행 패키지는 매일 아침 호텔에서 제공되는 조식을 포함하고 있습니다. 점심과 저녁 식사는 포함되어 있지 않아, 여행자가 자유롭게 현지의 다양한 음식을 경험할 수 있는 기회를 제공합니다. 또한, 각 도시별로 추천 식당 리스트를 제공하여 현지의 맛을 최대한 즐길 수 있도록 도와드립니다."
    },
)
memory.save_context(
    inputs={"human": "패키지 예약 시 예약금은 얼마인가요? 취소 정책은 어떻게 되나요?"},
    outputs={
        "ai": "패키지 예약 시 500유로의 예약금이 필요합니다. 취소 정책은 예약일로부터 30일 전까지는 전액 환불이 가능하며, 이후 취소 시에는 예약금이 환불되지 않습니다. 여행 시작일로부터 14일 전 취소 시 50%의 비용이 청구되며, 그 이후는 전액 비용이 청구됩니다."
    },
)

# 저장된 메모리 확인
print(memory.load_memory_variables({})["history"])
