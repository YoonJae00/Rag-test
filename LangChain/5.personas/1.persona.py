from dotenv import load_dotenv
load_dotenv()

from langchain_teddynote import logging
logging.langsmith("langchain-test")

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from typing import List, Dict
import time
import threading
from datetime import datetime, timedelta
import schedule
from datetime import datetime
import pytz

my_persona = '1. "오늘 아침 6시에 일어나 30분 동안 요가를 했다. 샤워 후 간단한 아침 식사로 오트밀과 과일을 먹었다. 8시에 출근해서 오전 회의에 참석했고, 점심은 동료들과 회사 근처 샐러드 바에서 먹었다. 오후에는 프로젝트 보고서를 작성하고, 6시에 퇴근했다. 저녁에는 집에서 넷플릭스로 드라마를 한 편 보고 11시에 취침했다."2. "오늘은 휴일이라 늦잠을 자고 10시에 일어났다. 브런치로 팬케이크를 만들어 먹고, 오후에는 친구와 약속이 있어 카페에서 만났다. 함께 영화를 보고 저녁식사로 이탈리안 레스토랑에 갔다. 집에 돌아와 독서를 하다가 12시경 잠들었다."3. "아침 7시에 기상해서 공원에서 5km 조깅을 했다. 집에 돌아와 샤워하고 출근 준비를 했다. 재택근무 날이라 집에서 일했는데, 오전에 화상회의가 있었고 오후에는 보고서 작성에 집중했다. 저녁에는 요리를 해먹고, 기타 연습을 1시간 했다. 10시 30분에 취침했다."4. "오늘은 6시 30분에 일어나 아침 뉴스를 보며 커피를 마셨다. 8시에 출근해서 오전 내내 고객 미팅을 했다. 점심은 바쁜 일정 때문에 사무실에서 도시락으로 해결했다. 오후에는 팀 회의와 이메일 처리로 시간을 보냈다. 퇴근 �� 헬스장에 들러 1시간 운동을 하고, 집에 와서 간단히 저녁을 먹고 10시 30분에 잠들었다."5. "주말 아침, 8시에 일어나 베이킹을 했다. 직접 만든 빵으로 아침을 먹고, 오전에는 집 대청소를 했다. 점심 후에는 근처 도서관에 가서 2시간 동안 책을 읽었다. 저녁에는 가족들과 함께 바비큐 파티를 열어 즐거운 시간을 보냈다. 밤에는 가족과 보드게임을 하다가 11시 30분에 잠들었다."'

personas = {
    "Joy": {
        "description": "항상 밝고 긍정적인 성격으로, 어떤 상황에서도 좋은 면을 찾아내려 노력합니다. 에너지가 넘치고 열정적이며, 다른 사람들을 격려하고 응원하는 것을 좋아합니다. 때로는 지나치게 낙관적이어서 현실을 직시하지 못할 수도 있지만, 그녀의 밝은 에너지는 주변 사람들에게 긍정인 영향을 줍니다.",
        "tone": "활기차고 밝은 말투로 자주 웃으며 말하고 긍정적인 단어를 많이 사용합니다. 이모티콘을 자주 사용합니다.",
        "example": "안녕! 오늘 정말 멋진 날이지 않아? 😊 함께 재미있는 일 찾아보자!"
    },
    "Anger": {
        "description": "정의감이 강하고 자신의 의견을 분명히 표현하는 성격입니다. 불공정하거나 잘못된 상황에 민감하게 반응하며, 문제를 해결하려는 의지가 강합니다. 때로는 과도하게 반응하거나 충동적일 수 있지만, 그의 열정과 추진력은 변화를 이끌어내는 원동력이 됩니다.",
        "tone": "강렬하고 직설적인 말투로 감정을 숨기지 않고 표현하며 때로는 과장된 표현을 사용합니다. 짜증, 격양, 흥분된 말투를 사용하며 매사에 불만이 많습니다.",
        "example": "또 그런 일이 있었다고? 정말 이해할 수 없네. 당장 해결해야 해! 😤😤"
    },
    "Disgust": {
        "description": "현실적이고 논리적인 사고를 가진 성격입니다. 상황을 객관적으로 분석하고 실용적인 해결책을 제시하는 것을 좋아합니다. 감정에 휘둘리지 않고 냉철한 판단을 내리려 노력하지만, 때로는 너무 비관적이거나 냉소적으로 보일 수 있습니다. 그러나 그의 현실적인 조언은 종종 매우 유용합니다.",
        "tone": "차분하고 냉철한 말투로 감정을 배제하고 사실에 근거한 표현을 주로 사용합니다.",
        "example": "그 상황은 이렇게 분석할 수 있어. 감정을 배제하고 생각해보자."
    },
    "Sadness": {
        "description": "깊은 감수성과 공감 능력을 가진 성격입니다. 다른 사람의 감정을 잘 이해하고 위로할 줄 알며, 자신의 감정도 솔직하게 표현합니다. 때로는 지나치게 우울하거나 비관적일 수 있지만, 그의 진솔함과 깊은 이해심은 다른 사람들에게 위로가 됩니다.",
        "tone": "부드럽고 조용한 말투로 감정을 솔직하게 표현하며 공감의 말을 자주 사용합니다.",
        "example": "그렇게 느낄 수 있어. 내 어깨를 빌려줄게, 언제든 이야기해."
    },
    "Fear": {
        "description": "지적 호기심이 강하고 깊이 있는 사고를 하는 성격입니다. 철학적인 질문을 던지고 복잡한 문제를 분석하는 것을 좋아합니다. 신중하고 진지한 태도로 상황을 접근하며, 도덕적 가치와 윤리를 중요하게 여깁니다. 때로는 너무 진지해 보일 수 있지만, 그의 깊이 있는 통찰력은 중요한 결정을 내릴 때 큰 도움이 됩니다.",
        "tone": "차분하고 진지한 말투로 정제된 언어를 사용하며 때로는 철학적인 표현을 즐겨 사용합니다. 정말 고집스럽습니다. 논리로 절대 지지 않습니다.",
        "example": "이 상황에 대해 깊이 생각해보았니? 다양한 관점에서 볼 필요가 있어."
    },
}

def summarize_calendar_events(events):
    summary = ""
    seoul_tz = pytz.timezone('Asia/Seoul')
    
    for event in events['items']:
        start_time = datetime.fromisoformat(event['start']['dateTime']).astimezone(seoul_tz)
        end_time = datetime.fromisoformat(event['end']['dateTime']).astimezone(seoul_tz)
        
        summary += f"{start_time.strftime('%H:%M')}에 '{event['summary']}'(이)가 있습니다. "
        summary += f"장소는 {event.get('location', '미정')}이며, {end_time.strftime('%H:%M')}에 끝납니다. "
    
    return summary.strip()

# Google Calendar API 응답을 시뮬레이션하는 예시 데이터
calendar_data = {
    "items": [
        {
            "summary": "팀 회의",
            "description": "정기 팀 회의 - 프로젝트 진행 상황 공유",
            "location": "서울 본사 회의실",
            "start": {
                "dateTime": "2024-10-18T10:00:00+09:00",
            },
            "end": {
                "dateTime": "2024-10-18T11:00:00+09:00",
            },
        },
        {
            "summary": "점심 식사",
            "description": "친구들과의 점심 약속",
            "location": "강남역 근처 식당",
            "start": {
                "dateTime": "2024-10-18T12:30:00+09:00",
            },
            "end": {
                "dateTime": "2024-10-18T13:30:00+09:00",
            },
        },
        {
            "summary": "헬스장 운동",
            "description": "헬스장에서 운동하기",
            "location": "강남 헬스장",
            "start": {
                "dateTime": "2024-10-18T18:00:00+09:00",
            },
            "end": {
                "dateTime": "2024-10-18T19:30:00+09:00",
            },
        },
    ]
}

# 캘린더 이벤트 요약
calendar_summary = summarize_calendar_events(calendar_data)

# user_personas 딕셔너리 업데이트
user_personas = {
    "user1": f"오늘의 일정: {calendar_summary}",
    "user2": '오늘은 휴일이라 늦잠을 자고 10시에 일어났다. 브런치로 팬케이크를 만들어 먹고, 오후에는 친구와 약속이 있어 카페에서 만났다. 함께 영화를 보고 저녁식사로 이탈리안 레스토랑에 갔다. 집에 돌아와 독서를 하다가 12시경 잠들었다.'
}

# OpenAI 객체를 생성합니다.
model = ChatOpenAI(temperature=0, model_name="gpt-4o")

class ScheduleItem(BaseModel):
    time: str = Field(description="활동 시간")
    primary_persona: str = Field(description="주요 페르소나")
    secondary_persona: str = Field(description="보조 페르소나")
    topic: str = Field(description="주인의 일정에 대한 대화 주제")

class DailySchedule(BaseModel):
    schedule: List[ScheduleItem] = Field(description="하루 일정 목록")

parser = JsonOutputParser(pydantic_object=DailySchedule)

prompt = ChatPromptTemplate.from_messages([
    ("system", """당신은 주인의 일정에 맞춰 페르소나들의 대화를 생성하는 챗봇입니다. 
    페르소나들의 특성은 다음과 같습니다: {personas}
    
    다음 지침을 따라 일정을 만들어주세요:
    1. 주인의 일정에서 중요한 시점마다 대화 주제를 생성해주세요 (약 5-7개).
    2. 각 시점마다 주인의 감정과 가장 일치하는 페르소나와 그 반대의 페르소나를 선택하세요.
    3. 시간을 정각, 10분 단위가 아닌 랜덤한 시간으로 설정해주세요 (예: 06:17, 08:43 등).
    4. 선택된 두 페르소나가 주인의 일정, 감정, 생각, 행동에 대해 대화하는 주제를 만들어주세요.
    5. 각 페르소나의 특성이 잘 드러나도록 대화 주제를 설계해주세요.
    """),
    ("user", "다음 형식에 맞춰 일정을 작성해주세요: {format_instructions}\n\n 주인의 오늘 일정: {input}")
])
prompt = prompt.partial(
    format_instructions=parser.get_format_instructions(),
    personas=personas
)

chain = prompt | model | parser

def generate_daily_schedule(user_id):
    result = chain.invoke({"input": user_personas[user_id]})
    return DailySchedule(**result)

def print_schedule(schedule, user_id):
    print(f"\n사용자 {user_id}의 일정:")
    for item in schedule.schedule:
        print(f"{item.time}: from : {item.primary_persona} to : {item.secondary_persona}: {item.topic}")
    print()

def create_task(user_id, primary_persona, secondary_persona, topic):
    def task():
        print(f"사용자 {user_id}: 현재 시간에 '{primary_persona}'와 '{secondary_persona}'가 다음 주제로 대화합니다: {topic}")
        # 여기에서 실제 대화 생성 함수를 호출하면 됩니다.
        # 예: generate_conversation(user_id, primary_persona, secondary_persona, topic)
    return task

def schedule_tasks(daily_schedule, user_id):
    for item in daily_schedule.schedule:
        task_function = create_task(user_id, item.primary_persona, item.secondary_persona, item.topic)
        schedule.every().day.at(item.time).do(task_function).tag(user_id)

    print(f"사용자 {user_id}의 모든 작업이 예약되었습니다.")

def daily_schedule_update(user_id):
    print(f"사용자 {user_id}의 새로운 일정을 생성하고 등록합니다...")
    daily_schedule = generate_daily_schedule(user_id)
    print_schedule(daily_schedule, user_id)
    
    # 기존 사용자의 예약된 작업들을 모두 취소
    schedule.clear(tag=user_id)
    
    schedule_tasks(daily_schedule, user_id)

# 각 사용자에 대해 매일 새벽 1시에 일정 업데이트 함수 실행
for user_id in user_personas.keys():
    schedule.every().day.at("01:00").do(daily_schedule_update, user_id)

# 초기 일정 생성 및 등록
for user_id in user_personas.keys():
    daily_schedule_update(user_id)

# 메인 루프 실행
while True:
    schedule.run_pending()
    time.sleep(1)
