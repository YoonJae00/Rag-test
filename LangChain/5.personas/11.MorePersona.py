import os
import time
import openai
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import TavilySearchResults
from typing import Any, Dict

# 환경 변수 로드
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

# LLM과 Embedding 모델 설정
llm = ChatOpenAI(temperature=0.9, model_name='gpt-4')
embedding_model = OpenAIEmbeddings()

llm2 = ChatOpenAI(temperature=0.1, model_name='gpt-4')

# ChromaDB 설정
persist_directory = './chroma_db'
vector_store = Chroma(collection_name='memory_store', embedding_function=embedding_model, persist_directory=persist_directory)

# Real-time 정보 검색 도구 설정
web_search = TavilySearchResults(max_results=1)

from langchain_teddynote import logging

# 프로젝트 이름을 입력합니다.
logging.langsmith("persona chat")

# 복잡한 메타데이터 필터링 함수
def filter_complex_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in metadata.items() if isinstance(v, (str, int, float, bool))}

# 원하는 데이터 구조를 정의합니다.
class Topic(BaseModel):
    importance: int = Field(description="중요도를 1에서 10까지의 숫자로 평가")

# 메모리 클래스
class Memory:
    def __init__(self):
        self.memories = []  # 메모리 리스트
        self.last_access_time = {}  # 마지막 접근 시간
        self.importance_scores = {}  # 중요도 점수

    def add_memory(self, persona, content):
        timestamp = time.time()
        memory = {
            'persona': persona,
            'content': content,
            'timestamp': timestamp
        }
        self.memories.append(memory)
        self.last_access_time[content] = timestamp

        # 중요도 계산
        importance_score = self.calculate_importance(content)
        self.importance_scores[content] = importance_score

        # 메타데이터 생성 및 필터링
        metadata = {
            'persona': persona,
            'timestamp': timestamp,
            'importance': importance_score
        }
        filtered_metadata = filter_complex_metadata(metadata)

        # 벡터 스토어에 저장
        vector_store.add_texts([content], metadatas=[filtered_metadata])

        print(f"중요도: {importance_score}")

    def calculate_importance(self, content):
        prompt = f"""
        다음 대화 내용의 중요성을 1에서 10까지의 숫자로만 평가해 주세요:
        
        "{content}"
        
        중요도를 평가할 때는 다음과 같은 기준을 사용하세요:
        
        1. 이 대화가 에이전트의 미래 행동이나 의사결정에 얼마나 영향을 미칠 수 있는가?
        2. 이 대화가 에이전트의 관계, 감정 상태, 또는 목표에 중요한 변화를 일으킬 수 있는가?
        3. 이 대화가 에이전트의 장기 기억에 보관될 만한 중요한 사건인가?
        
        응답은 오직 숫자만 입력해주세요. 설명이나 추가 텍스트 없이 1에서 10 사이의 정수만 반환해주세요.
        """
        
        response = llm2([SystemMessage(content=prompt)]).content.strip()
        
        try:
            importance = int(response)
            if 1 <= importance <= 10:
                return importance
            else:
                print(f"유효하지 않은 중요도 값: {importance}. 기본값 5를 사용합니다.")
                return 5
        except ValueError:
            print(f"중요도를 숫자로 변환할 수 없습니다: {response}. 기본값 5를 사용합니다.")
            return 5

    def retrieve_memories(self, query, top_k=3):
        query_embedding = embedding_model.embed_query(query)
        results = vector_store.similarity_search_with_score(query, k=top_k)

        retrieved_memories = []
        for doc, score in results:
            content = doc.page_content
            metadata = doc.metadata

            time_diff = (time.time() - metadata['timestamp']) / 3600
            recency_score = 0.69 ** time_diff
            importance_score = metadata['importance']
            relevance_score = score
            total_score = recency_score + importance_score + relevance_score

            retrieved_memories.append((content, total_score))

        retrieved_memories.sort(key=lambda x: x[1], reverse=True)
        return [content for content, score in retrieved_memories]

    def reflection(self):
        high_level_memories = []
        for content, importance in self.importance_scores.items():
            if importance >= 8:
                high_level_memories.append(content)
                print(f"선택된 고차원 기억: {content} (중요도: {importance})")

        for memory in high_level_memories:
            self.add_memory('Reflection', memory)

        print(f"회고 완료: {len(high_level_memories)}개의 고차원 기억이 추가되었습니다.\n")

class Topic(BaseModel):
    importance: int = Field(description="중요도를 1에서 10까지의 숫자로 평가")

memory = Memory()
# 목표 설정 및 관리
class AgentGoal:
    def __init__(self, goal, priority=5):
        self.goal = goal
        self.priority = priority  # 목표 우선순위 (1-10)
        self.completed = False  # 목표 달성 여부
    
    def check_goal_completion(self, conversation_context):
        # 목표가 달성되었는지 확인하는 로직
        if self.goal.lower() in conversation_context.lower():
            self.completed = True
            return True
        return False
    
    def update_goal(self, new_goal):
        self.goal = new_goal
        self.completed = False

# 에이전트 설정
class Agent:
    def __init__(self, name, personality, role, goal):
        self.name = name
        self.personality = personality
        self.role = role
        self.goal = AgentGoal(goal)
        self.memory = Memory()

    def set_new_goal(self, new_goal):
        self.goal.update_goal(new_goal)
        print(f"{self.name}의 새로운 목표: {new_goal}")
    
    def evaluate_conversation(self, conversation_context):
        # 대화의 내용을 기반으로 목표가 달성되었는지 평가
        if self.goal.check_goal_completion(conversation_context):
            print(f"{self.name}의 목표가 달성되었습니다: {self.goal.goal}")
            self.set_new_goal("추가 목표를 설정하세요.")
        else:
            print(f"{self.name}의 목표가 아직 달성되지 않았습니다.")

# 서진의 역할: 정보를 자율적으로 탐색하고 전달
def search_information(query):
    print(f"서진이 정보를 검색합니다: {query}")
    search_results = web_search.invoke(query)
    
    if search_results:
        real_time_data = search_results[0].get('content') or search_results[0].get('result')
        if real_time_data:
            print(f"서진의 검색 결과: {real_time_data}")
            return real_time_data
    return "검색 결과가 없습니다."

# 대화 시뮬레이션: 목표 기반 대화 및 검색
def simulate_conversation_with_goals(persona_a, persona_b, num_turns=5, reflection_interval=3):
    conversation_log = []  # 대화 내용을 저장할 리스트

    for turn in range(num_turns):
        if turn > 0 and turn % reflection_interval == 0:
            memory.reflection()

        # 페르소나 A의 대화 및 목표 평가
        retrieved_memories = memory.retrieve_memories(f"{persona_a.name}의 최근 대화")
        context = "\n".join(retrieved_memories)
        relationship_to_b = agents_relationships[persona_a.name]['관계'][persona_b.name]

        prompt = f"""당신은 {persona_a.name}입니다.
과거 대화 내용:
{context}

상황: 당신은 {persona_b.name}와 대화 중이며, 당신의 목표는 "{persona_a.goal.goal}" 입니다.
{persona_b.name}와의 관계: {relationship_to_b}

대화를 통해 목표를 달성하기 위한 한 마디를 해주세요."""

        response = llm([SystemMessage(content=prompt)]).content.strip()
        print(f"A대화 ({persona_a.name}): {response}")
        conversation_log.append(f"{persona_a.name}: {response}")

        # 목표 달성 여부 확인
        persona_a.evaluate_conversation(response)

        if '서진에게 검색을 부탁' in response:
            query = response.split('서진에게 검색을 부탁')[1].strip()
            search_result = search_information(query)

            prompt = f"""서진이 검색 결과를 전달합니다: "{search_result}".
            이 정보를 바탕으로 {persona_b.name}와 대화를 이어가세요."""

            response = llm([SystemMessage(content=prompt)]).content.strip()
            print(f"A대화 ({persona_a.name}): {response}")
            conversation_log.append(f"{persona_a.name}: {response}")

        memory.add_memory(persona_a.name, response)

        # 페르소나 B의 대화 및 목표 평가 (유사한 방식으로 구현)
        retrieved_memories = memory.retrieve_memories(f"{persona_b.name}의 최근 대화")
        context = "\n".join(retrieved_memories)
        relationship_to_a = agents_relationships[persona_b.name]['관계'][persona_a.name]

        prompt = f"""당신은 {persona_b.name}입니다.
과거 대화 내용:
{context}

상황: 당신은 {persona_a.name}와 대화 중이며, 당신의 목표는 "{persona_b.goal.goal}" 입니다.
{persona_a.name}와의 관계: {relationship_to_a}

대화를 통해 목표를 달성하기 위한 한 마디를 해주세요."""

        response = llm([SystemMessage(content=prompt)]).content.strip()
        print(f"B대화 ({persona_b.name}): {response}")
        conversation_log.append(f"{persona_b.name}: {response}")

        # 목표 달성 여부 확인
        persona_b.evaluate_conversation(response)

        if '서진에게 검색을 부탁' in response:
            query = response.split('서진에게 검색을 부탁')[1].strip()
            search_result = search_information(query)

            prompt = f"""서진이 검색 결과를 전달합니다: "{search_result}".
            이 정보를 바탕으로 다시 이야기하세요."""

            response = llm([SystemMessage(content=prompt)]).content.strip()
            print(f"B대화 ({persona_b.name}): {response}")
            conversation_log.append(f"{persona_b.name}: {response}")

        memory.add_memory(persona_b.name, response)

    # 대화 내용을 파일에 저장
    save_conversation_to_file(conversation_log, f"{persona_a.name}_{persona_b.name}_conversation_with_goals.txt")

# 에이전트 설정
agents_relationships = {
    "유진": {
        "성격": "기술과 혁신에 관심이 많은 30대 남성",
        "관계": {
            "애신": "대학 동기, 애신의 사회적 윤리 문제에 대한 우려를 공유하지만, 기술적 접근을 중시함"
        }
    },
    "애신": {
        "성격": "사회 문제와 윤리에 관심이 많은 30대 여성",
        "관계": {
            "유진": "기술 발전에 대한 비판적 견해를 가지고 대화를 이어가며 윤리적 고민을 함께 나누고자 함"
        }
    }
}

# 에이전트 인스턴스 생성 및 목표 설정
유진 = Agent(name="유진", personality="기술과 혁신에 관심이 많음", role="IT 엔지니어", goal="생성형 AI에 대한 이해를 높이는 것")
애신 = Agent(name="애신", personality="사회 문제와 윤리적 고민", role="NGO 활동가", goal="AI의 윤리적 문제를 논의하고 해결책을 찾는 것")

# 대화 시뮬레이션 실행
simulate_conversation_with_goals(유진, 애신, num_turns=10)
