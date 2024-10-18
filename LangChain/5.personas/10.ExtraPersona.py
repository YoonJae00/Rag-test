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

# 새로운 페르소나 설정 (유진, 애신 추가)
agents_relationships = {
    "유진": {
        "성격": "냉철하고 분석적인 성향, 논리적 사고를 중시함",
        "나이": 32,
        "직업": "기술 컨설턴트",
        "성별": "남성",
        "관계": {
            "애신": "동료, 가끔 감정적으로 대립하지만 서로 존중하는 관계"
        }
    },
    "애신": {
        "성격": "감정적이고 직관적인 성향, 타인의 감정을 잘 이해함",
        "나이": 28,
        "직업": "심리 상담사",
        "성별": "여성",
        "관계": {
            "유진": "동료, 유진의 분석적인 성향을 존중하면서도 때때로 다른 관점을 제시함"
        }
    }
}

agents_scenarios = {
    "유진": [
        "유진은 최근 북한의 도발과 관련된 뉴스에 관심이 많다.",
    ],
    "애신": [
        "애신은 북한의 도발이 사람들의 심리에 미치는 영향을 우려한다.",
    ]
}

# 초기 대화 설정
def initial_conversation():
    memory.add_memory("유진", "유진은 최근 북한의 도발과 관련된 뉴스를 읽고 애신과 대화를 나누고 싶어한다.")
    memory.add_memory("애신", "애신은 북한의 도발이 사람들의 심리에 미치는 영향을 우려하고 있다.")

initial_conversation()

# 대화 시뮬레이션: 검색 결과를 반영한 대화 흐름
def simulate_conversation_with_real_data(persona_a, persona_b, num_turns=5, reflection_interval=3):
    for turn in range(num_turns):
        if turn > 0 and turn % reflection_interval == 0:
            memory.reflection()

        retrieved_memories = memory.retrieve_memories(f"{persona_a}와 관련된 최근 대화 내용")
        context = "\n".join(retrieved_memories)
        relationship_to_b = agents_relationships[persona_a]['관계'][persona_b]

        prompt = f"""당신은 {persona_a}입니다.
과거 대화 내용:
{context}

상황: {agents_scenarios[persona_a][min(turn, len(agents_scenarios[persona_a])-1)]}
{persona_b}와의 관계: {relationship_to_b}

{persona_b}에게 한 마디 해주세요. 필요시 '검색:'이라고 말한 후 검색 내용을 적어주세요."""

        response = llm([SystemMessage(content=prompt)]).content.strip()
        print(f"A대화 ({persona_a}): {response}")

        if '검색:' in response:
            query = response.split('검색:')[1].strip()
            print(f"검색 요청: {query}")

            search_results = web_search.invoke(query)
            if search_results:
                real_time_data = search_results[0].get('content') or search_results[0].get('result')
                if real_time_data:
                    print(f"검색 결과: {real_time_data}")
                    
                    prompt = f"""검색 결과에 따르면, "{real_time_data}"라는 정보가 있습니다.
                    이 정보를 바탕으로 {persona_b}와 대화를 이어가세요."""

                    response = llm([SystemMessage(content=prompt)]).content.strip()
                    print(f"A대화 ({persona_a}): {response}")

        memory.add_memory(persona_a, response)

        retrieved_memories = memory.retrieve_memories(f"{persona_b}와 관련된 최근 대화 내용")
        context = "\n".join(retrieved_memories)
        relationship_to_a = agents_relationships[persona_b]['관계'][persona_a]

        prompt = f"""당신은 {persona_b}입니다.
과거 대화 내용:
{context}

상황: {agents_scenarios[persona_b][min(turn, len(agents_scenarios[persona_b])-1)]}
{persona_a}와의 관계: {relationship_to_a}

{persona_a}의 말에 답해주세요. 필요시 '검색:'을 사용해 검색 후 대화를 이어가세요."""

        response = llm([SystemMessage(content=prompt)]).content.strip()
        print(f"B대화 ({persona_b}): {response}")

        if '검색:' in response:
            query = response.split('검색:')[1].strip()
            print(f"검색 요청: {query}")

            search_results = web_search.invoke(query)
            if search_results:
                real_time_data = search_results[0].get('content') or search_results[0].get('result')
                if real_time_data:
                    # print(f"검색 결과: {real_time_data}")

                    prompt = f"""검색 결과에 따르면, "{real_time_data}"라는 정보가 있습니다.
                    이 정보를 바탕으로 다시 이야기하세요."""

                    response = llm([SystemMessage(content=prompt)]).content.strip()
                    print(f"B대화 ({persona_b}): {response}")

        memory.add_memory(persona_b, response)

# 대화 시뮬레이션 실행
simulate_conversation_with_real_data("유진", "애신", num_turns=50)

# ChromaDB에 저장된 내용 영구 저장
vector_store.persist()
