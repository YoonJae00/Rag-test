import os
import time
from dotenv import load_dotenv
import openai
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.llms import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
# 환경 변수 로드
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
from langchain_teddynote import logging
from langchain_core.prompts import ChatPromptTemplate

# 프로젝트 이름을 입력합니다.
logging.langsmith("personas")

# LLM과 Embedding 모델 설정
llm = ChatOpenAI(temperature=0.9, model_name='gpt-4o-mini')
embedding_model = OpenAIEmbeddings()

llm2 = ChatOpenAI(temperature=0.1, model_name='gpt-4o-mini')
# ChromaDB 설정
persist_directory = './chroma_db'
vector_store = Chroma(collection_name='memory_store', embedding_function=embedding_model, persist_directory=persist_directory)

# 기억 저장을 위한 데이터 구조
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

        # 벡터 스토어에 저장
        vector_store.add_texts([content], metadatas=[{'persona': persona, 'timestamp': timestamp, 'importance': importance_score}])

        print(f"중요도: {importance_score}")

    def calculate_importance(self, content):
        prompt = f"""
        다음 대화 내용의 중요성을 JSON 형식으로 평가해 주세요:
        
        "{content}"
        
        중요도를 평가할 때는 다음과 같은 기준을 사용하세요:
        
        1. 이 대화가 에이전트의 미래 행동이나 의사결정에 얼마나 영향을 미칠 수 있는가?
        2. 이 대화가 에이전트의 관계, 감정 상태, 또는 목표에 중요한 변화를 일으킬 수 있는가?
        3. 이 대화가 에이전트의 장기 기억에 보관될 만한 중요한 사건인가?
        
        위 기준에 따라 1에서 10까지의 숫자로 중요도를 평가해 주세요.
        
        JSON 형식으로 답변해 주세요. 다음과 같은 형식으로:
        {{
            "importance": 숫자
        }}
        """
        parser = JsonOutputParser(pydantic_object=Topic)
        
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are an AI that evaluates the importance of text."),
                ("user", "{input}"),
            ]
        )
        prompt = prompt.partial(format_instructions=parser.get_format_instructions())

        chain = prompt | llm2 | parser

        result = chain.invoke({"input": prompt})
        
        # 결과가 딕셔너리인 경우를 처리합니다
        if isinstance(result, dict) and 'importance' in result:
            return result['importance']
        else:
            # 예상치 못한 형식의 경우 기본값을 반환하거나 오류를 처리합니다
            print(f"예상치 못한 결과 형식: {result}")
            return 5  # 기본값으로 중간 중요도를 반환

    def retrieve_memories(self, query, top_k=3):
        current_time = time.time()
        query_embedding = embedding_model.embed_query(query)
        results = vector_store.similarity_search_with_score(query, k=top_k)

        retrieved_memories = []
        for doc, score in results:
            content = doc.page_content
            metadata = doc.metadata

            time_diff = (current_time - metadata['timestamp']) / 3600
            recency_score = 0.69 ** time_diff
            importance_score = metadata['importance']
            relevance_score = score
            total_score = recency_score + importance_score + relevance_score

            retrieved_memories.append((content, total_score))

        retrieved_memories.sort(key=lambda x: x[1], reverse=True)
        return [content for content, score in retrieved_memories]

    def reflection(self):
        # 회고 메커니즘: 중요도가 높은 상위 기억을 고차원적인 기억으로 판단
        print("\n=== 회고 메커니즘 실행 중 ===")
        high_level_memories = []

        # 메모리에서 중요도가 8 이상인 기억을 고차원 기억으로 선택
        for content, importance in self.importance_scores.items():
            if importance >= 8:
                high_level_memories.append(content)
                print(f"선택된 고차원 기억: {content} (중요도: {importance})")

        # 고차원 기억을 다시 메모리 스트림에 주입하여 강화
        for memory in high_level_memories:
            self.add_memory('Reflection', memory)

        print(f"회고 완료: {len(high_level_memories)}개의 고차원 기억이 메모리 스트림에 추가되었습니다.\n")

# 원하는 데이터 구조를 정의합니다.
class Topic(BaseModel):
    importance: int = Field(description="중요도를 1에서 10까지의 숫자로 평가")


memory = Memory()

# 에이전트 관계 및 성격 설정
agents_relationships = {
    "기쁨이": {
        "성격": "낙천적이고 긍정적",
        "관계": {
            "슬픔이": "친구, 슬픔이의 감정을 잘 이해하려고 하지만 종종 겉돌음."
        }
    },
    "슬픔이": {
        "성격": "감정적으로 불안정하고 우울함",
        "관계": {
            "기쁨이": "친구, 기쁨이의 긍정적인 태도에 종종 부담을 느낌."
        }
    }
}

# 페르소나에 따른 초기 상황 설정
agents_scenarios = {
    "기쁨이": "기쁨이는 오늘 날씨가 좋아서 기분이 매우 좋다. 슬픔이를 만나서 같이 산책하자고 권유하고 싶어 한다.",
    "슬픔이": "슬픔이는 기분이 좋지 않다. 내일 비가 온다고 들어서 더 우울해졌다."
}


# 초기 대화: 기쁨이와 슬픔이의 주제 설정
def initial_conversation():
    memory.add_memory(agents_scenarios["기쁨이"], "오늘 날씨가 좋아서 산책 가고 싶어.")
    memory.add_memory(agents_scenarios["슬픔이"], "하지만 내일 비가 올 거라서 마음이 우울해.")

initial_conversation()

# 대화 시뮬레이션에서 상황을 기반으로 대화 흐름을 생성
def simulate_conversation_with_context(persona_a, persona_b, num_turns=5, reflection_interval=3):
    for turn in range(num_turns):
        if turn > 0 and turn % reflection_interval == 0:
            memory.reflection()  # 주기적으로 회고 실행

        # 페르소나 A가 말할 내용 생성 (상황 및 관계 반영)
        retrieved_memories = memory.retrieve_memories(f"{persona_a}와 관련된 최근 대화 내용")
        context = "\n".join(retrieved_memories)

        # 페르소나 A의 성격 및 상황 반영
        situation_a = agents_scenarios.get(persona_a, "")
        relationship_to_b = agents_relationships[persona_a]['관계'][persona_b]

        prompt = f"""당신은 {persona_a}입니다.
과거 대화 내용:
{context}

상황: {situation_a}
{persona_b}와의 관계: {relationship_to_b}

{persona_b}에게 한 마디 해주세요. 친구처럼 비속어를 사용하며 대화합니다."""

        response = llm([SystemMessage(content=prompt)]).content.strip()
        print(f"A대화 ({persona_a}): {response}")
        memory.add_memory(persona_a, response)

        # 페르소나 B가 답변 생성
        retrieved_memories = memory.retrieve_memories(f"{persona_b}와 관련된 최근 대화 내용")
        context = "\n".join(retrieved_memories)

        situation_b = agents_scenarios.get(persona_b, "")
        relationship_to_a = agents_relationships[persona_b]['관계'][persona_a]

        prompt = f"""당신은 {persona_b}입니다.
과거 대화 내용:
{context}

상황: {situation_b}
{persona_a}와의 관계: {relationship_to_a}

{persona_a}의 말에 답해주세요."""

        response = llm([SystemMessage(content=prompt)]).content.strip()

        print(f"B대화 ({persona_b}): {response}")

        # 메모리에 추가
        memory.add_memory(persona_b, response)

# 대화 시뮬레이션 실행
simulate_conversation_with_context("기쁨이", "슬픔이", num_turns=20)

# ChromaDB에 저장된 내용 영구 저장
vector_store.persist()
