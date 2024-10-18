import os
import time
from dotenv import load_dotenv
import openai
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.llms import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 환경 변수 로드
load_dotenv()

# LLM과 Embedding 모델 설정
llm = ChatOpenAI(temperature=0.9, model_name='gpt-3.5-turbo')
embedding_model = OpenAIEmbeddings()

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

    def calculate_importance(self, content):
        # ChatGPT를 사용하여 중요도를 1에서 10 사이로 평가
        prompt = f"다음 내용의 중요도를 1에서 10 사이로 평가해줘: '{content}'"
        response = openai.Completion.create(
            engine='text-davinci-003',
            prompt=prompt,
            max_tokens=10,
            temperature=0
        )
        importance_score = int(response.choices[0].text.strip())
        return importance_score

    def retrieve_memories(self, query, top_k=3):
        # 현재 시간
        current_time = time.time()

        # 쿼리 임베딩
        query_embedding = embedding_model.embed_query(query)

        # 벡터 스토어에서 유사한 기억 검색
        results = vector_store.similarity_search_with_score(query, k=top_k)

        retrieved_memories = []
        for doc, score in results:
            content = doc.page_content
            metadata = doc.metadata

            # Recency 계산
            time_diff = (current_time - metadata['timestamp']) / 3600  # 시간 차이 (시간 단위)
            recency_score = 0.99 ** time_diff

            # Importance 가져오기
            importance_score = metadata['importance']

            # Relevance는 이미 cosine 유사도로 계산된 score를 사용
            relevance_score = score

            # 최종 스코어 계산
            total_score = recency_score + importance_score + relevance_score

            retrieved_memories.append((content, total_score))

        # 총 스코어 기준으로 정렬
        retrieved_memories.sort(key=lambda x: x[1], reverse=True)

        return [content for content, score in retrieved_memories]

# 메모리 인스턴스 생성
memory = Memory()

# 페르소나 정의
persona_joy = "기쁨이"
persona_sad = "슬픔이"

# 초기 대화 설정
def initial_conversation():
    memory.add_memory(persona_joy, "오늘 날씨가 정말 좋아! 기분이 너무 좋아진다.")
    memory.add_memory(persona_sad, "하지만 난 비가 와서 그런지 마음이 우울해.")

initial_conversation()

# 대화 시뮬레이션 함수
def simulate_conversation(persona_a, persona_b, num_turns=3):
    for _ in range(num_turns):
        # 페르소나 A가 말할 내용 생성
        retrieved_memories = memory.retrieve_memories(f"{persona_a}와 관련된 최근 대화 내용")
        context = "\n".join(retrieved_memories)

        prompt = f"""당신은 {persona_a}입니다.
과거 대화 내용:
{context}

{persona_b}에게 한 마디 해주세요."""

        response = llm([SystemMessage(content=prompt)]).content.strip()

        print(f"{persona_a}: {response}")

        # 메모리에 추가
        memory.add_memory(persona_a, response)

        # 페르소나 B가 말할 내용 생성
        retrieved_memories = memory.retrieve_memories(f"{persona_b}와 관련된 최근 대화 내용")
        context = "\n".join(retrieved_memories)

        prompt = f"""당신은 {persona_b}입니다.
과거 대화 내용:
{context}

{persona_a}의 말에 답해주세요."""

        response = llm([SystemMessage(content=prompt)]).content.strip()

        print(f"{persona_b}: {response}")

        # 메모리에 추가
        memory.add_memory(persona_b, response)

# 대화 시뮬레이션 실행
simulate_conversation(persona_joy, persona_sad, num_turns=3)

# ChromaDB에 저장된 내용 영구 저장
vector_store.persist()
