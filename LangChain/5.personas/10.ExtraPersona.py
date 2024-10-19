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

# 새로운 페르소나 설정 (유진, 애신, 서진 추가)
agents_relationships = {
    "유진": {
        "성격": "기술과 혁신에 관심이 많은 30대 남성, 논리적이고 분석적인 성향",
        "나이": 32,
        "직업": "IT 기업 엔지니어",
        "성별": "남성",
        "관계": {
            "애신": "대학 동기, 서로 다른 관점을 가지고 있지만 존중하는 사이",
            "서진": "비서 같은 역할, 유진의 정보 탐색과 조언을 돕는다."
        }
    },
    "애신": {
        "성격": "사회 문제와 윤리에 관심이 많은 30대 여성, 감성적이고 직관적인 성향",
        "나이": 31,
        "직업": "NGO 활동가",
        "성별": "여성",
        "관계": {
            "유진": "대학 동기, 유진의 기술 중심적 사고방식에 때때로 의문을 제기함",
            "서진": "정보를 제공받고 정리해주는 역할을 해줌."
        }
    },
    "서진": {
        "성격": "논리적이고 효율적인 성향, 기술에 능통하며 필요한 정보를 신속하게 제공하는 비서 역할",
        "나이": 28,
        "직업": "가상 비서",
        "성별": "남성",
        "관계": {
            "유진": "정보 제공과 검색을 담당",
            "애신": "필요할 때 정보를 전달"
        }
    }
}

# 초기 시나리오 설정
agents_scenarios = {
    "유진": "유진은 최근 AI 기술의 발전에 대해 흥분해 있으며, 특히 생성형 AI의 가능성에 대해 이야기하고 싶어 한다.",
    "애신": "애신은 AI 기술의 윤리적 측면과 사회적 영향에 대해 고민하고 있으며, 이에 대해 유진과 깊이 있는 대화를 나누고 싶어 한다."
}

# 초기 대화 설정
def initial_conversation():
    memory.add_memory("유진", "유진은 최근 ChatGPT와 같은 생성형 AI의 발전에 대해 애신과 이야기를 나누고 싶어한다.")
    memory.add_memory("애신", "애신은 AI 기술의 발전이 일자리와 프라이버시에 미칠 영향에 대해 우려하고 있다.")

initial_conversation()

# 서진의 역할을 추가하여 검색 처리
def search_information(query):
    """서진이 정보를 검색하여 결과를 반환하는 역할"""
    print(f"서진이 검색 요청을 받았습니다: {query}")
    search_results = web_search.invoke(query)
    
    if search_results:
        real_time_data = search_results[0].get('content') or search_results[0].get('result')
        if real_time_data:
            print(f"서진의 검색 결과: {real_time_data}")
            return real_time_data
    return "검색 결과가 없습니다."

# 대화 시뮬레이션: 서진을 통한 검색 결과 반영
def simulate_conversation_with_real_data(persona_a, persona_b, num_turns=5, reflection_interval=3):
    conversation_log = []  # 대화 내용을 저장할 리스트

    for turn in range(num_turns):
        if turn > 0 and turn % reflection_interval == 0:
            memory.reflection()

        # 페르소나 A의 대화
        retrieved_memories = memory.retrieve_memories(f"{persona_a}와 관련된 최근 대화 내용")
        context = "\n".join(retrieved_memories)
        relationship_to_b = agents_relationships[persona_a]['관계'][persona_b]

        prompt = f"""당신은 {persona_a}입니다.
과거 대화 내용:
{context}

상황: {agents_scenarios[persona_a][min(turn, len(agents_scenarios[persona_a])-1)]}
{persona_b}와의 관계: {relationship_to_b}

{persona_b}에게 한 마디 해주세요. 필요시 '검색:서진에게 부탁해'라고 말한 후 검색할 내용을 적어주세요."""

        response = llm([SystemMessage(content=prompt)]).content.strip()
        print(f"A대화 ({persona_a}): {response}")
        conversation_log.append(f"{persona_a}: {response}")

        if '검색:' in response:
            query = response.split('검색:')[1].strip()
            print(f"서진이 검색 요청을 처리합니다: {query}")
            search_result = search_information(query)
            
            prompt = f"""서진이 검색 결과를 전달합니다: "{search_result}".
            이 정보를 바탕으로 {persona_b}와 대화를 이어가세요."""

            response = llm([SystemMessage(content=prompt)]).content.strip()
            print(f"A대화 ({persona_a}): {response}")
            conversation_log.append(f"{persona_a}: {response}")

        memory.add_memory(persona_a, response)

        # 페르소나 B의 대화 (유사한 방식으로 구현)
        retrieved_memories = memory.retrieve_memories(f"{persona_b}와 관련된 최근 대화 내용")
        context = "\n".join(retrieved_memories)
        relationship_to_a = agents_relationships[persona_b]['관계'][persona_a]

        prompt = f"""당신은 {persona_b}입니다.
과거 대화 내용:
{context}

상황: {agents_scenarios[persona_b][min(turn, len(agents_scenarios[persona_b])-1)]}
{persona_a}와의 관계: {relationship_to_a}

{persona_a}의 말에 답해주세요. 필요시 '검색:서진에게 부탁해'라고 말한 후 검색할 내용을 적어주세요."""

        response = llm([SystemMessage(content=prompt)]).content.strip()
        print(f"B대화 ({persona_b}): {response}")
        conversation_log.append(f"{persona_b}: {response}")

        if '검색:' in response:
            query = response.split('검색:')[1].strip()
            print(f"서진이 검색 요청을 처리합니다: {query}")
            search_result = search_information(query)

            prompt = f"""서진이 검색 결과를 전달합니다: "{search_result}".
            이 정보를 바탕으로 다시 이야기하세요."""

            response = llm([SystemMessage(content=prompt)]).content.strip()
            print(f"B대화 ({persona_b}): {response}")
            conversation_log.append(f"{persona_b}: {response}")

        memory.add_memory(persona_b, response)

    # 대화 내용을 파일에 저장
    save_conversation_to_file(conversation_log, f"{persona_a}_{persona_b}_conversation.txt")

# 서진의 검색 요청을 처리하는 함수 추가
def search_information(query):
    """서진이 검색 요청을 처리하여 결과 반환"""
    print(f"서진이 검색 요청을 받았습니다: {query}")
    search_results = web_search.invoke(query)
    if search_results:
        real_time_data = search_results[0].get('content') or search_results[0].get('result')
        if real_time_data:
            return real_time_data
    return "검색 결과를 찾을 수 없습니다."

# 대화 시뮬레이션 실행
simulate_conversation_with_real_data("유진", "애신", num_turns=10)

def save_conversation_to_file(conversation_log, filename):
    # 현재 스크립트의 디렉토리 경로를 가져옵니다.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 프로젝트 루트 디렉토리 경로를 구성합니다 (현재 디렉토리의 상위 디렉토리).
    project_root = os.path.dirname(current_dir)
    # 'textlog' 디렉토리 경로를 생성합니다.
    textlog_dir = os.path.join(project_root, "textlog")
    
    # 'textlog' 디렉토리가 없으면 생성합니다.
    if not os.path.exists(textlog_dir):
        os.makedirs(textlog_dir)
    
    # 파일 경로를 구성합니다.
    file_path = os.path.join(textlog_dir, filename)
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(conversation_log))
    
    print(f"대화 내용이 {file_path}에 저장되었습니다.")
    
# ChromaDB에 저장된 내용 영구 저장
vector_store.persist()


# Generative agents에서 메모리 스트림은 다양한 종류의 정보를 포함하여 더 복잡하게 작동합니다. 메모리 스트림에는 대화 기록뿐만 아니라 에이전트의 관찰, 행동, 감정적 반응, 시간 흐름 같은 여러 요소들이 저장됩니다. 이러한 정보들은 에이전트가 다음에 행동하거나 응답할 때 그들이 더 자연스럽게 대화에 참여하거나 환경에 반응하게 만드는 중요한 요소가 됩니다.

# 현재 사용 중인 코드 흐름에서는 단일 주제에 대한 대화만 반복하며 중요한 정보를 요약한 후 다시 그 주제를 이어가지만, Generative agents는 다음과 같은 점에서 차이가 있습니다:

# 1. 메모리 스트림 구성
# Generative agents의 메모리 스트림에는 다음과 같은 정보들이 축적됩니다:

# 일반적 경험: 에이전트가 경험한 사건, 대화, 관찰 등이 저장됩니다. 이 경험은 지속적으로 쌓여 에이전트의 성격과 행동을 결정하는데 영향을 미칩니다.
# 시간 흐름: 에이전트가 활동하면서 시간이 흐르며 그에 따른 경험과 기억이 업데이트됩니다. 이는 과거 경험이 얼마나 '최근'이었는지에 따라 달라집니다.
# 관계: 다른 에이전트나 사용자와의 관계에 대한 기억도 저장됩니다. 이를 통해 이전에 맺은 관계나 친밀도 등이 다음 대화나 행동에 영향을 미치게 됩니다.
# 목표: 에이전트는 주어진 목표를 가지고 활동하며, 메모리 스트림에는 그 목표를 달성하기 위한 과정도 저장될 수 있습니다.
# 중요한 기억들: 중요도가 높은 기억이나 사건들은 장기 기억으로 저장되며, 후속 대화나 행동에 영향을 줍니다.
# 2. 멀티태스킹 및 주제 전환
# Generative agents는 하나의 주제에 대한 대화만 반복하는 것이 아니라, 멀티태스킹처럼 여러 주제를 넘나들 수 있습니다. 예를 들어, 대화 중간에 다른 중요한 사건이 발생하거나, 에이전트가 새로운 정보를 관찰하게 되면 주제가 자연스럽게 전환될 수 있습니다.

# 3. 기억 강화 및 반영
# Generative agents는 특정 사건이나 대화의 중요도를 계산하여 중요한 기억을 **반영(Reflection)**하고 이를 장기 기억으로 저장합니다. 이로 인해 에이전트는 대화를 할수록 점점 더 복잡한 성격을 갖추고 행동하게 됩니다.

# 4. 다양한 행동 및 감정 반응
# Generative agents는 단순히 대화만 하는 것이 아니라 행동이나 감정적 반응을 보여줍니다. 예를 들어, 다른 에이전트와의 대화에서 기쁨, 슬픔, 분노 같은 감정적 반응이 나타날 수 있으며, 그 감정은 다음 대화에 영향을 미칩니다. 감정적 경험도 메모리 스트림에 저장되며, 에이전트가 특정 사건에 어떻게 반응할지를 결정하게 됩니다.

# 현재 코드 개선 방향
# 현재 코드는 단일 주제에서만 대화를 반복하고 중요한 내용만 요약하는 방식입니다. 이를 좀 더 발전시키려면 다음과 같은 개선이 필요할 수 있습니다:

# 다양한 주제 도입: 대화 주제가 하나로 제한되지 않고, 자연스럽게 여러 주제로 확장될 수 있도록 코드를 수정합니다.

# 각 에이전트가 다양한 관심사를 갖고 있으며, 대화 중 다른 관심사로 주제가 이동하는 경우를 추가할 수 있습니다.
# 시간 흐름 반영: 메모리 스트림에 시간 정보를 반영하여, 에이전트가 '얼마 전에 있었던 일'과 '최근 일어난 사건'을 구분할 수 있도록 메모리를 관리합니다.

# 관찰과 행동: 에이전트가 단순히 대화를 이어가는 것 외에도, 관찰을 통해 정보를 얻고, 이를 바탕으로 행동을 선택하는 구조를 추가합니다.

# 목표 설정: 각 에이전트가 달성하려는 목표를 설정하고, 그 목표를 향해 행동하게 만들며, 목표 달성 여부에 따라 대화가 바뀔 수 있도록 합니다.