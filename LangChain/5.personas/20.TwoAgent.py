from typing import Callable, List


from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_openai import ChatOpenAI


class DialogueAgent:
    def __init__(
        self,
        name: str,
        system_message: SystemMessage,
        model: ChatOpenAI,
    ) -> None:
        # 에이전트의 이름을 설정합니다.
        self.name = name
        # 시스템 메시지를 설정합니다.
        self.system_message = system_message
        # LLM 모델을 설정합니다.
        self.model = model
        # 에이전트 이름을 지정합니다.
        self.prefix = f"{self.name}: "
        # 에이전트를 초기화합니다.
        self.reset()

    def reset(self):
        """
        대화 내역을 초기화합니다.
        """
        self.message_history = ["Here is the conversation so far."]

    def send(self) -> str:
        """
        메시지에 시스템 메시지 + 대화내용과 마지막으로 에이전트의 이름을 추가합니다.
        """
        message = self.model(
            [
                self.system_message,
                HumanMessage(content="\n".join(
                    [self.prefix] + self.message_history)),
            ]
        )
        return message.content

    def receive(self, name: str, message: str) -> None:
        """
        name 이 말한 message 를 메시지 내역에 추가합니다.
        """
        self.message_history.append(f"{name}: {message}")

class DialogueSimulator:
    def __init__(
        self,
        agents: List[DialogueAgent],
        selection_function: Callable[[int, List[DialogueAgent]], int],
    ) -> None:
        # 에이전트 목록을 설정합니다.
        self.agents = agents
        # 시뮬레이션 단계를 초기화합니다.
        self._step = 0
        # 다음 발언자를 선택하는 함수를 설정합니다.
        self.select_next_speaker = selection_function

    def reset(self):
        # 모든 에이전트를 초기화합니다.
        for agent in self.agents:
            agent.reset()

    def inject(self, name: str, message: str):
        """
        name 의 message 로 대화를 시작합니다.
        """
        # 모든 에이전트가 메시지를 받습니다.
        for agent in self.agents:
            agent.receive(name, message)

        # 시뮬레이션 단계를 증가시킵니다.
        self._step += 1

    def step(self) -> tuple[str, str]:
        # 1. 다음 발언자를 선택합니다.
        speaker_idx = self.select_next_speaker(self._step, self.agents)
        speaker = self.agents[speaker_idx]

        # 2. 다음 발언자에게 메시지를 전송합니다.
        message = speaker.send()

        # 3. 모든 에이전트가 메시지를 받습니다.
        for receiver in self.agents:
            receiver.receive(speaker.name, message)

        # 4. 시뮬레이션 단계를 증가시킵니다.
        self._step += 1

        # 발언자의 이름과 메시지를 반환합니다.
        return speaker.name, message

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain import hub


class DialogueAgentWithTools(DialogueAgent):
    def __init__(
        self,
        name: str,
        system_message: SystemMessage,
        model: ChatOpenAI,
        tools,
    ) -> None:
        # 부모 클래스의 생성자를 호출합니다.
        super().__init__(name, system_message, model)
        # 주어진 도구 이름과 인자를 사용하여 도구를 로드합니다.
        self.tools = tools

    def send(self) -> str:
        """
        메시지 기록에 챗 모델을 적용하고 메시지 문자열을 반환합니다.
        """
        prompt = hub.pull("hwchase17/openai-functions-agent")
        agent = create_openai_tools_agent(self.model, self.tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=self.tools, verbose=True)
        # AI 메시지를 생성합니다.
        message = AIMessage(
            content=agent_executor.invoke(
                {
                    "input": "\n".join(
                        [self.system_message.content]
                        + [self.prefix]
                        + self.message_history
                    )
                }
            )["output"]
        )

        # 생성된 메시지의 내용을 반환합니다.
        return message.content

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.document_loaders import TextLoader

# PDF 파일 로드. 파일의 경로 입력
loader1 = TextLoader("data/의대증원반대.txt")
loader2 = TextLoader("data/의대증원찬성.txt")

# 텍스트 분할기를 사용하여 문서를 분할합니다.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# 문서를 로드하고 분할합니다.
docs1 = loader1.load_and_split(text_splitter)
docs2 = loader2.load_and_split(text_splitter)

# VectorStore를 생성합니다.
vector1 = FAISS.from_documents(docs1, OpenAIEmbeddings())
vector2 = FAISS.from_documents(docs2, OpenAIEmbeddings())

# Retriever를 생성합니다.
doctor_retriever = vector1.as_retriever(search_kwargs={"k": 3})
gov_retriever = vector2.as_retriever(search_kwargs={"k": 3})

# langchain 패키지의 tools 모듈에서 retriever 도구를 생성하는 함수를 가져옵니다.
from langchain.tools.retriever import create_retriever_tool

doctor_retriever_tool = create_retriever_tool(
    doctor_retriever,
    name="document_search",
    description="This is a document about the Korean Medical Association's opposition to the expansion of university medical schools. "
    "Refer to this document when you want to present a rebuttal to the proponents of medical school expansion.",
)

gov_retriever_tool = create_retriever_tool(
    gov_retriever,
    name="document_search",
    description="This is a document about the Korean government's support for the expansion of university medical schools. "
    "Refer to this document when you want to provide a rebuttal to the opposition to medical school expansion.",
)

# TavilySearchResults 클래스를 langchain_community.tools.tavily_search 모듈에서 가져옵니다.
from langchain_community.tools.tavily_search import TavilySearchResults

# TavilySearchResults 클래스의 인스턴스를 생성합니다
# k=3은 검색 결과를 3개까지 가져오겠다는 의미입니다
search = TavilySearchResults(k=3)

names = {
    "Doctor Union(의사협회)": [doctor_retriever_tool],  # 의사협회 에이전트 도구 목록
    "Government(대한민국 정부)": [gov_retriever_tool],  # 정부 에이전트 도구 목록
}

# 토론 주제 선정
topic = "2024 현재, 대한민국 대학교 의대 정원 확대 충원은 필요한가?"

# 토론자를 설명하는 문구의 단어 제한
word_limit = 50

names = {
    "Doctor Union(의사협회)": [search],  # 의사협회 에이전트 도구 목록
    "Government(대한민국 정부)": [search],  # 정부 에이전트 도구 목록
}
# 토론 주제 선정
topic = "2024년 현재, 대한민국 대학교 의대 정원 확대 충원은 필요한가?"
word_limit = 50  # 작업 브레인스토밍을 위한 단어 제한

conversation_description = f"""Here is the topic of conversation: {topic}
The participants are: {', '.join(names.keys())}"""

agent_descriptor_system_message = SystemMessage(
    content="You can add detail to the description of the conversation participant."
)


def generate_agent_description(name):
    agent_specifier_prompt = [
        agent_descriptor_system_message,
        HumanMessage(
            content=f"""{conversation_description}
            Please reply with a description of {name}, in {word_limit} words or less in expert tone. 
            Speak directly to {name}.
            Give them a point of view.
            Do not add anything else. Answer in KOREAN."""
        ),
    ]
    # ChatOpenAI를 사용하여 에이전트 설명을 생성합니다.
    agent_description = ChatOpenAI(temperature=0)(
        agent_specifier_prompt).content
    return agent_description


# 각 참가자의 이름에 대한 에이전트 설명을 생성합니다.
agent_descriptions = {name: generate_agent_description(name) for name in names}

# 생성한 에이전트 설명을 출력합니다.

agent_descriptions = {
    "Doctor Union(의사협회)": "의사협회는 의료계의 권익을 보호하고 의사들의 이해관계를 대변하는 기관입니다. 의사들의 업무 환경과 안전을 중시하며, 환자 안전과 질 높은 의료 서비스를 제공하기 위해 노력합니다. "
    "지금도 의사의 수는 충분하다는 입장이며, 의대 증원은 필수 의료나 지방 의료 활성화에 대한 실효성이 떨어집니다. 의대 증원을 감행할 경우, 의료 교육 현장의 인프라가 갑작스러운 증원을 감당하지 못할 것이란 우려를 표합니다.",
    "Government(대한민국 정부)": "대한민국 정부는 국가의 행정을 책임지는 주체로서, 국민의 복지와 발전을 책임져야 합니다. "
    "우리나라는 의사수가 절대 부족한 상황이며, 노인인구가 늘어나면서 의료 수요가 급증하고 있습니다. OECD 국가들도 최근 의사수를 늘렸습니다. 또한, 증원된 의사 인력이 필수의료와 지역 의료로 갈 수있도록 튼튼한 의료사고 안정망 구축 및 보상 체계의 공정성을 높이고자 합니다.",
}

def generate_system_message(name, description, tools):
    return f"""{conversation_description}

Your name is {name}.

Your description is as follows: {description}

Your goal is to persuade your conversation partner of your point of view.

DO look up information with your tool to refute your partner's claims.
DO cite your sources.

DO NOT fabricate fake citations.
DO NOT cite any source that you did not look up.

Do not add anything else.

Stop speaking the moment you finish speaking from your perspective.

Answer in KOREAN.
"""


agent_system_messages = {
    name: generate_system_message(name, description, tools)
    for (name, tools), description in zip(names.items(), agent_descriptions.values())
}

# 에이전트 시스템 메시지를 순회합니다.
for name, system_message in agent_system_messages.items():
    # 에이전트의 이름을 출력합니다.
    print(name)
    # 에이전트의 시스템 메시지를 출력합니다.
    print(system_message)


topic_specifier_prompt = [
    # 주제를 더 구체적으로 만들 수 있습니다.
    SystemMessage(content="You can make a topic more specific."),
    HumanMessage(
        content=f"""{topic}

        You are the moderator. 
        Please make the topic more specific.
        Please reply with the specified quest in 100 words or less.
        Speak directly to the participants: {*names,}.  
        Do not add anything else.
        Answer in Korean."""  # 다른 것은 추가하지 마세요.
    ),
]
# 구체화된 주제를 생성합니다.
specified_topic = ChatOpenAI(temperature=1.0)(topic_specifier_prompt).content

print(f"Original topic:\n{topic}\n")  # 원래 주제를 출력합니다.
print(f"Detailed topic:\n{specified_topic}\n")  # 구체화된 주제를 출력합니다.

# 이는 결과가 컨텍스트 제한을 초과하는 것을 방지하기 위함입니다.
agents = [
    DialogueAgentWithTools(
        name=name,
        system_message=SystemMessage(content=system_message),
        model=ChatOpenAI(model_name="gpt-4-turbo-preview", temperature=0.2),
        tools=tools,
    )
    for (name, tools), system_message in zip(
        names.items(), agent_system_messages.values()
    )
]
def select_next_speaker(step: int, agents: List[DialogueAgent]) -> int:
    # 다음 발언자를 선택합니다.
    # step을 에이전트 수로 나눈 나머지를 인덱스로 사용하여 다음 발언자를 순환적으로 선택합니다.
    idx = (step) % len(agents)
    return idx
max_iters = 6  # 최대 반복 횟수를 6으로 설정합니다.
n = 0  # 반복 횟수를 추적하는 변수를 0으로 초기화합니다.

# DialogueSimulator 객체를 생성하고, agents와 select_next_speaker 함수를 전달합니다.
simulator = DialogueSimulator(agents=agents, selection_function=select_next_speaker)

# 시뮬레이터를 초기 상태로 리셋합니다.
simulator.reset()

# Moderator가 지정된 주제를 제시합니다.
simulator.inject("Moderator", specified_topic)

# Moderator가 제시한 주제를 출력합니다.
print(f"(Moderator): {specified_topic}")
print("\n")

while n < max_iters:  # 최대 반복 횟수까지 반복합니다.
    name, message = (
        simulator.step()
    )  # 시뮬레이터의 다음 단계를 실행하고 발언자와 메시지를 받아옵니다.
    print(f"({name}): {message}")  # 발언자와 메시지를 출력합니다.
    print("\n")
    n += 1  # 반복 횟수를 1 증가시킵니다.
