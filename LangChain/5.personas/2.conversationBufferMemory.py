from dotenv import load_dotenv
load_dotenv()

from langchain.memory import ConversationBufferMemory

from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain

# LLM 모델을 생성합니다.
llm = ChatOpenAI(temperature=0)

# ConversationChain을 생성합니다.
conversation = ConversationChain(
    # ConversationBufferMemory를 사용합니다.
    llm=llm,
    memory=ConversationBufferMemory(),
)
# 대화를 시작합니다.
response = conversation.predict(
    input="안녕하세요, 비대면으로 은행 계좌를 개설하고 싶습니다. 어떻게 시작해야 하나요?"
)
print(response)
# 이전 대화내용을 불렛포인트로 정리해 달라는 요청을 보냅니다.
response = conversation.predict(
    input="이전 답변을 불렛포인트 형식으로 정리하여 알려주세요."
)
print(response)
