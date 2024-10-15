from dotenv import load_dotenv
load_dotenv()

# 2-튜플 형태의 메시지 목록으로 프롬프트 생성 (type, content)

# from langchain_core.prompts import ChatPromptTemplate

# chat_prompt = ChatPromptTemplate.from_messages([
#     ("system", "이 시스템은 천문학 질문에 답변할 수 있습니다."),
#     ("user", "{user_input}"),
# ])

# messages = chat_prompt.format_messages(user_input="태양계에서 가장 큰 행성은 무엇인가요?")
# print(messages)

# from langchain_core.output_parsers import StrOutputParser
# from langchain_openai import ChatOpenAI

# llm = ChatOpenAI(model="gpt-4o-mini")

# chain = chat_prompt | llm | StrOutputParser()

# result = chain.invoke({"user_input": "태양계에서 가장 큰 행성은 무엇인가요?"})
# print(result)

# 3. MessagePromptTemplate 활용

# MessagePromptTemplate 활용
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import SystemMessagePromptTemplate,  HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

chat_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template("이 시스템은 천문학 질문에 답변할 수 있습니다."),
        HumanMessagePromptTemplate.from_template("{user_input}"),
    ]
)

messages = chat_prompt.format_messages(user_input="태양계에서 가장 큰 행성은 무엇인가요?")

llm = ChatOpenAI(model="gpt-4o-mini")

chain = chat_prompt | llm | StrOutputParser()

stream = chain.stream({"user_input": "태양계에서 가장 큰 행성은 무엇인가요?"})
for chunk in stream:
    print(chunk, end="", flush=True)
print()