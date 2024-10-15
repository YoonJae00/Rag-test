from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# prompt + model + output parser
prompt = ChatPromptTemplate.from_template("You are an expert in astronomy. Answer the question. <Question>: {input}")
llm = ChatOpenAI(model="gpt-4o-mini")
output_parser = StrOutputParser()
prompt2 = ChatPromptTemplate.from_template("너는 거짓말쟁이야 너의 논리의 반대로 대답해 <Question>: {output_parser}")
output_parser2 = StrOutputParser()
# LCEL chaining
chain = prompt | llm | output_parser | prompt2 | llm | output_parser2

# chain 호출
result = chain.invoke({"input": "랭체인은 유익해?"})
print(result)
