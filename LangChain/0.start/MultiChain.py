from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt1 = ChatPromptTemplate.from_template("translates {korean_word} to English.")
prompt2 = ChatPromptTemplate.from_template(
    "explain {english_word} using oxford dictionary to me in Korean."
)

llm = ChatOpenAI(model="gpt-4o-mini")

chain1 = prompt1 | llm | StrOutputParser()

chain2 = (
    {"english_word": chain1}
    | prompt2
    | llm
    | StrOutputParser()
)

result = chain2.invoke({"korean_word":"미래"})
print(result)