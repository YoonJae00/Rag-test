from dotenv import load_dotenv
load_dotenv()


from langchain_core.output_parsers import CommaSeparatedListOutputParser

output_parser = CommaSeparatedListOutputParser()
format_instructions = output_parser.get_format_instructions()

print(format_instructions)

from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    template="List five {subject}.\n{format_instructions}",
    input_variables=["subject"],
    partial_variables={"format_instructions": format_instructions},
)

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

chain = prompt | llm | output_parser

result = chain.invoke({"subject": "popular Korean cusine"})

print(result)