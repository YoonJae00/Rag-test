from dotenv import load_dotenv
load_dotenv()


from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 자료구조 정의 (pydantic)
class CusineRecipe(BaseModel):
    name: str = Field(description="name of a cuisine")
    recipe: str = Field(description="recipe to cook the cuisine")

# 출력 파서 정의
output_parser = JsonOutputParser(pydantic_object=CusineRecipe)

format_instructions = output_parser.get_format_instructions()

print("Format Instructions:")
print(format_instructions)

# prompt 구성
prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": format_instructions},
)

print("\nPrompt Template:")
print(prompt)

model = ChatOpenAI(model="gpt-4-0125-preview", max_tokens=1000)

# 첫 번째 체인: 요리 레시피 생성
chain1 = prompt | model | output_parser

# 두 번째 prompt 템플릿
prompt2 = PromptTemplate.from_template(
    "Explain the recipe for {cuisine_name} in Korean."
)

# 입력을 그대로 전달하는 함수
def pass_query(x):
    return {"query": x}

# JSON 결과를 다음 단계의 입력으로 변환하는 함수
def prepare_second_prompt(json_result):
    return {
        "cuisine_name": json_result["name"],
        "recipe": json_result["recipe"]
    }

# 체인 구성
chain = (
    RunnablePassthrough(pass_query)
    | chain1
    | prepare_second_prompt
    | prompt2
    | model
    | StrOutputParser()
)

result = chain.invoke("Let me know how to cook Bibimbap")

print("\nFinal Result:")
print(result)
