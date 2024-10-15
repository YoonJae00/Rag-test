from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import PromptTemplate

# # 기본 프롬프트 템플릿 정의
# prompt = PromptTemplate.from_template("지구의 {layer}에서 가장 흔한 원소는 {element}입니다.")

# # 'layer' 변수에 '지각' 값을 미리 지정하여 부분 포맷팅
# partial_prompt = prompt.partial(layer="지각")

# # 나머지 'element' 변수만 입력하여 완전한 문장 생성
# print(partial_prompt.format(element="산소"))


# 프롬프트 초기화 시 부분 변수 지정
prompt = PromptTemplate(
    template="지구의 {layer}에서 가장 흔한 원소는 {element}입니다.",
    input_variables=["element"],  # 사용자 입력이 필요한 변수
    partial_variables={"layer": "맨틀"}  # 미리 지정된 부분 변수
)

# 남은 'element' 변수만 입력하여 문장 생성
print(prompt.format(element="규소"))

from datetime import datetime

# 현재 계절을 반환하는 함수 정의
def get_current_season():
    month = datetime.now().month
    if 3 <= month <= 5:
        return "봄"
    elif 6 <= month <= 8:
        return "여름"
    elif 9 <= month <= 11:
        return "가을"
    else:
        return "겨울"

# 함수를 사용한 부분 변수가 있는 프롬프트 템플릿 정의
prompt = PromptTemplate(
    template="{season}에 일어나는 대표적인 지구과학 현상은 {phenomenon}입니다.",
    input_variables=["phenomenon"],  # 사용자 입력이 필요한 변수
    partial_variables={"season": get_current_season}  # 함수를 통해 동적으로 값을 생성하는 부분 변수
)

# 'phenomenon' 변수만 입력하여 현재 계절에 맞는 문장 생성
print(prompt.format(phenomenon="꽃가루 증가"))
