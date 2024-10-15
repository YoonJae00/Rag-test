from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import PromptTemplate

example_prompt = PromptTemplate.from_template("질문: {question}\n{answer}")

examples = [
    {
        "question": "지구의 대기 중 가장 많은 비율을 차지하는 기체는 무엇인가요?",
        "answer": "지구 대기의 약 78%를 차지하는 질소입니다."
    },
    {
        "question": "광합성에 필요한 주요 요소들은 무엇인가요?",
        "answer": "광합성에 필요한 주요 요소는 빛, 이산화탄소, 물입니다."
    },
    {
        "question": "피타고라스 정리를 설명해주세요.",
        "answer": "피타고라스 정리는 직각삼각형에서 빗변의 제곱이 다른 두 변의 제곱의 합과 같다는 것입니다."
    },
    {
        "question": "지구의 자전 주기는 얼마인가요?",
        "answer": "지구의 자전 주기는 약 24시간(정확히는 23시간 56분 4초)입니다."
    },
    {
        "question": "DNA의 기본 구조를 간단히 설명해주세요.",
        "answer": "DNA는 두 개의 폴리뉴클레오티드 사슬이 이중 나선 구조를 이루고 있습니다."
    },
    {
        "question": "원주율(π)의 정의는 무엇인가요?",
        "answer": "원주율(π)은 원의 지름에 대한 원의 둘레의 비율입니다."
    }
]

from langchain_core.prompts import FewShotPromptTemplate

# FewShotPromptTemplate을 생성합니다.
prompt = FewShotPromptTemplate(
    examples=examples,              # 사용할 예제들
    example_prompt=example_prompt,  # 예제 포맷팅에 사용할 템플릿
    suffix="질문: {input}",          # 예제 뒤에 추가될 접미사
    input_variables=["input"],      # 입력 변수 지정
)

# 새로운 질문에 대한 프롬프트를 생성하고 출력합니다.
print(prompt.invoke({"input": "화성의 표면이 붉은 이유는 무엇인가요?"}).to_string())

from langchain_chroma import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings

# SemanticSimilarityExampleSelector를 초기화합니다.
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,            # 사용할 예제들
    OpenAIEmbeddings(),  # 임베딩 모델
    Chroma,              # 벡터 저장소
    k=1,                 # 선택할 예제 수
)

# 새로운 질문에 대해 가장 유사한 예제를 선택합니다.
question = "화성의 표면이 붉은 이유는 무엇인가요?"
selected_examples = example_selector.select_examples({"question": question})
print(f"입력과 가장 유사한 예제: {question}")
for example in selected_examples:
    print("\n")
    for k, v in example.items():
        print(f"{k}: {v}")

# 백터 db 유사도 검색 vs FewShot 비교
# 왜 Few-Shot을 사용하는가?

# Few-shot prompting은 벡터 DB 기반 검색으로는 처리하기 어려운 복잡한 작업에 적합합니다. 예를 들어, 모델이 단순히 유사한 문장을 찾는 것이 아니라 새로운 맥락에서 추론을 해야 하는 경우, 단순한 유사도 검색보다 더 나은 결과를 제공합니다.

# 창의적인 문제 해결: 벡터 DB는 저장된 문서에서 유사한 답을 찾는 데 유용하지만, 새로운 문제에 대해 창의적으로 추론하거나 문맥을 이해하고 새로운 정보를 생성하는 데는 한계가 있습니다.
# 추론 기반 작업: Few-shot prompting은 모델이 몇 가지 예시를 보고 학습하여 그 예시를 기반으로 새로운 문제를 해결하도록 돕습니다. 이 방식은 단순한 유사도 검색보다 더 복잡한 작업(예: 요약, 창의적 글쓰기, 논리적 추론)을 요구할 때 유리합니다.

# 결론

# 단순한 정보 검색: 벡터 DB와 유사도 검색이 더 효율적일 수 있습니다.
# 복잡한 추론 및 창의적 작업: Few-shot prompting이 더 효과적일 수 있습니다.