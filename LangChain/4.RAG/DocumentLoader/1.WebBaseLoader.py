from dotenv import load_dotenv
load_dotenv()

# Data Loader - 웹페이지 데이터 가져오기
from langchain_community.document_loaders import WebBaseLoader

# 위키피디아 정책과 지침
# url = 'https://ko.wikipedia.org/wiki/%EC%9C%84%ED%82%A4%EB%B0%B1%EA%B3%BC:%EC%A0%95%EC%B1%85%EA%B3%BC_%EC%A7%80%EC%B9%A8'
url = 'https://wikidocs.net/231147'
loader = WebBaseLoader(url)

# 웹페이지 텍스트 -> Documents
docs = loader.load()

print(len(docs))
print(len(docs[0].page_content))
print(docs[0].page_content[5000:6000])

# Text Split (Documents -> small chunks: Documents)
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

print(len(splits))
print(splits[10])
# metadata 속성
print(splits[10].metadata)

# Indexing (Texts -> Embedding -> Store)
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

vectorstore = Chroma.from_documents(documents=splits,
                                    embedding=OpenAIEmbeddings())

# docs = vectorstore.similarity_search("인덱싱 방법에 대해 알려주세요.")
# print(len(docs))
# print(docs[0].page_content)

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Prompt
template = '''Answer the question based only on the following context:
{context}

Question: {question}
'''

prompt = ChatPromptTemplate.from_template(template)



# LLM
model = ChatOpenAI(model='gpt-4o-mini', temperature=0)

# Rretriever
retriever = vectorstore.as_retriever()

# Combine Documents
def format_docs(docs):
    return '\n\n'.join(doc.page_content for doc in docs)

# RAG Chain 연결
rag_chain = (
    {'context': retriever | format_docs, 'question': RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# Chain 실행
result = rag_chain.invoke("이력에 대해 알려주세요.")
print(result)

# 1. 입력 질문 처리:
# "이력에 대해 알려주세요."라는 질문이 rag_chain.invoke()를 통해 체인에 입력됩니다.
# Retriever 실행:
# retriever가 질문과 관련된 문서를 벡터 저장소에서 검색합니다.
# 문서 포맷팅:
# 검색된 문서들이 format_docs 함수를 통해 하나의 문자열로 결합됩니다.
# Prompt 구성:
# RunnablePassthrough()를 통해 원본 질문이 그대로 전달됩니다.
# 검색된 문서(context)와 원본 질문이 미리 정의된 prompt 템플릿에 삽입됩니다.
# LLM (Language Model) 실행:
# 구성된 prompt가 ChatOpenAI 모델에 입력되어 응답을 생성합니다.
# 출력 파싱:
# 모델의 출력이 StrOutputParser()를 통해 문자열로 변환됩니다.