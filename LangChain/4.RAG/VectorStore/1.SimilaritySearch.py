from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

loader = TextLoader('/Users/yoonjae/Desktop/AI-X/RAG/LangChain/4.RAG/DocumentLoader/history.txt')
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, # 각 텍스트 청크의 목표 크기를 토큰 수로 지정.
    chunk_overlap=50, # 연속된 청크 간의 중복되는 토큰 수를 지정.
    encoding_name='cl100k_base' # 'cl100k_base'는 OpenAI의 GPT-4와 최신 GPT-3.5 모델에서 사용되는 인코딩
)

texts = text_splitter.split_text(data[0].page_content)
print(texts[0])

embeddings_model = OpenAIEmbeddings()
db = Chroma.from_texts(
    texts, 
    embeddings_model,
    collection_name = 'history',
    persist_directory = './db/chromadb',
    collection_metadata = {'hnsw:space': 'cosine'}, # 유사도 계산에 코사인 유사도를 사용
)


query = '누가 한글을 창제했나요?'
docs = db.similarity_search(query)
print('result : ', docs[0].page_content)