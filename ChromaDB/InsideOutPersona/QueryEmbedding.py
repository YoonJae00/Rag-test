import chromadb
from chromadb.utils import embedding_functions
import openai
import os
from dotenv import load_dotenv
import json
import datetime
from openai import OpenAI
from datetime import date

# 환경 변수 로드
load_dotenv()

# OpenAI API 키 설정
openai.api_key = os.getenv('OPENAI_API_KEY')

# ChromaDB 클라이언트 초기화
client = chromadb.PersistentClient(path="./diary_db")

# OpenAI 클라이언트 초기화
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# OpenAI의 임베딩 함수 설정
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv('OPENAI_API_KEY'),
    model_name="text-embedding-ada-002"
)

# 일기 컬렉션 생성 또는 가져오기
diary_collection = client.get_or_create_collection(
    name="diary_entries",
    embedding_function=openai_ef
)

def extract_metadata(text):
    today = date.today().strftime("%Y-%m-%d")
    prompt = f"""
    다음 텍스트에서 날짜, 장소, 활동, 감정, 동행인, 날씨를 추출하세요. JSON 형식으로 반환하되, 날짜는 YYYY-MM-DD 형식으로 변환하세요.
    오늘 날짜는 {today}입니다. 정확한 날짜가 언급되지 않았다면 오늘 날짜를 사용하세요.

    텍스트: {text}

    JSON 형식:
    {{
        "date": "YYYY-MM-DD",
        "location": "장소",
        "activity": "활동",
        "emotion": "감정",
        "companion": "동행인",
        "weather": "날씨"
    }}
    """
    
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return json.loads(response.choices[0].message.content)

def store_embedding_with_metadata(text):
    metadata = extract_metadata(text)
    
    # 데이터를 Vector DB에 저장
    diary_collection.add(
        documents=[text],
        metadatas=[metadata],
        ids=[datetime.datetime.now().isoformat()]
    )
    
    print("일기가 성공적으로 저장되었습니다.")
    print(f"추출된 메타데이터: {metadata}")

def query_diary(query_text, n_results=3):
    query_embedding = get_embedding(query_text)
    results = diary_collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    return results

def get_embedding(text):
    response = openai_client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

# 메인 함수
def main():
    while True:
        print("\n1. 일기 추가")
        print("2. 일기 검색")
        print("3. 종료")
        choice = input("선택하세요: ")
        
        if choice == '1':
            text = input("일기를 입력하세요: ")
            store_embedding_with_metadata(text)
        
        elif choice == '2':
            query = input("검색어를 입력하세요: ")
            results = query_diary(query)
            
            print("\n검색 결과:")
            for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                print(f"\n결과 {i+1}:")
                print(f"날짜: {metadata['date']}")
                print(f"장소: {metadata['location']}")
                print(f"활동: {metadata['activity']}")
                print(f"감정: {metadata['emotion']}")
                print(f"동행인: {metadata['companion']}")
                print(f"날씨: {metadata['weather']}")
                print(f"내용: {doc}")
        
        elif choice == '3':
            print("프로그램을 종료합니다.")
            break
        
        else:
            print("잘못된 선택입니다. 다시 선택해주세요.")

if __name__ == "__main__":
    main()