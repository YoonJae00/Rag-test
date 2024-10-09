personas = {
    "Joy": {
        "description": "항상 밝고 긍정적인 성격으로, 모든 일의 좋은 면을 봅니다.",
        "tone": "활기차고 긍정적인 말투",
    },
    "Sadness": {
        "description": "깊은 생각에 잠기며, 공감 능력이 뛰어납니다.",
        "tone": "우울하고 사색적인 말투",
    },
}
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

aiclient = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

import chromadb

client = chromadb.PersistentClient(path="./chroma_db")


def get_relevant_memories(persona_name, query, k=3):
    collection = get_persona_collection(persona_name)
    query_embedding = aiclient.embeddings.create(
        input=query,
        model="text-embedding-ada-002"
    ).data[0].embedding
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )
    return results['documents'][0]

def generate_response(persona_name, user_input):
    persona = personas[persona_name]
    relevant_memories = get_relevant_memories(persona_name, user_input)
    
    context = "\n".join(relevant_memories)
    prompt = f"{persona_name}의 관점에서, {persona['description']} 사용자에게 {persona['tone']}로 응답하세요.\n\n관련 기억:\n{context}\n\n사용자: {user_input}\n{persona_name}:"
    
    response = aiclient.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=150,
        temperature=0.7,
    )
    return response.choices[0].text.strip()

import uuid

def store_embedding(persona_name, text, is_user=True):
    embedding = aiclient.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    ).data[0].embedding
    collection = get_persona_collection(persona_name)
    metadata = {"is_user": is_user, "persona": persona_name}
    unique_id = str(uuid.uuid4())  # 고유한 ID 생성
    collection.add(
        documents=[text],
        embeddings=[embedding],
        metadatas=[metadata],
        ids=[unique_id]  # 고유한 ID 추가
    )

def get_persona_collection(persona_name):
    return client.get_or_create_collection(f"inside_out_persona_{persona_name}")

def chat_with_persona(persona_name):
    print(f"{persona_name}와 대화를 시작합니다. 'exit'을 입력하면 종료됩니다.")
    while True:
        user_input = input("당신: ")
        if user_input.lower() == 'exit':
            break
        response = generate_response(persona_name, user_input)
        print(f"{persona_name}: {response}")
        # 대화 내역 저장
        store_embedding(persona_name, f"사용자: {user_input}", is_user=True)
        store_embedding(persona_name, f"{persona_name}: {response}", is_user=False)

def main():
    print("사용 가능한 페르소나:")
    for persona in personas.keys():
        print(f"- {persona}")
    selected_persona = input("대화할 페르소나를 선택하세요: ")
    if selected_persona in personas:
        chat_with_persona(selected_persona)
    else:
        print("선택한 페르소나가 존재하지 않습니다.")

if __name__ == "__main__":
    main()