from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    BaseMessage
)
from langchain.memory import ConversationBufferMemory
from typing import List
import time

class ConversationAgent:
    def __init__(self, name: str, system_message: str):
        self.name = name
        self.llm = ChatOpenAI(temperature=0.7)
        self.system_message = SystemMessage(content=system_message)
        self.memory = ConversationBufferMemory()
        self.message_history: List[BaseMessage] = [self.system_message]

    def send_message(self, message: str, sender: str) -> str:
        # Add the incoming message to history
        self.message_history.append(HumanMessage(content=f"{sender}: {message}"))
        
        # Generate response
        response = self.llm.predict_messages(self.message_history)
        
        # Add the response to history
        self.message_history.append(AIMessage(content=response.content))
        
        return response.content

def simulate_conversation():
    # Initialize agents
    agent1 = ConversationAgent(
        "Tech Expert",
        "You are a technical expert who loves discussing programming and technology."
    )
    
    agent2 = ConversationAgent(
        "Business Analyst",
        "You are a business analyst who focuses on practical business applications of technology."
    )
    
    conversation_active = True
    current_speaker = agent1
    other_speaker = agent2
    
    print("대화가 시작되었습니다. 언제든 'intervene'을 입력하여 대화에 참여하실 수 있습니다.")
    print("대화를 종료하려면 'quit'을 입력하세요.\n")
    
    initial_message = "AI 가 발전하면 자율주행 자동차는 상용화 될까?"
    print(f"{agent1.name}: {initial_message}")
    
    while conversation_active:
        # Get response from current speaker
        response = other_speaker.send_message(initial_message, current_speaker.name)
        print(f"\n{other_speaker.name}: {response}")
        
        # Allow user intervention
        user_input = input("\n대화에 참여하시려면 'intervene'을 입력하세요 (또는 Enter를 눌러 계속): ")
        
        if user_input.lower() == 'quit':
            conversation_active = False
            print("\n대화가 종료되었습니다.")
            break
        
        elif user_input.lower() == 'intervene':
            user_message = input("\n메시지를 입력하세요: ")
            
            # Both agents respond to user
            response1 = agent1.send_message(user_message, "User")
            print(f"\n{agent1.name}: {response1}")
            
            response2 = agent2.send_message(user_message, "User")
            print(f"\n{agent2.name}: {response2}")
            
            initial_message = response2  # Continue conversation from agent2's response
            
        else:
            # Switch speakers
            initial_message = response
            temp = current_speaker
            current_speaker = other_speaker
            other_speaker = temp
        
        time.sleep(1)  # Add small delay between messages

if __name__ == "__main__":
    simulate_conversation()