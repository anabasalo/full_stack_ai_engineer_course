from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()
model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

chat_history = [
    SystemMessage(content="You are a helpful AI assistant."),
]

while True:
    user_input=input("User: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input.lower() in ['exit', 'quit']:
        print("Exiting chat.")
        break
    response = model.invoke(chat_history)
    chat_history.append(AIMessage(content=response.content))
    print(f"AI: {response.content}")
    
print("Chat history:", chat_history)