from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
)

# SystemMessage: sets the context or role for the model.
# HumanMessage: actual user’s input or question to the model.

messages = [
    SystemMessage("You are an expert in Social Media content strategy."),
    HumanMessage("Give a short tip to create an enagaging post on Instagram.")
]

response = llm.invoke(messages)
print(response.content)
