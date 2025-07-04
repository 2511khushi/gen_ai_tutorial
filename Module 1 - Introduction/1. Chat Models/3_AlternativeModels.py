from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

#LangChain's Chat Models Docs
# https://python.langchain.com/docs/integrations/chat/

# Setup environment variables and messages
load_dotenv()

messages = [
    SystemMessage(content="Solve the following math problems"),
    HumanMessage(content="What is the square root of 49?"),
]

# ---- LangChain OpenAI Chat Model Example ----

model = ChatOpenAI(model="gpt-3.5-turbo")
result = model.invoke(messages)
print(f"Answer from OpenAI: {result.content}")


# ---- Anthropic Chat Model Example ----

# Anthropic models: https://docs.anthropic.com/en/docs/models-overview
model = ChatAnthropic(model="claude-3-opus-20240229")
result = model.invoke(messages)
print(f"Answer from Anthropic: {result.content}")


# ---- Google Chat Model Example ----

# https://console.cloud.google.com/gen-app-builder/engines
# https://ai.google.dev/gemini-api/docs/models/gemini
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
result = model.invoke(messages)
print(f"Answer from Google: {result.content}")