from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")
result = model.invoke("How long does the president's term last?")
print(result.content)

# you can provide more settings
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5, max_tokens=1000, top_p=0.95, frequency_penalty=0, presence_penalty=0)
response = llm.invoke("What is the capital of the United States?")
print(response.content)




