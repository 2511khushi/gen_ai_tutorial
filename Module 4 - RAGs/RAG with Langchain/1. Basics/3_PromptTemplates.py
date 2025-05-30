from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5, max_tokens=1000, top_p=0.95, frequency_penalty=0, presence_penalty=0)


# Using a prompt template so that we can dynamically change the prompt
prompt = PromptTemplate(template="Translate the following text to French: {input}")
chain = prompt | llm # this is a chain of the prompt and the model
result = chain.invoke({"input": "I love programming."})
print(result.content)


# Adding a system prompt, which is used to set the behavior of the model.
messages = [
    SystemMessage(content="You are a helpful assistant that answers questions about the United States. If you are asked about ANYTHING that is not related to the United States, you must say 'I cannot answer that question.'"),
    HumanMessage(content="What is your favorite color?"),
]
ai_message = llm.invoke(messages)
print(ai_message.content)


# You can also use the ChatPromptTemplate to create a prompt and then chain it to the model so that you can dynamically change the prompt
prompt = ChatPromptTemplate([
    SystemMessage(content="You are a helpful assistant that answers questions about the United States. If you are asked about ANYTHING that is not related to the United States, you must say 'I cannot answer that question.'"),
    HumanMessage(content="{input}")
])
chain = prompt | llm
response = chain.invoke({"input": "What is the capital of the United States?"})
print(response.content)


prompt = ChatPromptTemplate([
    SystemMessagePromptTemplate.from_template("You are a {occupation} named {name}. Get into character and pretend to be this role. Answer questions accordingly."),
    HumanMessagePromptTemplate.from_template("{input}")
])
chain = prompt | llm
output = chain.invoke({
    "occupation": "old wizard",
    "name": "Lord Jaquarious",
    "input": "Hi!"
})
print(output.content)


# A simplistic example of RAG to come. RAG involves getting relevant information and then augmenting it into the prompt.
prompt = ChatPromptTemplate([
    ("system", "You are a helpful assistant that answers questions. ONLY use the context provided to answer questions. If the context does not provide the answer, say 'I don't know.' Context: {context}"),
    ("user", "{input}")
])

chain = prompt | llm

x = chain.invoke({"context": "The capital of Germany is Berlin.", "input": "What is the capital of the United States?"})
print(x.content)

y =chain.invoke({"context": "The capital of Germany is Berlin.", "input": "What is the capital of Germany?"})
print(y.content)