from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo")

# Example 1: Templated Email Prompt 
template = (
    "Write a {tone} email to {company} expressing interest in the "
    "{position} position, mentioning {skill} as a key strength. "
    "Keep it to 4 lines max."
)
prompt_template = ChatPromptTemplate.from_template(template)
prompt = prompt_template.invoke({
    "tone": "energetic", 
    "company": "Samsung", 
    "position": "AI Engineer", 
    "skill": "AI"
})

result = llm.invoke(prompt)
print(result.content)


# Example 2: Prompt with System and Human Messages (Using Tuples)
messages = [
    ("system", "You are a comedian who tells jokes about {topic}."),
    ("human", "Tell me {joke_count} jokes."),
]

prompt_template_jokes = ChatPromptTemplate.from_messages(messages)

prompt_jokes = prompt_template_jokes.invoke({
    "topic": "lawyers", 
    "joke_count": 3
})

result_jokes = llm.invoke(prompt_jokes)
print("\n----- Jokes Prompt Output -----")
print(result_jokes.content)










from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo")

# template = "Write a {tone} email to {company} expressing interest in the {position} position, mentioning {skill} as a key strength. Keep it to 4 lines max"
# prompt_template = ChatPromptTemplate.from_template(template)

# prompt =  prompt_template.invoke({
#     "tone": "energetic", 
#     "company": "samsung", 
#     "position": "AI Engineer", 
#     "skill": "AI"
# })

# result = llm.invoke(prompt)
# print(result.content)


# Example 2: Prompt with System and Human Messages (Using Tuples)
messages = [
    ("system", "You are a comedian who tells jokes about {topic}."),
    ("human", "Tell me {joke_count} jokes."),
]

prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"topic": "lawyers", "joke_count": 3})
result = llm.invoke(prompt)
print(result)