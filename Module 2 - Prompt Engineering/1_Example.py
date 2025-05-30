from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain_core.runnables import RunnableLambda

demo_template = '''I want you to act as a acting financial advisor for people.
In an easy way, explain the basics of {financial_concept}.'''

prompt = PromptTemplate(
    input_variables=['financial_concept'],
    template=demo_template
)


llm = OpenAI(temperature=0.7)

chain = prompt | llm

response = chain.invoke({"financial_concept": "GDP"})
print(response)

