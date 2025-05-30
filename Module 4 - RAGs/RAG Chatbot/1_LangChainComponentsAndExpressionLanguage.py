# Large Language Model (LLM)

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")
llm_response = llm.invoke("Tell me a joke")
print(llm_response)


# Output Parsers

from langchain_core.output_parsers import StrOutputParser

output_parser = StrOutputParser()
chain = llm | output_parser
result = chain.invoke("Tell me a joke")
print(result)


# Structured Output

from typing import List
from pydantic import BaseModel, Field

class MobileReview(BaseModel):
    phone_model: str = Field(description="Name and model of the phone")
    rating: float = Field(description="Overall rating out of 5")
    pros: List[str] = Field(description="List of positive aspects")
    cons: List[str] = Field(description="List of negative aspects")
    summary: str = Field(description="Brief summary of the review")

review_text = """
Just got my hands on the new Galaxy S21 and wow, this thing is slick! The screen is gorgeous,
colors pop like crazy. Camera's insane too, especially at night - my Insta game's never been
stronger. Battery life's solid, lasts me all day no problem.
Not gonna lie though, it's pretty pricey. And what's with ditching the charger? C'mon Samsung.
Also, still getting used to the new button layout, keep hitting Bixby by mistake.
Overall, I'd say it's a solid 4 out of 5. Great phone, but a few annoying quirks keep it from
being perfect. If you're due for an upgrade, definitely worth checking out!
"""

structured_llm = llm.with_structured_output(MobileReview)
output = structured_llm.invoke(review_text)
print(output)
print(output.pros)


# Prompt Templates

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("Tell me a short joke about {topic}")
chain = prompt | llm | output_parser
result = chain.invoke({"topic": "programming"})
print(result)


# LLM Messages

from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage(content="You are a helpful assistant that tells jokes."),
    HumanMessage(content="Tell me about programming")
]
response = llm.invoke(messages)
print(response)

template = ChatPromptTemplate([
    ("system", "You are a helpful assistant that tells jokes."),
    ("human", "Tell me about {topic}")
])
chain = template | llm
response = chain.invoke({"topic": "programming"})
print(response)


