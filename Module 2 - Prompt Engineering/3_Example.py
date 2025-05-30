from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI 

# Step 1: Define few-shot examples
examples = [
    {"word": "happy", "antonym": "sad"},
    {"word": "tall", "antonym": "short"},
]

# Step 2: Format the individual example
example_formatter_template = "Word: {word}, Antonym: {antonym}"
example_prompt = PromptTemplate(
    input_variables=["word", "antonym"],
    template=example_formatter_template,
)

# Step 3: Construct the full few-shot prompt
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Give the antonym of every input",
    suffix="Word: {input}\nAntonym:",
    input_variables=["input"],
    example_separator="\n"
)

# Step 4: Define the LLM
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Step 5: Build the chain
chain = LLMChain(llm=llm, prompt=few_shot_prompt)

# Step 6: Invoke the chain
response = chain.invoke({'input': "big"})
print(response['text'])
