from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain

template = '''In an easy way translate the following sentence: '{sentence}' into {target_language}'''
language_prompt = PromptTemplate(
    input_variables=['sentence', 'target_language'],
    template=template,
)

llm = ChatOpenAI(model="gpt-3.5-turbo")

# Just formatting prompt template (no LLM used here)
# print(language_prompt.format(sentence="How are you", target_language="hindi"))

chain = LLMChain(llm=llm, prompt=language_prompt)
response = chain.invoke({'sentence': "Hello, how are you?", 'target_language': 'hindi'})
print(response['text'])