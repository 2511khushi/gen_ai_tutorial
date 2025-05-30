from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnableSequence
# from langchain.chains import PALChain
from langchain.chains.api import open_meteo_docs
from langchain.chains.api.prompt import API_RESPONSE_PROMPT
from langchain.chains import APIChain
from langchain.chains.api import open_meteo_docs


load_dotenv()

article = '''Coinbase, the second-largest crypto exchange by trading volume, released its Q4 2022 earnings on Tuesday, giving shareholders and market players alike an updated look into its financials. In response to the report, the company's shares are down modestly in early after-hours trading.In the fourth quarter of 2022, Coinbase generated $605 million in total revenue, down sharply from $2.49 billion in the year-ago quarter. Coinbase's top line was not enough to cover its expenses: The company lost $557 million in the three-month period on a GAAP basis (net income) worth -$2.46 per share, and an adjusted EBITDA deficit of $124 million.Wall Street expected Coinbase to report $581.2 million in revenue and earnings per share of -$2.44 with adjusted EBITDA of -$201.8 million driven by 8.4 million monthly transaction users (MTUs), according to data provided by Yahoo Finance.Before its Q4 earnings were released, Coinbase's stock had risen 86% year-to-date. Even with that rally, the value of Coinbase when measured on a per-share basis is still down significantly from its 52-week high of $206.79.That Coinbase beat revenue expectations is notable in that it came with declines in trading volume; Coinbase historically generated the bulk of its revenues from trading fees, making Q4 2022 notable. Consumer trading volumes fell from $26 billion in the third quarter of last year to $20 billion in Q4, while institutional volumes across the same timeframe fell from $133 billion to $125 billion.The overall crypto market capitalization fell about 64%, or $1.5 trillion during 2022, which resulted in Coinbase's total trading volumes and transaction revenues to fall 50% and 66% year-over-year, respectively, the company reported.As you would expect with declines in trading volume, trading revenue at Coinbase fell in Q4 compared to the third quarter of last year, dipping from $365.9 million to $322.1 million. (TechCrunch is comparing Coinbase's Q4 2022 results to Q3 2022 instead of Q4 2021, as the latter comparison would be less useful given how much the crypto market has changed in the last year; we're all aware that overall crypto activity has fallen from the final months of 2021.)There were bits of good news in the Coinbase report. While Coinbase's trading revenues were less than exuberant, the company's other revenues posted gains. What Coinbase calls its "subscription and services revenue" rose from $210.5 million in Q3 2022 to $282.8 million in Q4 of the same year, a gain of just over 34% in a single quarter.And even as the crypto industry faced a number of catastrophic events, including the Terra/LUNA and FTX collapses to name a few, there was still growth in other areas. The monthly active developers in crypto have more than doubled since 2020 to over 20,000, while major brands like Starbucks, Nike and Adidas have dived into the space alongside social media platforms like Instagram and Reddit.With big players getting into crypto, industry players are hoping this move results in greater adoption both for product use cases and trading volumes. Although there was a lot of movement from traditional retail markets and Web 2.0 businesses, trading volume for both consumer and institutional users fell quarter-over-quarter for Coinbase.Looking forward, it'll be interesting to see if these pieces pick back up and trading interest reemerges in 2023, or if platforms like Coinbase will have to keep looking elsewhere for revenue (like its subscription service) if users continue to shy away from the market.'''

llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0, max_tokens=256)


# FACT EXTRACTION PROMPT
fact_extraction_prompt = PromptTemplate(
    input_variables=["text_input"],
    template="Extract the key facts out of this text. Don't include opinions. Give each fact a number and keep them short sentences. :\n\n {text_input}"
)
fact_extraction_chain = fact_extraction_prompt | llm

facts = fact_extraction_chain.invoke({"text_input": article})
print("Facts:\n", facts.content)


# INVESTOR UPDATE PROMPT
investor_update_prompt = PromptTemplate(
    input_variables=["facts"],
    template="You are a Goldman Sachs analyst. Take the following list of facts and use them to write a short paragraph for investors. Don't leave out key info:\n\n {facts}"
)
investor_update_chain = investor_update_prompt | llm

investor_update = investor_update_chain.invoke({"facts": facts.content})
print("\nInvestor Update:\n", investor_update.content)


# TRIPLES EXTRACTION PROMPT (subject, predicate, object)
triples_prompt = PromptTemplate(
    input_variables=["facts"],
    template="Take the following list of facts and turn them into triples for a knowledge graph:\n\n {facts}"
)
triples_chain = triples_prompt | llm

triples = triples_chain.invoke({"facts": facts.content})
print("\nTriples:\n", triples.content)


# SIMPLE SEQUENTIAL CHAIN (Combined using RunnableSequence)
full_chain = RunnableSequence(first=fact_extraction_chain, last=investor_update_chain)
response = full_chain.invoke({"text_input": article})
print("\nFinal Investor Update (via Sequential Chain):\n", response.content)


# PALChain Example
# llm_code = ChatOpenAI(model_name='code-davinci-002', temperature=0, max_tokens=512)
# pal_chain = PALChain.from_math_prompt(llm_code, verbose=True)

# question = "The cafeteria had 23 apples. If they used 20 for lunch and bought 6 more, how many apples do they have?"
# pal_result = pal_chain.run(question)
# print("\nPAL Chain Result:\n", pal_result)

math_prompt = PromptTemplate(
    input_variables=["question"],
    template="Solve the following math problem step-by-step:\n\n{question}"
)
math_chain = math_prompt | llm

question = "The cafeteria had 23 apples. If they used 20 for lunch and bought 6 more, how many apples do they have?"
math_result = math_chain.invoke({"question": question})
print("\nMath Problem Result:\n", math_result.content)


# APIChain for weather
llm_api = ChatOpenAI(temperature=0, max_tokens=100)

api_chain = APIChain.from_llm_and_api_docs(
    llm=llm_api,
    api_docs=open_meteo_docs.OPEN_METEO_DOCS,
    limit_to_domains=["https://api.open-meteo.com"],
    verbose=True
)

weather_now = api_chain.invoke("What is the temperature like right now in North India in degrees Celsius?")
print("\nWeather Now:\n", weather_now)

is_raining = api_chain.invoke("Is it raining in North India?")
print("\nIs it Raining?\n", is_raining)