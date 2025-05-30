from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, ConfigurableField
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

embeddings = OpenAIEmbeddings()
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

social_media_data = [
    {
        "manager": "Sarah Chen", 
        "company": "SunnyVibes", 
        "account": "@sunnyvibes_summer", 
        "platform": "Instagram", 
        "account_id": "SM001",
        "likes": "Summer content, beach photos, hot weather activities, sunny days, outdoor events",
        "dislikes": "Winter sports, cold weather, indoor activities, rainy day content, snow photos",
        "details": "Summer lifestyle brand, 80K followers, seasonal products focus"
    },
    {
        "manager": "Mike Rodriguez", 
        "company": "CozyWinter", 
        "account": "@cozywinter_life", 
        "platform": "Instagram", 
        "account_id": "SM002",
        "likes": "Winter activities, cozy indoor content, hot drinks, snow scenes, cold weather fashion",
        "dislikes": "Summer heat, beach content, hot weather complaints, outdoor summer events, sweaty activities",
        "details": "Winter lifestyle brand, 65K followers, cold season enthusiasts"
    },
    {
        "manager": "Sarah Chen", 
        "company": "FastFeast", 
        "account": "@fastfeast_quick", 
        "platform": "TikTok", 
        "account_id": "SM003",
        "likes": "Quick recipes, fast food, instant meals, time-saving cooking, microwave hacks",
        "dislikes": "Slow cooking, elaborate recipes, farm-to-table content, lengthy prep time, complex ingredients",
        "details": "Fast food content, 200K followers, busy lifestyle audience"
    },
    {
        "manager": "Sarah Chen", 
        "company": "SlowKitchen", 
        "account": "@slowkitchen_craft", 
        "platform": "TikTok", 
        "account_id": "SM004",
        "likes": "Slow cooking, elaborate recipes, artisanal ingredients, lengthy prep processes, traditional techniques",
        "dislikes": "Fast food, instant meals, microwave cooking, shortcuts, processed ingredients",
        "details": "Artisanal cooking content, 150K followers, culinary purists audience"
    },
    {
        "manager": "Sarah Chen", 
        "company": "CityLife", 
        "account": "@citylife_urban", 
        "platform": "Twitter", 
        "account_id": "SM005",
        "likes": "Urban lifestyle, city events, nightlife, public transport, skyscrapers, busy streets",
        "dislikes": "Rural content, countryside photos, nature walks, quiet environments, farm life",
        "details": "Urban lifestyle blog, 45K followers, metropolitan audience"
    },
    {
        "manager": "Sarah Chen", 
        "company": "CountryEscape", 
        "account": "@country_escape", 
        "platform": "Twitter", 
        "account_id": "SM006",
        "likes": "Rural lifestyle, countryside views, nature walks, farm life, quiet environments, wildlife",
        "dislikes": "City noise, urban pollution, traffic, crowded places, high-rise buildings",
        "details": "Rural lifestyle content, 35K followers, nature lovers audience"
    },
    {
        "manager": "Mike Rodriguez", 
        "company": "TechNow", 
        "account": "@technow_future", 
        "platform": "LinkedIn", 
        "account_id": "SM007",
        "likes": "Latest technology, AI innovations, digital transformation, automation, cutting-edge gadgets",
        "dislikes": "Traditional methods, manual processes, outdated technology, analog solutions, tech skepticism",
        "details": "Tech innovation content, 30K connections, early adopters audience"
    },
    {
        "manager": "Mike Rodriguez", 
        "company": "RetroTech", 
        "account": "@retrotech_classic", 
        "platform": "LinkedIn", 
        "account_id": "SM008",
        "likes": "Vintage technology, traditional methods, analog solutions, manual processes, tech nostalgia",
        "dislikes": "Over-automation, AI hype, digital-only solutions, tech dependency, constant upgrades",
        "details": "Vintage tech appreciation, 20K connections, tech traditionalists audience"
    },
    {
        "manager": "Mike Rodriguez", 
        "company": "MinimalLiving", 
        "account": "@minimal_clean", 
        "platform": "Instagram", 
        "account_id": "SM009",
        "likes": "Minimalist aesthetics, clean spaces, decluttering, simple living, neutral colors",
        "dislikes": "Cluttered spaces, excessive decorations, maximalist design, bright colors, busy patterns",
        "details": "Minimalist lifestyle brand, 90K followers, simplicity advocates"
    },
    {
        "manager": "Mike Rodriguez", 
        "company": "MaximalistHome", 
        "account": "@maximalist_bold", 
        "platform": "Instagram", 
        "account_id": "SM010",
        "likes": "Bold decorations, colorful designs, eclectic collections, pattern mixing, statement pieces",
        "dislikes": "Minimalist design, neutral colors, empty spaces, understated decor, monochrome schemes",
        "details": "Maximalist design content, 70K followers, bold design enthusiasts"
    },
]

documents = [
    Document(
        page_content=f"""Social media manager {account['manager']} manages {account['account']} ({account['platform']}) for {account['company']}. 
        Account preferences - LIKES: {account['likes']}. DISLIKES: {account['dislikes']}. 
        Account details: {account['details']}""",
        metadata={
            "manager": account['manager'], 
            "company": account['company'], 
            "account": account['account'],
            "platform": account['platform'],
            "account_id": account['account_id'],
            "likes": account['likes'],
            "dislikes": account['dislikes']
        }
    )
    for account in social_media_data
]

vector_store.add_documents(documents=documents)

llm = ChatOpenAI(model="gpt-4o")

configurable_retriever = vector_store.as_retriever().configurable_fields(
    search_kwargs=ConfigurableField(
        id="search_kwargs",
        name="Search Kwargs",
        description="Search parameters including filters"
    )
)

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": configurable_retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

    
result1 = chain.with_config(
    configurable={"search_kwargs": {"filter": {"manager": "Sarah Chen"}}}).invoke("What are the platforms that I manage?")
print(result1)

print("\n" + "="*50 + "\n")
    
result2 = chain.with_config(
    configurable={"search_kwargs": {"filter": {"$and": [{"manager": "Mike Rodriguez"}, {"platform": "LinkedIn"}]}}}
).invoke("Can you generate some captions for the content these accounts can post?")
print(result2)