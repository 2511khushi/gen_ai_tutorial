from langchain_community.retrievers import WikipediaRetriever
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore

# 1. Retrieve a Wikipedia article
wiki_retriever = WikipediaRetriever()
wiki_docs = wiki_retriever.invoke("France")
print("Wikipedia Snippet (France):\n", wiki_docs[0].page_content[:200])
print("="*100 + "\n")

# 2. Load LangChain introduction page
loader = WebBaseLoader("https://en.wikipedia.org/wiki/President_of_the_United_States")
documents = loader.load()

# 3. Split the web page into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)
chunks = text_splitter.split_documents(documents)

# 4. Convert the chunks into embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = InMemoryVectorStore.from_documents(chunks, embeddings)

# 5. Create a retriever and query it
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
query = "How long can senators serve?"
docs = retriever.invoke(query)

# 6. Display the results
def print_docs(docs):
    for i, doc in enumerate(docs, 1):
        print(f"Result {i}:\n{doc.page_content[:500]}")
        print("-" * 100 + "\n")

print(f"Query: {query}\n")
print_docs(docs)
