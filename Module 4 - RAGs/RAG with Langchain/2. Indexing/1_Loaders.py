from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer


file_path = "/Users/apple/Desktop/LangChain/Module 4 - RAGs/RAG with Langchain/2. Indexing/text.txt"
loader = TextLoader(file_path, encoding="utf-8")
document = loader.load()
print(document[0])


# Define a custom User-Agent that mimics a Chrome browser on macOS, which should bypass most basic bot protections.
headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
}
loader = WebBaseLoader("https://python.langchain.com/docs/introduction/", header_template=headers)
documents = loader.load()
print(documents[0])
print(documents[0].metadata)


# Load multiple pages
headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
}

loader = WebBaseLoader(
    ["https://www.espn.com/", "https://python.langchain.com/docs/integrations/document_loaders/web_base/"],
    header_template=headers
)

documents = loader.load()

if len(documents) > 1:
    print(documents[1].page_content)
else:
    print(f"Only {len(documents)} document(s) were loaded. Expected at least 2.")


# Loading CSV files into Document objects
file_path = "/Users/apple/Desktop/LangChain/Module 4 - RAGs/RAG with Langchain/2. Indexing/Air_Quality.csv"
loader = CSVLoader(file_path=file_path, content_columns=["Name"])
documents = loader.load()
print(len(documents))


# Loading PDFs
loader = PyPDFLoader("/Users/apple/Desktop/LangChain/Module 4 - RAGs/RAG with Langchain/2. Indexing/moby-dick.pdf")
documents = loader.load()

print("number of pages: ", len(documents))

print("First 500 characters of first page:")
print(documents[7].page_content[:500])


