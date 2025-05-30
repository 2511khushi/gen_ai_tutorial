from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import MarkdownHeaderTextSplitter

file_path = "/Users/apple/Desktop/LangChain/Module 4 - RAGs/RAG with Langchain/2. Indexing/text.txt"
loader = TextLoader(file_path, encoding="utf-8")
document = loader.load()
print(document[0])

# a basic text splitter
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separator="\n")
chunks = text_splitter.split_documents(document)
print("Number of chunks: ", len(chunks))

for chunk in chunks:
    print(f"Chunk {chunks.index(chunk)} size: {len(chunk.page_content)}")
    print(chunk.page_content)



headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
}
loader = WebBaseLoader("https://python.langchain.com/docs/introduction/", header_template=headers)
documents = loader.load()
print(documents[0].page_content)

# improving splitting by using the RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""])
chunks = text_splitter.split_documents(documents)
print("Number of chunks: ", len(chunks))

for i in range(5):
    print(f"Chunk {i+1} size: {len(chunks[i].page_content)}")

print("\nEnd of chunk 0:")
print(chunks[3].page_content[-300:])

print("\nBeginning of chunk 1:")
print(chunks[4].page_content[:300])



# # Splitting based on document structure
markdown_document = "# Foo\n\n ## Bar\n\nHi this is Jim\n\nHi this is Joe\n\n ### Boo \n\n Hi this is Lance \n\n ## Baz\n\n Hi this is Molly"
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
md_header_splits = markdown_splitter.split_text(markdown_document)
print(md_header_splits)