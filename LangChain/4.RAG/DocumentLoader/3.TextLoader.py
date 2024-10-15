from langchain_community.document_loaders import TextLoader

loader = TextLoader('LangChain/4.RAG/DocumentLoader/history.txt')
data = loader.load()

print(type(data))
print(len(data))

print(data)