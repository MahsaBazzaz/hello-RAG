import glob
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

def main():
    print("ingesting...")
   
    paths = glob.glob("docs/*.txt")

    documents = []
    for p in paths:
        loader = TextLoader(p)
        documents.extend(loader.load())
        
    print("spliting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.getenv("INDEX_NAME"))


if __name__ == "__main__":
    main()