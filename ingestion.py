import glob
import os
from dotenv import load_dotenv
from operator import itemgetter

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

print("ingesting...")
paths = glob.glob("docs/*.txt")
documents = []
for p in paths:
    loader = TextLoader(p)
    documents.extend(loader.load())
    
print("spliting...")
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

print("initializing...")
llm = ChatOpenAI()
embeddings = OpenAIEmbeddings()
vector_store = PineconeVectorStore(embedding=embeddings, index_name=os.getenv("INDEX_NAME"))
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_template(
        """
        Answer the question based only on the following context.
        {context}
        Question: {question}
        provide a detailed answer.
        """
    )

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

def create_retrieval_chain():
    retrieval_chain = (
        RunnablePassthrough.assign(
        context=itemgetter("question")|retriever | format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )

    return retrieval_chain

def main():
    print("hello RAG")
    # PineconeVectorStore.from_documents(texts, embeddings, index_name=os.getenv("INDEX_NAME"))
    

if __name__ == "__main__":
    main()
    query = "What is the main topic of the documents?"
    chain = create_retrieval_chain()
    answer = chain.invoke({"question": query})
    print(answer)