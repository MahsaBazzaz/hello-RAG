from dotenv import load_dotenv

from typing import List
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from callbacks import AgentCallbackHandler

load_dotenv()

class Source(BaseModel):
    """ Scheme used for source used by the agent."""
    url: str = Field(description="URL of the source")

class AgentResponse(BaseModel):
    """ Scheme used for agent response."""
    answer: str = Field(description="Agent's answer to the question")
    sources: List[Source] = Field(default_factory=list, description="List of sources used to answer the question")

def main():
    print("Hello, RAG!")
    # llm = ChatOpenAI(model="gpt-4o", temperature=0)
    llm = ChatOllama(model="llama3.1:8b", temperature=0, callbacks=[AgentCallbackHandler()])
    tools = [TavilySearch()]
    agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)
    response = agent.invoke({"messages": [HumanMessage(content="What is the weather in Dubai right now? Compare it with San Francisco. Output in Celsius")]})
    print(response)


if __name__ == "__main__":
    main()