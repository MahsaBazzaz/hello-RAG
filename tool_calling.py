from dotenv import load_dotenv
from typing import List
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()


@tool
def multiply(x: float, y: float) -> float:
    """Multiply 'x' times 'y'."""
    return x * y

@tool
def sum(x: float, y: float) -> float:
    """Sum 'x' and 'y'."""
    return x + y

class Source(BaseModel):
    """ Scheme used for source used by the agent."""
    url: str = Field(description="URL of the source")

class AgentResponse(BaseModel):
    """ Scheme used for agent response."""
    answer: str = Field(description="Agent's answer to the question")
    sources: List[Source] = Field(default_factory=list, description="List of sources used to answer the question")

if __name__ == "__main__":
    print("Hello Tool Calling")

    tools = [TavilySearchResults(), multiply, sum]
    # llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)
    llm = ChatOllama(model = "llama3.1:8b", temperature=0)
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt="""
    You are a weather assistant.
    Use tools when necessary.
    Always respond in Celsius.
    """,
    response_format=AgentResponse
    )

    res = agent.invoke(
        {
        "messages": [
            {"role": "user", "content": "Give me the summation of weather temperatures for Boston right now and San Francisco right now. Output in Celsius."}
        ]
        }
    )

    print(res)
