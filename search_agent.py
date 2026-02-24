from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from tavily import TavilyClient

tavily = TavilyClient()

@tool
def search(query: str) -> str:
    """
        Tool that searches internt.
        Args:
            query (str): The search query.
        Returns:
            str: The search results.
    """
    res = tavily.search(query)
    return f"Search results for '{query}': {res}"


def main():
    print("Hello, RAG!")
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    tools = [search]
    agent = create_agent(model=llm, tools=tools)
    response = agent.invoke({"messages": [HumanMessage(content="Search for three internship postings for AI research in the US in LinkedIn.")]})
    print(response)


if __name__ == "__main__":
    main()