from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

load_dotenv()




def main():
    print("Hello, RAG!")
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    tools = [TavilySearch()]
    agent = create_agent(model=llm, tools=tools)
    response = agent.invoke({"messages": [HumanMessage(content="Search for three internship postings for AI research in the US in LinkedIn.")]})
    print(response)


if __name__ == "__main__":
    main()