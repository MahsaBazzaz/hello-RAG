from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()


@tool
def multiply(x: float, y: float) -> float:
    """Multiply 'x' times 'y'."""
    return x * y


if __name__ == "__main__":
    print("Hello Tool Calling")

    tools = [TavilySearchResults(), multiply]
    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt="""
    You are a weather assistant.
    Use tools when necessary.
    Always respond in Celsius.
    """
    )

    res = agent.invoke(
        {
        "messages": [
            {"role": "user", "content": "What is the weather in Dubai right now? Compare it with San Francisco. Output in Celsius."}
        ]
        }
    )

    print(res)
