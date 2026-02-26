from dotenv import load_dotenv
import os
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

load_dotenv()

def main():
    print("Hello, RAG!")
    with open("info.txt", "r") as f:
        info = f.read()

    if info:
        print("Information loaded successfully.")

        my_template = """
        given the following information: {info} about this game:
        1. What is the name of the game?
        2. What is the genre of the game?
        3. What is the release date of the game?
        4. What is the developer of the game?
        5. Tell me two interesting facts about the game.
        """

        my_prompt_template = PromptTemplate(
            input_variables=["info"],
            template=my_template
        )
        # llm = ChatOpenAI(model="gpt-5", temperature=0)
        # llm = ChatOllama(model = "gemma3:270m", temperature=0)
        llm = ChatOllama(model = "llama3.1:8b", temperature=0)
        chain = my_prompt_template | llm
        response = chain.invoke(input={"info": info})
        print("Response from the model:")
        print(response.content)
    else:
        print("No information found in info.txt.")


if __name__ == "__main__":
    main()