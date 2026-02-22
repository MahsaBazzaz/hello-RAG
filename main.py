from dotenv import load_dotenv
import os
load_dotenv()

def main():
    print("Hello, RAG!")
    print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))
    print("LANGCHAIN_API_KEY:", os.getenv("LANGCHAIN_API_KEY"))

if __name__ == "__main__":
    main()