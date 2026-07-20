from dotenv import load_dotenv
# from langchain_huggingface import HuggingFaceEndpoint
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model='gemini-3.5-flash',
    temperature=0.3,
    max_output_tokens=512,
)

response = llm.invoke("Say hi in one sentence.")
print(response)
