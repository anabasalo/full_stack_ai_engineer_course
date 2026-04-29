from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

# schema
class Review(TypedDict):
    summary: str
    sentiment: str
    
structured_model = model.with_structured_output(Review)

prompt = """
This hardware is great, but the software feels kind of bloated. So many boilerplate apps and my phone keeps hanging when I play games.
"""

result = structured_model.invoke(prompt)

print(result)
