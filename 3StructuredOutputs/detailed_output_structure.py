from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

# schema
class Review(TypedDict):
    key_themes: Annotated[list[str], 'must write down all the important oncept discussed in the review in a list']
    summary: Annotated[str, 'must write down a short summary of the review']
    sentiment: Annotated[str, 'must write down the sentiment of the review, either Positive or Negative']
    pros=Annotated[Optional[list[str]], 'write down the pros of the review, if there are any']
    cons=Annotated[Optional[list[str]], 'write down the cons of the review, if there are any']
    
structured_model = model.with_structured_output(Review)

prompt = """
Olm, by zoologist

I love olm, I tried a sample of it, not thinking I’d like it but just wanted 
to see. But I enjoy it so much I got a full bottle. It is so mineral-y and 
does smell like a cave, surrounded by stone. It’s not dank and dark though, 
it feels very clean and fresh and gives the feeling of crisp pristine water 
flowing through the cave. 

It also kind of reminds me of the smell of fall leaves just after a rainstorm. 
Like the leaves on the ground are a little wet but it hasn’t gotten gross yet, 
it just smells a little bit damp, a little earthy, but still fresh and not too 
musty. 

I don’t know if this scent is for everyone as it is very unique. But I enjoy 
it a lot.
"""

result = structured_model.invoke(prompt)

print(result)

