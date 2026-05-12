from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Literal, Optional
from pydantic import BaseModel, Field

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

class Review(BaseModel):
    key_themes: list[str] = Field(description='write down e key themes discussed in the review in a list')
    summary: str = Field(description='write down a short summary of the review')
    sentiment: Literal['Positive', 'Negative'] = Field(description='the sentiment of the review, either Positive or Negative')
    name: Optional[str] = Field(description='the name of the reviewer, if there is any')

structured_model = model.with_structured_output(Review, strict=True)

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

Review by Frangrantica User
"""

result = structured_model.invoke(prompt)

print(result)