from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")


class FeedBack (BaseModel):
  #Field allows you to add metadata and validation rules to a field.
  #Literal means only these values are allowed:
  sentiment : Literal['positive', 'negative'] = Field(description='The sentiment of the feedback.')

parser2 = PydanticOutputParser(pydantic_object=FeedBack)

# print("Format instructions",parser2.get_format_instructions())

prompt = PromptTemplate(
  template='Classify the sentiment of the following feedback text into positive or negative. \n {feedback} \n {format_instruction}',
  input_variables = ['feedback'],
  partial_variables = {'format_instruction':parser2.get_format_instructions()}
)

# print('Prompt : ', prompt)

parser = StrOutputParser()

classifierChain = prompt | model | parser2

# text = classifierChain.invoke(prompt.invoke({'feedback':'The pizza was good, but the environment was boring'}))

# print(text)

prompt2 = PromptTemplate(
  template = 'Write an appropriate one line response to this positive feedback that can be directly sent to user \n {feedback}',
  input_variables=['feedback']
)

prompt3 = PromptTemplate(
  template = 'Write an appropriate one line response to this negative feedback that can be directly sent to user \n {feedback}',
  input_variables=['feedback']
)

branchChain = RunnableBranch(
  (lambda x : x["sentiment"].sentiment  =='positive', prompt2 | model | parser),
  (lambda x : x["sentiment"].sentiment  =='negative', prompt3 | model | parser),
  RunnableLambda(lambda x : "could not find sentiment")
  # (condition, chain),
  # (condition, chain),
  # default
) 

chain = (
  RunnablePassthrough.assign(
    sentiment=classifierChain
  )
  | branchChain
)
print(chain.invoke({'feedback':'The pizza was good, but the environment was boring'}))