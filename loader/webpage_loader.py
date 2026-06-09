from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")

loader = WebBaseLoader('https://medium.com/@mutahirmanzoor1/getting-started-with-spring-boot-a-beginners-guide-9dcd38c2cd8c')
#we can pass multiple urls too
docs = loader.load()

text = docs[0].page_content

# print(text)
parser = StrOutputParser()

prompt = PromptTemplate(
  template='Generate a summary of {text}',
  input_variables = ['text']
) 

chain = prompt | model | parser

answer = chain.invoke({'text':text})
print(answer)
# print(len(docs))

# print(text)