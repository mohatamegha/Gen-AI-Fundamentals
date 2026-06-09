from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt1 = PromptTemplate(
  template='Generate a detailed report on {topic}',
  input_variables = ['topic']
) 

model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")
parser = StrOutputParser()

prompt2 = PromptTemplate(
  template='Generate a summary of the report: {text}',
  input_variables = ['text']
) 

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({"topic": "artificial intelligence"})

print(result)

chain.get_graph().print_ascii() # to visualize the graph