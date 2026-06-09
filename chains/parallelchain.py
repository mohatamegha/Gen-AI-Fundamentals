from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")

prompt = PromptTemplate(
  template='Generate a detailed report on {topic}',
  input_variables = ['topic']
) 

text = model.invoke(prompt.invoke({'topic':'Gen AI'}))

prompt_notes = PromptTemplate(
  template='Generate notes on the following: {text}',
  input_variables=['text']
)

prompt_quiz = PromptTemplate(
  template='Generate quiz of 3 questions on the following: {text}',
  input_variables=['text']
)

prompt_merge = PromptTemplate(
  template='Merge the following notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}',
  input_variables=['notes', 'quiz']
)

model_topic = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")
model_notes = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")
model_quiz = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")

parser = StrOutputParser()

parallelChain = RunnableParallel({
  "notes" : prompt_notes | model_notes | parser,
  "quiz" : prompt_quiz | model_quiz | parser
})

mergeChain = prompt_merge | model | parser

chain = parallelChain | mergeChain

result = chain.invoke({'text' : text})

print(result)

chain.get_graph().print_ascii() # to visualize the graph