from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate, load_prompt

from dotenv import load_dotenv
import streamlit as st 

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")


st.header('Research assistant')


paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )

submit_button = st.button('Summarize')


# template = PromptTemplate( 
#   template= """Please summarize the research paper titled {paper_input} with the following specifications: 
#   Explanation style: {style_input}
#   Explanation length: {length_input}
#   1. Mathematical details:
#     - Include relevant mathematical equations if present in the paper
#     - Explain the mathematical concepts using simple, intuitive code snippets where applicable
#   2. Analogies:
#     - USe relatable analogies to simplify complex ideas.
#   If certain information is not available in the paper, repond with "Insuffient information" instead of guessing.
#   Ensure the summary is clear, accurate, and aligned with the provided style and length""",
#   input_variables={'paper_input', 'style_input', 'length_input'}
# )

template = load_prompt('template.json')

# prompt = template.invoke({
#   'paper_input': paper_input,
#   'style_input': style_input,
#   'length_input': length_input
# })

if(submit_button):
  chain = template | model
  result = chain.invoke({
    'paper_input': paper_input,
    'style_input': style_input,
    'length_input': length_input
  })
  # result = model.invoke(prompt)
  st.write(result.text)
  st.text('Tryna see the app')

