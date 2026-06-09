# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# from dotenv import load_dotenv

# load_dotenv()

# llm=HuggingFaceEndpoint(
#   repo_id="TinyLlama/TinyLlama-1.1B-Chat-v0.2",
#   task="text-generation",
#   huggingfacehub_api_token=os.environ["HF_TOKEN"],
# )
# model = ChatHuggingFace(llm=llm)

# response = model.invoke("What is the capital of India?")

# print(response)

# import os
# from huggingface_hub import InferenceClient

# from dotenv import load_dotenv

# load_dotenv()


# print(os.getenv("HF_TOKEN"))

# client = InferenceClient(
#     provider="featherless-ai",
#     api_key=os.environ["HF_TOKEN"],
# )

# result = client.text_generation(
#     "User: What is the capital of India?\nAssistant:",
#     model="TinyLlama/TinyLlama-1.1B-Chat-v0.2",
#     max_new_tokens=50,
# )

# print(result)

# import os
# from huggingface_hub import InferenceClient

# from dotenv import load_dotenv

# load_dotenv()

# client = InferenceClient(
#     provider="featherless-ai",
#     api_key=os.environ["HF_TOKEN"],
# )

# result = client.text_generation(
#     """<|system|>
# You are a helpful assistant.
# <|user|>
# What is the capital of India?
# <|assistant|>
# """,
#     model="TinyLlama/TinyLlama-1.1B-Chat-v0.2",
#     max_new_tokens=50,
# )

# print(result)

# import os
# from dotenv import load_dotenv
# from huggingface_hub import InferenceClient

# load_dotenv()

# client = InferenceClient(
#     provider="featherless-ai",
#     api_key=os.getenv("HF_TOKEN")
# )

# result = client.text_generation(
#     prompt="Question: What is the capital of India?\nAnswer:",
#     model="Qwen/Qwen2.5-7B-Instruct",
#     max_new_tokens=50,
#     temperature=0.1,
# )

# print(result)

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of India")

print(result.content)