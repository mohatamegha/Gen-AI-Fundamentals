from langchain_text_splitters import CharacterTextSplitter

text = "I am Megha, an MCA student from IGDTUW"

splitter = CharacterTextSplitter(
  chunk_size = 10,
  chunk_overlap = 0,
  separator=''
)

result = splitter.split_text(text)

print(result)