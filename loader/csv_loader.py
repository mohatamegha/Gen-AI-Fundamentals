from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path='Social_Network_Ads.csv')

docs = loader.load() #works with lazy load too

print(len(docs))
print(docs[1])