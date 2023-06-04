from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings

model_name = "sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
hf_embeddings_1 = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs)

openai_embedding = OpenAIEmbeddings(model="text-embedding-ada-002")


model_name = "GanymedeNil/text2vec-large-chinese"
hf_embeddings_2 = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs)


EMBEDDINGS_MAPPING = {"text-embedding-ada-002": openai_embedding,
                      "distilbert-dot-tas_b-b256-msmarco": hf_embeddings_1,
                      "text2vec-large-chinese": hf_embeddings_2}

def main():
    pass

if __name__ == "__main__":
    main()
