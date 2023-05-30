import json
import uuid
from langchain.vectorstores import FAISS
import os
from tqdm.auto import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
import zipfile
import pickle

tokenizer_name = tiktoken.encoding_for_model('gpt-4')
tokenizer = tiktoken.get_encoding(tokenizer_name.name)
EMBED_MODEL = "text-embedding-ada-002"
EMBED_DIM = 1536
METRIC = 'cosine'


#######################################################################################################################
# PDF Files handler
#######################################################################################################################
def list_pdf_files(directory):
    # List all files in the directory
    files_in_directory = os.listdir(directory)
    # Filter the list to only include PDF files
    pdf_files = [file for file in files_in_directory if file.endswith(".pdf")]
    return pdf_files


def tiktoken_len(text):
    # evaluate how many tokens for the given text
    tokens = tokenizer.encode(text, disallowed_special=())
    return len(tokens)


def get_chunks(docs, chunk_size=500, chunk_overlap=20, length_function=tiktoken_len):
    # docs should be the output of `loader.load()`
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                   chunk_overlap=chunk_overlap,
                                                   length_function=length_function,
                                                   separators=["\n\n", "\n", " ", ""])
    chunks = []
    for idx, page in enumerate(tqdm(docs)):
        source = page.metadata.get('source')
        pdf_name = page.metadata.get('file_name')
        content = page.page_content
        if len(content) > 50:
            texts = text_splitter.split_text(content)
            chunks.extend([str({'content': texts[i], 'chunk': i, 'source': source, 'pdf_name': pdf_name}) for i in
                           range(len(texts))])
    return chunks


#######################################################################################################################
# Create FAISS object
#######################################################################################################################
def create_faiss_index_from_zip(path_to_zip_file, embeddings=None, pdf_loader=None,
                                chunk_size=500, chunk_overlap=20, project_name="faiss_index", project_description=""):
    # initialize the file structure
    # structure: project_name
    #               - path-to-extract-pdf-files
    #                   - pdf data
    #                   - embeddings
    #                   - faiss_index
    if embeddings is None:
        from langchain.embeddings.openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    db_meta = {"project_name": project_name, "project_description": project_description,
               "pdf_loader": pdf_loader.__name__, "chunk_size": chunk_size, "chunk_overlap": chunk_overlap,
               "dim": EMBED_DIM, "model": EMBED_MODEL, "metric": METRIC}

    current_directory = os.getcwd()
    project_name = os.path.join(current_directory, project_name)
    data_id = str(uuid.uuid1())
    working_directory = os.path.join(project_name, data_id)
    if not os.path.exists(project_name):
        os.makedirs(project_name)
    pdf_data = os.path.join(working_directory, "pdf_data")
    embeddings_data = os.path.join(working_directory, "embeddings")
    index_data = os.path.join(working_directory, "faiss_index")
    os.makedirs(working_directory)  # uuid is always unique; no need to check existence
    os.makedirs(pdf_data)
    os.makedirs(embeddings_data)
    os.makedirs(index_data)
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(pdf_data)

    # enumerate all pdf files in `extract_to`
    all_pdf_files = list_pdf_files(pdf_data)
    db_meta["pdf_list"] = all_pdf_files
    with open(os.path.join(working_directory, "db_meta.json"), "w", encoding="utf-8") as f:
        json.dump(db_meta, f)
    for idx, pdf_file in enumerate(all_pdf_files):
        # load meta data corresponds to the given pdf_file
        filename = os.path.splitext(pdf_file)[0] + ".json"
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                meta_data = json.load(f)
        else:
            meta_data = {}
            meta_data["file_name"] = pdf_file

        pdf_path = os.path.join(pdf_data, pdf_file)
        loader = pdf_loader(pdf_path)
        pages = loader.load()
        for page in pages:
            page.metadata = {**page.metadata, **meta_data}

        # split pdf files into chunks and evaluate its embeddings; save all results into embeddings
        chunks = get_chunks(pages, chunk_size, chunk_overlap)
        text_embeddings = embeddings.embed_documents(chunks)
        text_embedding_pairs = list(zip(chunks, text_embeddings))
        embeddings_save_to = os.path.join(embeddings_data, 'text_embedding_pairs.pickle')
        with open(embeddings_save_to, 'wb') as handle:
            pickle.dump(text_embedding_pairs, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # create FAISS db
        if idx == 0:
            # create a new index
            db = FAISS.from_embeddings(text_embedding_pairs, embeddings)
        else:
            # add to existing index
            db.add_embeddings(text_embedding_pairs)

    db.save_local(index_data)
    print(db_meta)
    print("Success!")
    return db, project_name


def load_faiss_index_from_zip(path_to_zip_file, embeddings=None):
    if embeddings is None:
        from langchain.embeddings.openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    base_name = os.path.basename(path_to_zip_file)
    path_to_extract = os.path.join(os.getcwd(), os.path.splitext(base_name)[0])
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(path_to_extract)

    path_to_extract = os.path.join(path_to_extract, os.listdir(path_to_extract)[0])
    # list all directories
    directories = [os.path.join(path_to_extract, d) for d in os.listdir(path_to_extract) if
                   os.path.isdir(os.path.join(path_to_extract, d))]
    index_path = os.path.join(directories[0], "faiss_index")
    db = FAISS.load_local(index_path, embeddings)
    for idx, folder in enumerate(directories):
        if idx != 0:
            index_path = os.path.join(folder, "faiss_index")
            new_db = FAISS.load_local(index_path, embeddings)
            db.merge_from(new_db)
    return db


if __name__ == "__main__":
    from langchain.document_loaders import PyPDFLoader
    from langchain.embeddings.openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    create_faiss_index_from_zip(path_to_zip_file="pdf_data/document.zip", pdf_loader=PyPDFLoader, embeddings=embeddings)
    #
    # embeddings = OpenAIEmbeddings( model="text-embedding-ada-002" )
    # # docs = get_docs("pdf_data/ml_books", PyPDFLoader)
    # # db = create_faiss_db(docs, embeddings, index_name="ml_books_test")
    # db = load_faiss_db(embeddings, "ml_books_test")
    # # query = "Which Markov chain could be sub-geometrically ergodic?"
    # # docs = db.similarity_search(query)
    # # print(docs[0].page_content)
    # #
    # query = "What is hidden Markov models?"
    # count = 5
    # docs = db.similarity_search(query, k=count)
    # prompts = ["Reference {}: {}\n\n".format(i, docs[i].page_content.replace('\n', ' ')) for i in range(count)]
    # prompts = "".join(prompts)
    # with open("tmp.txt", "w", encoding="utf-8") as f:
    #     f.write(prompts)
