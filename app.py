import json
import os
import time
import uuid
from datetime import datetime

import gradio as gr
import openai
from huggingface_hub import HfApi
from langchain.document_loaders import PyPDFLoader, \
    UnstructuredPDFLoader, PyPDFium2Loader, PyMuPDFLoader, PDFPlumberLoader

from knowledge.faiss_handler import create_faiss_index_from_zip, load_faiss_index_from_zip
from knowledge.img_handler import process_image, add_markup
from llms.chatbot import OpenAIChatBot
from llms.embeddings import EMBEDDINGS_MAPPING
from utils import make_archive

UPLOAD_REPO_ID=os.getenv("UPLOAD_REPO_ID")
HF_TOKEN=os.getenv("HF_TOKEN")
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base == os.getenv("OPENAI_API_BASE")
hf_api = HfApi(token=HF_TOKEN)

ALL_PDF_LOADERS = [PyPDFLoader, UnstructuredPDFLoader, PyPDFium2Loader, PyMuPDFLoader, PDFPlumberLoader]
ALL_EMBEDDINGS = EMBEDDINGS_MAPPING.keys()
PDF_LOADER_MAPPING = {loader.__name__: loader for loader in ALL_PDF_LOADERS}


#######################################################################################################################
# Host multiple vector database for use
#######################################################################################################################
# todo: add this feature in the future



INSTRUCTIONS = '''# FAISS Chat: 和本地数据库聊天!

***2023-06-06更新:*** 
1. 支持读取图片格式的图表数据(目前支持JPG, PNG).
2. 在"总结图表(Demo)"的标签页里提供了这个模块的测试.
  
***2023-06-04更新:*** 
1. 支持更多的Embedding Model (目前支持[text-embedding-ada-002](https://openai.com/blog/new-and-improved-embedding-model), [text2vec-large-chinese](https://huggingface.co/GanymedeNil/text2vec-large-chinese), 和[distilbert-dot-tas_b-b256-msmarco](https://huggingface.co/sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco) )
2. 支持更多的文件格式(PDF, TXT, TEX, 和MD).
3. 所有生成的数据库都可以在[这个数据集](https://huggingface.co/datasets/shaocongma/shared-faiss-vdb)里访问了！如果不希望文件被上传，可以在高级设置里关闭. 
'''


def load_zip_as_db(file_from_gradio,
                   pdf_loader,
                   embedding_model,
                   chunk_size=300,
                   chunk_overlap=20,
                   upload_to_cloud=True):
    if chunk_size <= chunk_overlap:
        return "chunk_size小于chunk_overlap. 创建失败.", None, None
    if file_from_gradio is None:
        return "文件为空. 创建失败.", None, None
    pdf_loader = PDF_LOADER_MAPPING[pdf_loader]
    zip_file_path = file_from_gradio.name
    project_name = uuid.uuid4().hex
    db, project_name, db_meta = create_faiss_index_from_zip(zip_file_path, embeddings=embedding_model,
                                                   pdf_loader=pdf_loader, chunk_size=chunk_size,
                                                         chunk_overlap=chunk_overlap, project_name=project_name)
    index_name = project_name + ".zip"
    make_archive(project_name, index_name)
    date = datetime.today().strftime('%Y-%m-%d')
    if upload_to_cloud:
        hf_api.upload_file(path_or_fileobj=index_name,
                           path_in_repo=f"{date}/faiss_{index_name}.zip",
                           repo_id=UPLOAD_REPO_ID,
                           repo_type="dataset")
    return "成功创建知识库. 可以开始聊天了!", index_name, db, db_meta


def load_local_db(file_from_gradio):
    if file_from_gradio is None:
        return "文件为空. 创建失败.", None
    zip_file_path = file_from_gradio.name
    db = load_faiss_index_from_zip(zip_file_path)

    return "成功读取知识库. 可以开始聊天了!", db


def extract_image(image_path):
    from PIL import Image
    print("Image Path:", image_path)
    im = Image.open(image_path)
    table = process_image(im)
    print(f"Success in processing the image. Table: {table}")
    return table, add_markup(table)


def describe(image):
    table = add_markup(process_image(image))
    _INSTRUCTION = 'Read the table below to answer the following questions.'
    question = "Please refer to the above table, and write a summary of no less than 200 words based on it in Chinese, ensuring that your response is detailed and precise. "
    prompt_0shot = _INSTRUCTION + "\n" + add_markup(table) + "\n" + "Q: " + question + "\n" + "A:"

    messages = [{"role": "assistant", "content": prompt_0shot}]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.7,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    ret = response.choices[0].message['content']
    return ret


with gr.Blocks() as demo:
    local_db = gr.State(None)

    def get_augmented_message(message, local_db, query_count, preprocessing, meta):
        print(f"Receiving message: {message}")

        print("Detecting if the user need to read image from the local database...")
        # read the db_meta.json from the local file
        # read the images file list
        files = meta["files"]
        source_path = meta["source_path"]
        # with open(meta.name, "r", encoding="utf-8") as f:
        #     files = json.load(f)["files"]
        img_files = []
        for file in files:
            if os.path.splitext(file)[1] in [".png", ".jpg"]:
                img_files.append(file)

        # scan user's input to see if it contains images' name
        do_extract_image = False
        target_file = None
        for file in img_files:
            img = os.path.splitext(file)[0]
            if img in message:
                do_extract_image = True
                target_file = file
                break

        # extract image to tables
        image_info = ""
        if do_extract_image:
            print("The user needs to read image from the local database. Extract image ... ")
            target_file = os.path.join(source_path, target_file)
            _, image_info = extract_image(target_file)
        if len(image_info)>0:
            image_content = {"content": image_info, "source": os.path.basename(target_file)}
        else:
            image_content = None

        print("Querying references from the local database...")
        contents = []
        try:
            if query_count > 0:
                docs = local_db.similarity_search(message, k=query_count)
                for i in range(query_count):
                    # pre-processing each chunk
                    content = docs[i].page_content.replace('\n', ' ')
                    # pre-process meta data
                    contents.append(content)
        except:
            print("Failed to query from the local database. ")
        # generate augmented_message
        print("Success in querying references: {}".format(contents))
        if image_content is not None:
            augmented_message =  f"{image_content}\n\n---\n\n" + "\n\n---\n\n".join(contents) + "\n\n-----\n\n"
        else:
            augmented_message =  "\n\n---\n\n".join(contents) + "\n\n-----\n\n"
        return augmented_message + "\n\n" + f"'user_input': {message}"


    def respond(message, local_db, chat_history, meta, query_count=5, test_mode=False, response_delay=5, preprocessing=False):
        gpt_chatbot = OpenAIChatBot()
        print("Chat History: ", chat_history)
        print("Local DB: ", local_db is None)
        for chat in chat_history:
            gpt_chatbot.load_chat(chat)
        if local_db is None or query_count == 0:
            bot_message = gpt_chatbot(message)
            print(bot_message)
            print(message)
            chat_history.append((message, bot_message))
            return "", chat_history
        else:
            augmented_message = get_augmented_message(message, local_db, query_count, preprocessing, meta)
            bot_message = gpt_chatbot(augmented_message, original_message=message)
            print(message)
            print(augmented_message)
            print(bot_message)
            if test_mode:
                chat_history.append((augmented_message, bot_message))
            else:
                chat_history.append((message, bot_message))
            time.sleep(response_delay)  # sleep 5 seconds to avoid freq. wall.
            return "", chat_history

    with gr.Row():
        with gr.Column():
            gr.Markdown(INSTRUCTIONS)

            with gr.Row():
                with gr.Tab("从本地PDF文件创建知识库"):
                    zip_file = gr.File(file_types=[".zip"], label="本地PDF文件(.zip)")
                    create_db = gr.Button("创建知识库", variant="primary")
                    with gr.Accordion("高级设置", open=False):
                        embedding_selector = gr.Dropdown(ALL_EMBEDDINGS,
                                                         value="distilbert-dot-tas_b-b256-msmarco",
                                                         label="Embedding Models")
                        pdf_loader_selector = gr.Dropdown([loader.__name__ for loader in ALL_PDF_LOADERS],
                                                          value=PyPDFLoader.__name__, label="PDF Loader")
                        chunk_size_slider = gr.Slider(minimum=50, maximum=2000, step=50, value=500,
                                                      label="Chunk size (tokens)")
                        chunk_overlap_slider = gr.Slider(minimum=0, maximum=500, step=1, value=50,
                                                         label="Chunk overlap (tokens)")
                        save_to_cloud_checkbox = gr.Checkbox(value=False, label="把数据库上传到云端")


                    file_dp_output = gr.File(file_types=[".zip"], label="(输出)知识库文件(.zip)")
                with gr.Tab("读取本地知识库文件"):
                    file_local = gr.File(file_types=[".zip"], label="本地知识库文件(.zip)")
                    load_db = gr.Button("读取已创建知识库", variant="primary")

                with gr.Tab("总结图表(Demo)"):
                    gr.Markdown(r"代码来源于: https://huggingface.co/spaces/fl399/deplot_plus_llm")
                    input_image = gr.Image(label="Input Image", type="pil", interactive=True)
                    extract = gr.Button("总结", variant="primary")

                    output_text = gr.Textbox(lines=8, label="Output")




        with gr.Column():
            status = gr.Textbox(label="用来显示程序运行状态的Textbox")
            chatbot = gr.Chatbot()

            msg = gr.Textbox()
            submit = gr.Button("Submit", variant="primary")
            with gr.Accordion("高级设置", open=False):
                json_output = gr.JSON()
                with gr.Row():
                    query_count_slider = gr.Slider(minimum=0, maximum=10, step=1, value=3,
                                                  label="Query counts")
                    test_mode_checkbox = gr.Checkbox(label="Test mode")


    # def load_pdf_as_db(file_from_gradio,
    #                    pdf_loader,
    #                    embedding_model,
    #                    chunk_size=300,
    #                    chunk_overlap=20,
    #                    upload_to_cloud=True):
    msg.submit(respond, [msg, local_db, chatbot, json_output, query_count_slider, test_mode_checkbox], [msg, chatbot])
    submit.click(respond, [msg, local_db, chatbot, json_output, query_count_slider, test_mode_checkbox], [msg, chatbot])

    create_db.click(load_zip_as_db, [zip_file, pdf_loader_selector, embedding_selector, chunk_size_slider, chunk_overlap_slider, save_to_cloud_checkbox],
                    [status, file_dp_output, local_db, json_output])
    load_db.click(load_local_db, [file_local], [status, local_db])

    extract.click(describe, [input_image], [output_text])

demo.launch(show_api=False)
