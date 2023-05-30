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
from llms.chatbot import OpenAIChatBot
from llms.preprocessing import PreprocessingBot
from utils import make_archive

UPLOAD_REPO_ID=os.getenv("UPLOAD_REPO_ID")
HF_TOKEN=os.getenv("HF_TOKEN")
openai.api_key = os.getenv("OPENAI_API_KEY")
hf_api = HfApi(token=HF_TOKEN)

gpt_chatbot = OpenAIChatBot()
preprocessing_bot = PreprocessingBot()

LOCAL_DP = None
ALL_PDF_LOADERS = [PyPDFLoader, UnstructuredPDFLoader, PyPDFium2Loader, PyMuPDFLoader, PDFPlumberLoader]
PDF_LOADER_MAPPING = {loader.__name__: loader for loader in ALL_PDF_LOADERS}

INSTRUCTIONS = '''# FAISS Chat: 和本地数据库聊天!'''


def get_augmented_message(message, local_db, query_count, preprocessing):
    print(f"Receiving message: {message}")
    print("Querying references from the local database...")
    docs = local_db.similarity_search(message, k=query_count)
    contents = []
    for i in range(query_count):
        # pre-processing each chunk
        content = docs[i].page_content.replace('\n', ' ')
        # pre-process meta data
        contents.append(content)
    # generate augmented_message
    print("Success in querying references: {}".format(contents))
    if preprocessing:
        print("Pre-processing ...")
        try:
            augmented_message = preprocessing_bot("\n\n---\n\n".join(contents) + "\n\n-----\n\n")
            print("Success in pre-processing. ")
            try:
                msg = json.loads(augmented_message)
                msg['user_input'] = message
                return str(msg)
            except:
                return augmented_message + "\n\n" + f"{{'user_input': {message}}}"
        except Exception as e:
            print(f"Failed in pre-processing the documents: {e}. Return the raw input.")
            augmented_message = f"{{'user_input': {message}}}"
            return augmented_message + "\n\n" + message
    else:
        augmented_message = "\n\n---\n\n".join(contents) + "\n\n-----\n\n"
        return augmented_message + "\n\n" + f"'user_input': {message}"




def respond(message, chat_history, query_count=5, test_mode=False, response_delay=5, preprocessing=False):
    if LOCAL_DP is None or query_count==0:
        bot_message = gpt_chatbot(message)
        chat_history.append((message, bot_message))
        return "", chat_history
    else:
        augmented_message = get_augmented_message(message, LOCAL_DP, query_count, preprocessing)
        bot_message = gpt_chatbot(augmented_message, original_message=message)
        if test_mode:
            chat_history.append((augmented_message, bot_message))
        else:
            chat_history.append((message, bot_message))
        time.sleep(response_delay)  # sleep 5 seconds to avoid freq. wall.
        return "", chat_history


def load_pdf_as_db(file_from_gradio,
                   pdf_loader,
                   chunk_size=300,
                   chunk_overlap=20,
                   upload_to_cloud=True):
    if file_from_gradio is None:
        return "文件为空. 创建失败.", None
    global LOCAL_DP
    pdf_loader = PDF_LOADER_MAPPING[pdf_loader]
    zip_file_path = file_from_gradio.name
    LOCAL_DP, project_name = create_faiss_index_from_zip(zip_file_path, pdf_loader=pdf_loader, chunk_size=chunk_size,
                                                         chunk_overlap=chunk_overlap, project_name=str(uuid.uuid4()))
    index_name = str(uuid.uuid4()) + ".zip"
    make_archive(project_name, index_name)
    date = datetime.today().strftime('%Y-%m-%d')
    if upload_to_cloud:
        hf_api.upload_file(path_or_fileobj=index_name,
                           path_in_repo=f"{date}/faiss_{index_name}.zip",
                           repo_id=UPLOAD_REPO_ID,
                           repo_type="dataset")
    return "成功创建知识库. 可以开始聊天了!", index_name


def load_local_db(file_from_gradio):
    if file_from_gradio is None:
        return "文件为空. 创建失败.", None

    global LOCAL_DP
    zip_file_path = file_from_gradio.name
    LOCAL_DP = load_faiss_index_from_zip(zip_file_path)

    return "成功读取知识库. 可以开始聊天了!"


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            gr.Markdown(INSTRUCTIONS)

            with gr.Row():
                with gr.Tab("从本地PDF文件创建知识库"):
                    file_pdf = gr.File(file_types=[".zip"], label="本地PDF文件(.zip)")
                    create_db = gr.Button("创建知识库", variant="primary")
                    with gr.Accordion("高级设置", open=False):
                        pdf_loader_selector = gr.Dropdown([loader.__name__ for loader in ALL_PDF_LOADERS],
                                                          value=PyPDFLoader.__name__, label="PDF Loader")
                        chunk_size_slider = gr.Slider(minimum=50, maximum=500, step=50, value=300,
                                                      label="Chunk size (tokens)")
                        chunk_overlap_slider = gr.Slider(minimum=0, maximum=100, step=1, value=20,
                                                         label="Chunk overlap (tokens)")
                    file_dp_output = gr.File(file_types=[".zip"], label="(输出)知识库文件(.zip)")
                with gr.Tab("读取本地知识库文件"):
                    file_local = gr.File(file_types=[".zip"], label="本地知识库文件(.zip)")
                    load_db = gr.Button("读取已创建知识库", variant="primary")

        with gr.Column():
            status = gr.Textbox(label="用来显示程序运行状态的Textbox")
            chatbot = gr.Chatbot()

            msg = gr.Textbox()
            clear = gr.Button("Clear")
            submit = gr.Button("Submit", variant="primary")
            with gr.Accordion("高级设置", open=False):
                with gr.Row():
                    query_count_slider = gr.Slider(minimum=0, maximum=10, step=1, value=3,
                                                  label="Query counts")
                    test_mode_checkbox = gr.Checkbox(label="Test mode")

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)
    submit.click(respond, [msg, chatbot, query_count_slider, test_mode_checkbox], [msg, chatbot])

    create_db.click(load_pdf_as_db, [file_pdf, pdf_loader_selector, chunk_size_slider, chunk_overlap_slider],
                    [status, file_dp_output])
    load_db.click(load_local_db, [file_local], [status])

demo.launch()
