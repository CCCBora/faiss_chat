---
title: Faiss Chat
emoji: 🐠
colorFrom: indigo
colorTo: purple
sdk: gradio 
sdk_version: 3.32.0
app_file: app.py
pinned: false
license: mit
---

# FAISS Chat: Chat with FAISS database

Webui版本的Langchain-Chat. 目前支持两个功能:
* 将本地PDF和TXT文件打包上传, 构建FAISS向量数据库. 
* 直接上传本地的FAISS向量数据库. 


## 更新日志

* 2023-06-04: 
  * 支持更多文件格式 (目前支持PDF, TXT, MD, TEX)
  * 支持更多Embedding Models (目前支持[text-embedding-ada-002](https://openai.com/blog/new-and-improved-embedding-model), [text2vec-large-chinese](https://huggingface.co/GanymedeNil/text2vec-large-chinese), 和[distilbert-dot-tas_b-b256-msmarco](https://huggingface.co/sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco) )
  * 优化本地知识库文件结构. 

## 体验地址
[Huggingface Space](https://huggingface.co/spaces/shaocongma/faiss_chat)

