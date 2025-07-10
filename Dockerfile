FROM pytorch:23.12-py3

WORKDIR /workspace

RUN pip install uvicorn -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install fastapi -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install FlagEmbedding -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install transformers sentencepiece -i https://pypi.tuna.tsinghua.edu.cn/simple
WORKDIR /workspace/app
COPY reranker_fast_api.py /workspace/app/reranker_fast_api.py

RUN chmod 755 reranker_fast_api.py

CMD ["python3","reranker_fast_api.py"] 
