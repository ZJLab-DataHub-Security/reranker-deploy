import os
import logging
import json
from pydantic import BaseModel
import uvicorn as uvicorn
from fastapi import FastAPI, status
from FlagEmbedding import FlagReranker
import sys
import time
from datetime import datetime
from fastapi.responses import JSONResponse, Response
import asyncio
import aiohttp

model_dir = "/nas/zhangjinxin/reranker_service/bge-reranker-v2-m3"

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

app = FastAPI()
reranker = FlagReranker(model_dir, use_fp16=True)

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
#logging.basicConfig(filename='rerank-zj.log', level=logging.DEBUG, format=LOG_FORMAT)
#logging.FileHandler(filename='rerank-zj.log', encoding="utf-8")
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, handlers=[
    logging.FileHandler(filename='rerank-zj.log', encoding="utf-8"),
    logging.StreamHandler()
])

class Queries(BaseModel):
    qp_pairs: list = None

@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)

# post请求带参数数据
@app.post('/query')
def insert(data: Queries):
    # logging.info("输入语句：{0}".format(query.queries))
    # 获取当前时间
    ft1 = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    st1 = time.time()
    session_id = getattr(data, 'session_id', "")
    batch = len(data.qp_pairs)
    pred_scores = reranker.compute_score(data.qp_pairs)
    st2 = time.time()
    st = (st2 - st1)*1000
    pid = os.getpid()
    # 获取当前时间
    ft2 = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    logging.info("请求的batch大小是%d", batch)
    logging.info("请求进来的时间是%s, 结果返回时间是%s", ft1, ft2)
    logging.info("请求的session id是%s", session_id)
    logging.info("当前进程是%d,推理耗时是%d毫秒", pid, st)
    return {"pred_scores": json.dumps(pred_scores)}

# 心跳检测逻辑
async def heartbeat():
    while True:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("http://127.0.0.1:8566/health") as response:
                    if response.status == 200:
                        logging.info("心跳检测成功")
                    else:
                        logging.error("心跳检测失败，状态码：%d", response.status)
        except Exception as e:
            logging.error("心跳检测失败，错误：%s", str(e))
        await asyncio.sleep(5)

# 启动心跳检测
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(heartbeat())

if __name__ == '__main__':
    uvicorn.run(app=app, host="0.0.0.0", port=8566)
    # 注意：如果供其他电脑调用需要改为 "0.0.0.0"，另外需要关闭电脑的防火墙

