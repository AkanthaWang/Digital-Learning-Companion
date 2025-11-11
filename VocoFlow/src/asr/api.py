from fastapi import FastAPI, HTTPException, File, UploadFile
import numpy as np
import logging
from typing import List, Union
from pydantic import BaseModel
import os
import torch
from funasr_module import FunASRModule

app = FastAPI()

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 加载 ASR 模型
asr_config = {
    'model_name': 'paraformer-zh',  # 选择合适的模型名称
    'model_revision': 'v2.0.4',  # 模型版本
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',  # 选择设备
    'batch_size': 1,  # 批处理大小
    'sample_rate': 16000,  # 音频采样率
}

# 初始化 ASR 模块
asr_module = FunASRModule(asr_config)


class ASRRequest(BaseModel):
    language: str = "zh"  # 语言，默认为中文
    hotword: str = ''  # 可选热词


@app.post("/load-asr-model")
def load_asr_model():
    """
    加载 ASR 模型接口
    """
    try:
        asr_module._load_model()
        return {"message": "ASR模型加载成功"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"加载模型失败: {str(e)}")


@app.post("/asr")
def asr(request: ASRRequest, file: UploadFile = File(...)):
    """
    语音识别接口，接收音频文件并返回识别结果
    """
    try:
        # 读取音频文件
        audio_content = file.file.read()

        # 将音频内容转为 numpy 数组
        audio_array = np.frombuffer(audio_content, dtype=np.int16).astype(np.float32)
        audio_array = audio_array / 32768.0  # 归一化到[-1, 1]

        # 识别音频
        text = asr_module.transcribe_array(audio_array)

        return {"transcription": text}

    except Exception as e:
        logger.error(f"ASR处理错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"识别失败: {str(e)}")


@app.get("/asr-info")
def get_asr_info():
    """
    获取 ASR 模型信息
    """
    return asr_module.get_model_info()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
