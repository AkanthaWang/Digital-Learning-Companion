import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import torch
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
import logging
import numpy as np
from conversation import ConversationPipeline

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 应用实例
app = FastAPI()

# 配置 ConversationPipeline
conversation_config = {
    "conversation": {
        "max_history": 10,
        "save_audio": True,
        "audio_output_dir": "data/audio_output",
        "audio_input_dir": "data/audio_input"
    },
    "asr": {
        "model_name": "paraformer-zh",
        "model_revision": "v2.0.4",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "batch_size": 1,
        "sample_rate": 16000,
    },
    "llm": {
        "provider": "deepseek",
        "system_prompt": "我是数字学伴。",
        "deepseek": {
            "api_key": "sk-77ef298879ca497aba694729e3328dec",  # 请填入有效的 API 密钥
            "base_url": "https://api.deepseek.com",
            "model": "deepseek-chat",
            "temperature": 0.7,
            "max_tokens": 2000,
            "top_p": 0.95
        }
    },
    "tts": {
        "voice": "7556",  # 设置语音
        "emotion": "友好的"
    }
}

# 初始化 ConversationPipeline
conversation_pipeline = ConversationPipeline(config=conversation_config)

class AudioRequest(BaseModel):
    audio_file: UploadFile

@app.post("/process-audio")
async def process_audio(request: AudioRequest):
    """
    接收音频文件，依次执行 ASR、LLM 和 TTS 流程
    """
    try:
        # 将上传的音频文件读取为字节流
        audio_content = await request.audio_file.read()
        
        # 将字节流转为 numpy 数组（假设音频是 WAV 格式，您可以根据实际情况调整）
        audio_array = np.frombuffer(audio_content, dtype=np.int16).astype(np.float32)
        audio_array = audio_array / 32768.0  # 归一化到 [-1, 1]
        
        # 调用 ConversationPipeline 处理音频
        result = conversation_pipeline.process_audio_array(audio_array)
        
        if result["success"]:
            return {
                "success": True,
                "asr_text": result["asr_text"],
                "llm_response": result["llm_response"],
                "output_audio_path": result["output_audio_path"]  # 返回保存的 TTS 音频文件路径
            }
        else:
            raise HTTPException(status_code=500, detail=f"处理对话失败: {result['error']}")
    
    except Exception as e:
        logger.error(f"处理音频失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"处理音频失败: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
