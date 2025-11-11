from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from typing import List, Dict, Optional
import logging
from llm_interface import LLMInterface

app = FastAPI()

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化 LLM 配置
llm_config = {
    "provider": "deepseek",  # 使用 DeepSeek 或 Qwen
    "system_prompt": "数字学伴。",
    "deepseek": {
        "api_key": "your_deepseek_api_key",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
        "temperature": 0.7,
        "max_tokens": 2000,
        "top_p": 0.95
    },
    "qwen": {
        "api_key": "your_qwen_api_key",
        "base_url": "https://dashscope.aliyuncs.com/api/v1",
        "model": "qwen-turbo",
        "temperature": 0.7,
        "max_tokens": 2000,
        "top_p": 0.8
    }
}

# 初始化 LLM 接口模块
llm_interface = LLMInterface(config=llm_config)


class ChatRequest(BaseModel):
    user_message: str
    use_history: Optional[bool] = True


@app.post("/llm/chat")
def chat(request: ChatRequest):
    """
    与 LLM 对话接口，接受用户消息并返回 LLM 回复。
    """
    try:
        # 调用 LLM 接口进行对话
        response = llm_interface.chat(request.user_message, use_history=request.use_history)
        return {"response": response}

    except Exception as e:
        logger.error(f"LLM对话失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"LLM对话失败: {str(e)}")


@app.post("/llm/clear-history")
def clear_history():
    """
    清空对话历史接口
    """
    try:
        llm_interface.clear_history()
        return {"message": "对话历史已清空"}

    except Exception as e:
        logger.error(f"清空历史失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"清空历史失败: {str(e)}")


@app.get("/llm/history")
def get_history():
    """
    获取当前对话历史
    """
    try:
        history = llm_interface.get_history()
        return {"history": history}

    except Exception as e:
        logger.error(f"获取历史失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取历史失败: {str(e)}")


@app.post("/llm/set-system-prompt")
def set_system_prompt(prompt: str = Body(...)):
    """
    设置 LLM 的系统提示词
    """
    try:
        llm_interface.set_system_prompt(prompt)
        return {"message": "系统提示词已更新"}

    except Exception as e:
        logger.error(f"设置系统提示词失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"设置系统提示词失败: {str(e)}")


@app.post("/llm/switch-provider")
def switch_provider(provider: str = Body(...)):
    """
    切换 LLM 提供商接口（支持 deepseek 和 qwen）
    """
    try:
        llm_interface.switch_provider(provider)
        return {"message": f"已切换到提供商: {provider}"}

    except Exception as e:
        logger.error(f"切换提供商失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"切换提供商失败: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
