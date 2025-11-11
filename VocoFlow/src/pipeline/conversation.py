"""
实现完整的语音对话流程：
- ASR（语音识别）
- LLM（大语言模型对话）
- TTS（语音合成）
"""
import os
import sys
import logging
from datetime import datetime
from typing import Optional, Dict, Any

import numpy as np
import soundfile as sf

# 将 src 目录添加到 sys.path 中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.asr import FunASRModule
from src.llm import LLMInterface
from src.tts import EmotiVoiceTTS
# from .asr import FunASRModule
# from .llm import LLMInterface
# from .tts import EmotiVoiceTTS

logger = logging.getLogger(__name__)


class ConversationPipeline:
    """对话流程控制器"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.conversation_config = config.get("conversation", {})

        # 基础配置
        self.max_history = self.conversation_config.get("max_history", 10)
        self.save_audio = self.conversation_config.get("save_audio", True)
        self.audio_output_dir = self.conversation_config.get("audio_output_dir", "data/audio_output")
        self.audio_input_dir = self.conversation_config.get("audio_input_dir", "data/audio_input")

        os.makedirs(self.audio_output_dir, exist_ok=True)
        os.makedirs(self.audio_input_dir, exist_ok=True)

        # 初始化模块
        logger.info("正在初始化对话流程控制器...")
        self.asr = FunASRModule(config.get("asr", {}))
        self.llm = LLMInterface(config.get("llm", {}))
        self.tts = EmotiVoiceTTS(config.get("tts", {}))

        # TTS 相关默认参数（与 emotivoice_local.synthesize 对齐）
        tts_cfg = config.get("tts", {}) or {}
        self.tts_default_voice: str = str(tts_cfg.get("voice", "7556"))
        self.tts_default_emotion: str = str(tts_cfg.get("emotion", "友好的"))

        self.conversation_count = 0
        logger.info("对话流程控制器初始化完成")

    # -------------------- 文件输入 --------------------

    def process_audio_file(self, audio_path: str) -> Dict[str, Any]:
        result = {
            "success": False,
            "input_audio": audio_path,
            "asr_text": None,
            "llm_response": None,
            "tts_audio": None,            # ndarray(int16)
            "output_audio_path": None,
            "error": None,
        }
        try:
            logger.info(f"开始处理音频文件: {audio_path}")

            # 1) ASR
            try:
                logger.info("Step 1/3: ASR语音识别...")
                asr_text = self.asr.transcribe_file(audio_path)
                result["asr_text"] = asr_text
                logger.info(f"ASR识别结果: {asr_text}")
            except Exception as e:
                result["error"] = f"ASR识别失败: {e}"
                logger.exception(result["error"])
                return result

            # 2) LLM
            try:
                logger.info("Step 2/3: LLM生成回复...")
                llm_response = self.llm.chat(asr_text, use_history=True)
                result["llm_response"] = llm_response
                logger.info(f"LLM回复: {llm_response}")
                self.llm.trim_history(self.max_history)
            except Exception as e:
                result["error"] = f"LLM生成失败: {e}"
                logger.exception(result["error"])
                return result

            # 3) TTS（本地 EmotiVoiceTTS）
            try:
                logger.info("Step 3/3: TTS语音合成...")
                voice = self.tts_default_voice
                emotion = self.tts_default_emotion
                tts_audio = self.tts.synthesize(llm_response, voice=voice, emotion=emotion)  # int16 ndarray
                result["tts_audio"] = tts_audio

                if self.save_audio and tts_audio is not None:
                    output_path = self._save_output_wav(tts_audio, self.tts.sample_rate)
                    result["output_audio_path"] = output_path
                    logger.info(f"合成音频已保存: {output_path}")

            except Exception as e:
                result["error"] = f"TTS合成失败: {e}"
                logger.exception(result["error"])
                return result

            self.conversation_count += 1
            result["success"] = True
            logger.info(f"对话处理完成 (第{self.conversation_count}轮)")
            return result

        except Exception as e:
            logger.exception("处理对话时出错")
            return {"success": False, "error": str(e)}

    # -------------------- 内存数组输入 --------------------

    def process_audio_array(self, audio_array: np.ndarray, sample_rate: int = 16000) -> Dict[str, Any]:
        input_path: Optional[str] = None
        result = {
            "success": False,
            "input_audio": None,
            "asr_text": None,
            "llm_response": None,
            "tts_audio": None,
            "output_audio_path": None,
            "error": None,
        }
        try:
            if self.save_audio:
                input_path = self._save_input_wav(audio_array, sample_rate)
                logger.info(f"输入音频已保存: {input_path}")
            result["input_audio"] = input_path

            # 1) ASR
            try:
                logger.info("Step 1/3: ASR语音识别...")
                asr_text = self.asr.transcribe_array(audio_array, sample_rate)
                result["asr_text"] = asr_text
                logger.info(f"ASR识别结果: {asr_text}")
            except Exception as e:
                result["error"] = f"ASR识别失败: {e}"
                logger.exception(result["error"])
                return result

            # 2) LLM
            try:
                logger.info("Step 2/3: LLM生成回复...")
                llm_response = self.llm.chat(asr_text, use_history=True)
                result["llm_response"] = llm_response
                logger.info(f"LLM回复: {llm_response}")
                self.llm.trim_history(self.max_history)
            except Exception as e:
                result["error"] = f"LLM生成失败: {e}"
                logger.exception(result["error"])
                return result

            # 3) TTS（本地 EmotiVoiceTTS）
            try:
                logger.info("Step 3/3: TTS语音合成...")
                voice = self.tts_default_voice
                emotion = self.tts_default_emotion
                tts_audio = self.tts.synthesize(llm_response, voice=voice, emotion=emotion)  # int16 ndarray
                result["tts_audio"] = tts_audio

                if self.save_audio and tts_audio is not None:
                    output_path = self._save_output_wav(tts_audio, self.tts.sample_rate)
                    result["output_audio_path"] = output_path
                    logger.info(f"合成音频已保存: {output_path}")

            except Exception as e:
                result["error"] = f"TTS合成失败: {e}"
                logger.exception(result["error"])
                return result

            self.conversation_count += 1
            result["success"] = True
            logger.info(f"对话处理完成 (第{self.conversation_count}轮)")
            return result

        except Exception as e:
            logger.exception("处理对话时出错")
            return {"success": False, "error": str(e)}

    # -------------------- 其它控制 --------------------

    def reset_conversation(self):
        self.llm.clear_history()
        self.conversation_count = 0
        logger.info("对话状态已重置")

    def get_conversation_count(self) -> int:
        return self.conversation_count

    def get_history(self):
        return self.llm.get_history()

    def set_system_prompt(self, prompt: str):
        self.llm.set_system_prompt(prompt)

    def set_speaker(self, speaker_id: int):
        self.tts.set_speaker(speaker_id)

    def set_speed(self, speed: float):
        # 本地实现未提供 speed；如需支持，请在 EmotiVoiceTTS 中扩展
        logger.info(f"（提示）本地 EmotiVoiceTTS 暂未实现 set_speed，忽略 speed={speed}")

    # -------------------- I/O --------------------

    def _save_input_wav(self, audio: np.ndarray, sr: int) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.audio_input_dir, f"input_{ts}_{self.conversation_count:04d}.wav")
        sf.write(path, self._ensure_mono_int16(audio), int(sr))
        return path

    def _save_output_wav(self, audio_int: np.ndarray, sr: int) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.audio_output_dir, f"response_{ts}_{self.conversation_count:04d}.wav")
        sf.write(path, self._ensure_mono_int16(audio_int), int(sr))
        return path

    @staticmethod
    def _ensure_mono_int16(a: np.ndarray) -> np.ndarray:
        a = np.asarray(a)
        if a.ndim == 2:
            a = a.mean(axis=1)
        if a.dtype != np.int16:
            # 为什么：本地 TTS 返回 int16，这里兜底将 float / int32 安全裁剪
            a = np.clip(a, -32768, 32767).astype(np.int16)
        return a
