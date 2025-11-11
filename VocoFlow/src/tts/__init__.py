"""
EmotiVoice TTS 模块入口
======================
提供统一的文本转语音接口。
"""

from .emotivoice_local import EmotiVoiceTTS
from .emotivoice_client import EmotiVoiceClient

__all__ = ["EmotiVoiceTTS", "EmotiVoiceClient"]
