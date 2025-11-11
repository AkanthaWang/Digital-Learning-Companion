"""
EmotiVoice 本地调用包装器
直接使用下载的 EmotiVoice 代码进行语音合成
"""

import os
import sys
import logging
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

# 添加 EmotiVoice 到路径
ROOT = Path(__file__).resolve().parents[2]
EMOTIVOICE_ROOT = ROOT / "models" / "EmotiVoice"
sys.path.insert(0, str(EMOTIVOICE_ROOT))

logger = logging.getLogger(__name__)


class EmotiVoiceTTS:
    """EmotiVoice 本地调用器"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化本地 EmotiVoice

        Args:
            config: TTS 配置
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 设置工作目录到 EmotiVoice 根目录
        os.chdir(EMOTIVOICE_ROOT)

        # 导入 EmotiVoice 模块
        self._import_emotivoice_modules()

        # 初始化模型
        self._init_models()

        logger.info(f"EmotiVoice 本地初始化完成，设备: {self.device}")

    def _import_emotivoice_modules(self):
        """导入 EmotiVoice 相关模块"""
        try:
            # 导入前端处理
            from frontend import g2p_cn_en, read_lexicon, G2p

            # 导入模型
            from models.prompt_tts_modified.jets import JETSGenerator
            from models.prompt_tts_modified.simbert import StyleEncoder

            # 导入配置
            from config.joint.config import Config

            # 保存引用
            self.g2p_cn_en = g2p_cn_en
            self.read_lexicon = read_lexicon
            self.G2p = G2p
            self.JETSGenerator = JETSGenerator
            self.StyleEncoder = StyleEncoder
            self.Config = Config

            logger.info("EmotiVoice 模块导入成功")

        except ImportError as e:
            logger.error(f"导入 EmotiVoice 模块失败: {e}")
            raise

    def _init_models(self):
        """初始化模型"""
        try:
            # 加载配置
            self.emotivoice_config = self.Config()

            # 初始化前端
            lexicon_path = EMOTIVOICE_ROOT / "lexicon" / "librispeech-lexicon.txt"
            self.lexicon = self.read_lexicon(str(lexicon_path))
            self.g2p = self.G2p()

            # 加载模型
            self.style_encoder, self.generator, self.tokenizer, self.token2id, self.speaker2id = self._load_models()

            logger.info("EmotiVoice 模型加载完成")

        except Exception as e:
            logger.error(f"初始化 EmotiVoice 模型失败: {e}")
            raise

    def _load_models(self):
        """加载所有必要的模型"""
        import glob
        from transformers import AutoTokenizer
        from yacs import config as CONFIG

        # 扫描检查点
        def scan_checkpoint(cp_dir, prefix, c=8):
            pattern = os.path.join(cp_dir, prefix + '?' * c)
            cp_list = glob.glob(pattern)
            if len(cp_list) == 0:
                return None
            return sorted(cp_list)[-1]

        # 模型路径
        am_checkpoint_path = scan_checkpoint(
            f'{self.emotivoice_config.output_directory}/prompt_tts_open_source_joint/ckpt', 'g_'
        )
        style_encoder_checkpoint_path = scan_checkpoint(
            f'{self.emotivoice_config.output_directory}/style_encoder/ckpt', 'checkpoint_', 6
        )

        # 加载模型配置
        with open(self.emotivoice_config.model_config_path, 'r') as fin:
            conf = CONFIG.load_cfg(fin)

        conf.n_vocab = self.emotivoice_config.n_symbols
        conf.n_speaker = self.emotivoice_config.speaker_n_labels

        # 初始化模型
        style_encoder = self.StyleEncoder(self.emotivoice_config)
        generator = self.JETSGenerator(conf).to(self.device)

        # 加载权重
        if style_encoder_checkpoint_path and os.path.exists(style_encoder_checkpoint_path):
            model_CKPT = torch.load(style_encoder_checkpoint_path, map_location=self.device)
            model_ckpt = {key[7:]: value for key, value in model_CKPT['model'].items()}
            style_encoder.load_state_dict(model_ckpt, strict=False)

        if am_checkpoint_path and os.path.exists(am_checkpoint_path):
            model_CKPT = torch.load(am_checkpoint_path, map_location=self.device)
            generator.load_state_dict(model_CKPT['generator'])

        generator.eval()

        # 加载 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.emotivoice_config.bert_path)

        # 加载 token 和说话人映射
        with open(self.emotivoice_config.token_list_path, 'r') as f:
            token2id = {t.strip(): idx for idx, t, in enumerate(f.readlines())}

        with open(self.emotivoice_config.speaker2id_path, encoding='utf-8') as f:
            speaker2id = {t.strip(): idx for idx, t in enumerate(f.readlines())}

        return style_encoder, generator, tokenizer, token2id, speaker2id

    def _get_style_embedding(self, prompt, tokenizer, style_encoder):
        """获取样式嵌入"""
        prompt = tokenizer([prompt], return_tensors="pt")
        input_ids = prompt["input_ids"]
        token_type_ids = prompt["token_type_ids"]
        attention_mask = prompt["attention_mask"]
        with torch.no_grad():
            output = style_encoder(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask
            )
        return output["pooled_output"].squeeze()

    def synthesize(self, text: str, voice: str = None, emotion: str = None) -> np.ndarray:
        """
        合成语音

        Args:
            text: 要合成的文本
            voice: 说话人ID (默认使用配置中的第一个)
            emotion: 情感提示

        Returns:
            音频数据数组
        """
        try:
            # 使用默认说话人
            if voice is None:
                voice = list(self.speaker2id.keys())[0] if self.speaker2id else "7556"

            # 使用默认情感
            if emotion is None:
                emotion = "友好的"

            logger.info(f"开始合成: '{text}' | 说话人: {voice} | 情感: {emotion}")

            # 文本预处理
            processed_text = self.g2p_cn_en(text, self.g2p, self.lexicon)

            # 获取样式嵌入
            style_embedding = self._get_style_embedding(emotion, self.tokenizer, self.style_encoder)
            content_embedding = self._get_style_embedding(text, self.tokenizer, self.style_encoder)

            # 说话人ID
            speaker_id = self.speaker2id.get(voice, 0)

            # 文本转token
            text_int = [self.token2id[ph] for ph in processed_text.split() if ph in self.token2id]

            # 准备输入
            sequence = torch.from_numpy(np.array(text_int)).to(self.device).long().unsqueeze(0)
            sequence_len = torch.from_numpy(np.array([len(text_int)])).to(self.device)
            style_embedding = style_embedding.to(self.device).unsqueeze(0)
            content_embedding = content_embedding.to(self.device).unsqueeze(0)
            speaker = torch.from_numpy(np.array([speaker_id])).to(self.device)

            # 推理
            with torch.no_grad():
                infer_output = self.generator(
                    inputs_ling=sequence,
                    inputs_style_embedding=style_embedding,
                    input_lengths=sequence_len,
                    inputs_content_embedding=content_embedding,
                    inputs_speaker=speaker,
                    alpha=1.0
                )

            # 处理输出
            audio = infer_output["wav_predictions"].squeeze() * 32768.0  # MAX_WAV_VALUE
            audio_data = audio.cpu().numpy().astype('int16')

            logger.info(f"合成成功，音频长度: {len(audio_data)}")
            if len(audio_data) == 0:
                logger.error("生成的音频数据为空！")
            return audio_data

        except Exception as e:
            logger.error(f"语音合成失败: {e}")
            raise

    def synthesize_to_file(self, text: str, output_path: str, voice: str = None, emotion: str = None) -> bool:
        """合成语音并保存到文件"""
        try:
            audio_data = self.synthesize(text, voice, emotion)

            import soundfile as sf
            sf.write(output_path, audio_data, self.emotivoice_config.sampling_rate)
            logger.info(f"音频已保存: {output_path}")
            return True

        except Exception as e:
            logger.error(f"保存音频失败: {e}")
            return False

    @property
    def sample_rate(self) -> int:
        """获取采样率"""
        return self.emotivoice_config.sampling_rate

    def get_available_voices(self) -> list:
        """获取可用的说话人列表"""
        return list(self.speaker2id.keys())

    def get_available_emotions(self) -> list:
        """获取可用的情感列表"""
        return ["友好的", "开心的", "悲伤的", "严肃的", "兴奋的", "平静的"]

    def set_speaker(self, speaker_id: str):
        """设置默认说话人"""
        logger.info(f"设置说话人为: {speaker_id}")
        # 注意：这里只是记录，实际使用时需要在 synthesize 方法中指定

    def _load_speaker_ids(self):
        """从 speaker2id 文件加载所有可用的说话人ID"""
        try:
            speaker_ids = list(self.speaker2id.keys())
            if not speaker_ids:
                logger.warning("未找到可用的说话人ID，使用默认ID。")
                return ["default_speaker"]  # 提供一个明确的默认值
            return speaker_ids
        except Exception as e:
            logger.error(f"加载说话人ID失败: {e}")
            return ["default_speaker"]  # 如果加载失败，使用默认值

    def set_emotion(self, emotion: str):
        """设置默认情感"""
        logger.info(f"设置情感为: {emotion}")
        # 注意：这里只是记录，实际使用时需要在 synthesize 方法中指定