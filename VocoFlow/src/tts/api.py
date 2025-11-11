import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import logging
import io
import torch
import glob
import time
from pathlib import Path
import sys

# 添加 EmotiVoice 到路径
ROOT = Path(__file__).resolve().parents[2]
EMOTIVOICE_ROOT = ROOT / "models" / "EmotiVoice"

sys.path.insert(0, str(EMOTIVOICE_ROOT))

# print(sys.path)  # 查看当前的模块搜索路径

os.chdir(EMOTIVOICE_ROOT)

from fastapi import FastAPI, Response, HTTPException
from pydantic import BaseModel
from frontend import g2p_cn_en, ROOT_DIR, read_lexicon, G2p
from models.prompt_tts_modified.jets import JETSGenerator
from models.prompt_tts_modified.simbert import StyleEncoder
from transformers import AutoTokenizer
import numpy as np
import soundfile as sf
import pyrubberband as pyrb
from pydub import AudioSegment
from yacs import config as CONFIG
from config.joint.config import Config

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)
#torch.cuda.set_device(2)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
config = Config()
MAX_WAV_VALUE = 32768.0

app = FastAPI()

from typing import Optional
class SpeechRequest(BaseModel):
    input: str
    voice: str = '7556'
    prompt: Optional[str] = ''
    language: Optional[str] = 'zh_us'
    model: Optional[str] = 'emoti-voice'
    response_format: Optional[str] = 'wav'
    speed: Optional[float] = 1.0

def scan_checkpoint(cp_dir, prefix, c=8):
    pattern = os.path.join(cp_dir, prefix + '?'*c)
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]

def get_models():
    am_checkpoint_path = scan_checkpoint(f'{config.output_directory}/prompt_tts_open_source_joint/ckpt', 'g_')
    style_encoder_checkpoint_path = scan_checkpoint(f'{config.output_directory}/style_encoder/ckpt', 'checkpoint_', 6)

    with open(config.model_config_path, 'r') as fin:
        conf = CONFIG.load_cfg(fin)

    conf.n_vocab = config.n_symbols
    conf.n_speaker = config.speaker_n_labels

    style_encoder = StyleEncoder(config)
    #model_CKPT = torch.load(style_encoder_checkpoint_path, map_location="cpu")     1
    model_CKPT = torch.load(style_encoder_checkpoint_path, map_location=DEVICE)
    model_ckpt = {key[7:]: value for key, value in model_CKPT['model'].items()}
    style_encoder.load_state_dict(model_ckpt, strict=False)
    generator = JETSGenerator(conf).to(DEVICE)

    model_CKPT = torch.load(am_checkpoint_path, map_location=DEVICE)
    generator.load_state_dict(model_CKPT['generator'])
    generator.eval()

    tokenizer = AutoTokenizer.from_pretrained(config.bert_path)

    with open(config.token_list_path, 'r') as f:
        token2id = {t.strip(): idx for idx, t, in enumerate(f.readlines())}

    with open(config.speaker2id_path, encoding='utf-8') as f:
        speaker2id = {t.strip(): idx for idx, t in enumerate(f.readlines())}

    return (style_encoder, generator, tokenizer, token2id, speaker2id)

def get_style_embedding(prompt, tokenizer, style_encoder):
    prompt = tokenizer([prompt], return_tensors="pt")
    input_ids = prompt["input_ids"]
    token_type_ids = prompt["token_type_ids"]
    attention_mask = prompt["attention_mask"]
    with torch.no_grad():
        output = style_encoder(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
    #style_embedding = output["pooled_output"].cpu().squeeze().numpy()
    style_embedding = output["pooled_output"].squeeze()
    return style_embedding

def emotivoice_tts(text, prompt, content, speaker, models):
    (style_encoder, generator, tokenizer, token2id, speaker2id) = models

    style_embedding = get_style_embedding(prompt, tokenizer, style_encoder)
    content_embedding = get_style_embedding(content, tokenizer, style_encoder)

    speaker = speaker2id[speaker]
    text_int = [token2id[ph] for ph in text.split()]

    sequence = torch.from_numpy(np.array(text_int)).to(DEVICE).long().unsqueeze(0)
    sequence_len = torch.from_numpy(np.array([len(text_int)])).to(DEVICE)
    #style_embedding = torch.from_numpy(style_embedding).to(DEVICE).unsqueeze(0)
    style_embedding = style_embedding.to(DEVICE).unsqueeze(0)
    #content_embedding = torch.from_numpy(content_embedding).to(DEVICE).unsqueeze(0)
    content_embedding = content_embedding.to(DEVICE).unsqueeze(0)
    speaker = torch.from_numpy(np.array([speaker])).to(DEVICE)

    with torch.no_grad():
        infer_output = generator(
            inputs_ling=sequence,
            inputs_style_embedding=style_embedding,
            input_lengths=sequence_len,
            inputs_content_embedding=content_embedding,
            inputs_speaker=speaker,
            alpha=1.0
        )

    audio = infer_output["wav_predictions"].squeeze() * MAX_WAV_VALUE
    audio = audio.cpu().numpy().astype('int16')
    return audio

speakers = config.speakers
models = None # 全局变量，初始值为None
lexicon = read_lexicon(f"{ROOT_DIR}/lexicon/librispeech-lexicon.txt")
g2p = G2p()
'''
@app.on_event("startup")
def startup_event():
    global models
    models = get_models()
'''

@app.post("/load")
def load_models():
    global models
    try:
        models = get_models()  # 加载模型
        return {"message": "模型加载成功"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
def emotts(speechRequest: SpeechRequest):
    global models
    if models is None:
        raise HTTPException(status_code=500, detail="模型尚未加载，请先调用/load接口加载模型。")
    
    try:
        start_time = time.time()  # 开始计时
        text = g2p_cn_en(speechRequest.input, g2p, lexicon)
        np_audio = emotivoice_tts(text, speechRequest.prompt, speechRequest.input, speechRequest.voice, models)
        y_stretch = np_audio
        if speechRequest.speed != 1.0:
            y_stretch = pyrb.time_stretch(np_audio, config.sampling_rate, speechRequest.speed)
        wav_buffer = io.BytesIO()
        sf.write(file=wav_buffer, data=y_stretch, samplerate=config.sampling_rate, format='WAV')
        buffer = wav_buffer
        response_format = speechRequest.response_format
        if response_format != 'wav':
            wav_audio = AudioSegment.from_wav(wav_buffer)
            wav_audio.frame_rate = config.sampling_rate
            buffer = io.BytesIO()
            wav_audio.export(buffer, format=response_format)
        end_time = time.time()  # 结束计时
        duration = end_time - start_time
        LOGGER.info(f"Predict function took {duration:.2f} seconds")
        return Response(content=buffer.getvalue(), media_type=f"audio/{response_format}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "欢迎使用文字转语音API"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9003, reload=True)
