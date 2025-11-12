# EmotiVoice TTS

本模块用于将文本转换为语音（TTS），底层基于网易有道的 **[EmotiVoice](https://github.com/netease-youdao/EmotiVoice)**，主要用于项目流程的最后阶段：

> ASR → LLM → TTS  
> 将模型生成的文字转化为语音输出。

---

## 🚀 功能特性

- 文本 → 语音合成  
- 支持多说话人（`voice` / `speaker_id`）  
- 支持情感与风格提示（如“友好的”“开心的”）  
- 输出格式：WAV  
- 路径与参数可自定义配置  

---

## ⚙️ 快速开始

### 1. 克隆 EmotiVoice 仓库

```bash
git lfs clone https://github.com/netease-youdao/EmotiVoice.git models/EmotiVoice
```

### 2. 下载模型文件  
按照 EmotiVoice 官方说明下载模型文件，并放入对应目录（例如 `outputs/`、`WangZeJun/` 等）。

---

### 示例目录结构

以下展示了 EmotiVoice 的核心目录（仅保留与推理 / 使用相关的关键文件）：

```
EmotiVoice/
├── config/
│   └── joint/
│       ├── config.yaml              # 主配置文件（需在 TTS 封装中引用）
│
├── models/
│   ├── prompt_tts_modified/         # 声学模型与文本前端模块
│   └── hifigan/                     # 声码器（vocoder）
│
├── outputs/
│   ├── prompt_tts_open_source_joint/
│   │   └── ckpt/
│   │       ├── g_00140000           # 声学模型 checkpoint
│   │       └── do_00140000
│   └── style_encoder/
│       └── ckpt/
│           └── checkpoint_163431    # 风格编码器 checkpoint
│
├── WangZeJun/
│   └── simbert-base-chinese/        # 语义嵌入模型（SimBERT）
│
├── text/                            # 文本清理与音素处理逻辑
│   ├── cleaners.py
│   ├── cmudict.py
│   ├── numbers.py
│   └── symbols.py
│
├── frontend.py                      # 推理入口：拼音/音素前端
├── inference_tts.py                 # 官方 TTS 推理脚本
├── laxapi2.py                       # HTTP 推理 API（简化封装）
└── README_小白安装教程.md           # 官方安装说明
```

> ⚠️ **注意**  
> - 推理仅需 `config/`、`models/`、`outputs/`、`WangZeJun/` 四个核心模块。  
> - 若使用自定义封装（如 `EmotiVoiceTTS`），请确保 `config.yaml` 与权重路径一致。  

---

### 3. 安装依赖

如果主项目中尚未安装，可执行：
```bash
pip install torch torchaudio numpy soundfile transformers yacs
```

---

## 🧩 使用示例

假设封装类名为 `EmotiVoiceTTS`（定义在 `emotivoice_local.py`）：

```python
from emotivoice_local import EmotiVoiceTTS

config = {
    "output_directory": "data/audio_output",
    "model_config_path": "models/EmotiVoice/config/joint/config.yaml",
    # 可选参数：
    # "token_list_path": "models/EmotiVoice/data/youdao/tokenlist",
    # "speaker2id_path": "models/EmotiVoice/data/youdao/speaker2",
}

tts = EmotiVoiceTTS(config)
text = "你好，欢迎使用语音合成。"

# 方法 1：返回音频数据（numpy 数组）
audio = tts.synthesize(text, voice="7556", emotion="友好的")

# 方法 2：直接合成并保存为文件
tts.synthesize_to_file(
    text,
    "data/audio_output/demo.wav",
    voice="7556",
    emotion="友好的",
)
```

---

## 🎙️ 获取可用说话人与情感

```python
voices = tts.get_available_voices()
emotions = tts.get_available_emotions()
print(voices, emotions)
```

---

## ❓ 常见问题

| 问题 | 可能原因与解决方案 |
|------|--------------------|
| **提示找不到模型** | 请确认 `models/EmotiVoice/outputs/...` 路径存在且模型文件完整。 |
| **输出音频为空或异常** | 检查 `config.yaml` 是否为原始版本，`voice` 是否存在于 `speaker2id` 中。 |
| **与 ASR / LLM 模块联动** | 可直接将 LLM 输出文本传入 `synthesize_to_file()`，生成 WAV 文件即可播放。 |

---

## 📜 许可证

本模块遵循主仓库的 LICENSE；  
使用 EmotiVoice 的部分请遵守其原始仓库的许可条款。