# 智能学伴 (LearningFriend)

一个基于语音交互的智能学习助手，实现 **语音输入 → ASR → LLM → TTS → 语音输出** 的完整对话流程。

## ✨ 特性

- 🎤 **高质量语音识别**：采用阿里达摩院的 FunASR，支持中文语音识别  
- 🤖 **智能对话生成**：集成硅基流动的 DeepSeek-V3 大语言模型  
- 🔊 **情感语音合成（TTS）**：引入 EmotiVoice，实现带有情绪色彩的高保真语音生成  
- 🔄 **端到端语音交互流程**：从语音输入到语音输出的自动化管线  
- 🧠 **多轮上下文记忆**：支持上下文对话保持与语义延续  
- ⚙️ **模块化与可扩展性**：ASR / LLM / TTS 模块均可独立替换或组合使用

## 🏗️ 系统架构

```
语音输入(wav/mp3)
   ↓
FunASR (ASR识别)
   ↓
DeepSeek-V3 (文本生成)
   ↓
EmotiVoice (TTS情感合成)
   ↓
语音输出(wav)
```

### 模块说明

1. **ASR模块** (`src/asr/`): 基于 FunASR 的中文语音识别 - [详细文档](src/asr/README.md)  
2. **LLM模块** (`src/llm/`): DeepSeek-V3 对话模型 - [详细文档](src/llm/README.md)  
3. **TTS模块** (`src/tts/`): 采用网易有道开源的 EmotiVoice 实现情感语音合成 - [详细文档](src/tts/README.md)  
4. **Pipeline模块** (`src/pipeline/`): 对话流程控制与多模块协同

## 🚀 快速开始

### 环境要求

- Python 3.8+  
- CUDA 环境（推荐使用 GPU）  
- 硅基流动 API Key（DeepSeek 模型）  

### 1️⃣ 克隆项目

```bash
git clone <repository_url>
```

### 2️⃣ 安装依赖

```bash
pip install -r requirements.txt
```

### 3️⃣ 安装FunASR（如果未安装）

```cd FunASR
pip install -e .
cd ..
```


### 4️⃣ 配置 LLM API Key

```bash
cp config/config.yaml.example config/config.yaml
```

修改`config/config.yaml`：

```yaml
llm:
  provider: "deepseek"
  deepseek:
    api_key: "sk-your-api-key"
    base_url: "https://api.siliconflow.cn/v1"
    model: "DeepSeek/DeepSeek-V3"
```

### 5️⃣ 测试运行

```bash
python test_pipeline.py
```

## ⚙️ TTS 配置

```yaml
tts:
  provider: "emotivoice"
  device: "cuda"
  voice: "default"
  emotion: "happy"
  sample_rate: 22050
  speed: 1.0
  pitch: 1.0
  model_path: "models/EmotiVoice/outputs"
```

## 🙏 致谢

- [FunASR](https://github.com/alibaba-damo-academy/FunASR)  
- [DeepSeek-V3](https://siliconflow.cn/)  
- [EmotiVoice](https://github.com/netease-youdao/EmotiVoice)

## 📄 许可证

MIT License © 2025 LearningFriend Team
