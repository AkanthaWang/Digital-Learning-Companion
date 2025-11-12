# 智能学伴 (LearningFriend)
🎓 让语音交互更自然、更智能的学习助手  

一个基于语音交互的智能学习系统，实现 **语音输入 → ASR → LLM → TTS → 语音输出** 的完整对话流程。  
系统通过模块化设计，将语音识别、语言生成与情感语音合成无缝衔接，构建高质量的人机对话体验。

---

## ✨ 核心特性

- 🎤 **高精度语音识别（ASR）**：集成阿里达摩院的 **FunASR**，支持高性能中文语音识别  
- 🤖 **智能对话生成（LLM）**：调用硅基流动的 **DeepSeek-V3** 模型，实现自然语言理解与生成  
- 🔊 **情感语音合成（TTS）**：采用网易有道开源的 **EmotiVoice**，生成富有情感的高保真语音  
- 🔄 **端到端语音交互流程**：自动完成从语音输入到语音输出的全流程  
- 🧠 **上下文记忆与多轮对话**：支持连续语境、情感延续与多轮对话  
- ⚙️ **模块化可扩展设计**：ASR / LLM / TTS 模块均可独立替换、组合与调试

---

## 🏗️ 系统架构

```
语音输入(wav/mp3) → FunASR(语音识别) → DeepSeek-V3(文本生成) → EmotiVoice(语音合成) → 语音输出(wav)
                      ↓                      ↓                         ↓
                  中文文本              智能回复文本                音频波形
```

---

## 📦 模块说明

| 模块 | 路径 | 功能 |
|------|------|------|
| **ASR 模块** | `src/asr/` | FunASR 中文语音识别 |
| **LLM 模块** | `src/llm/` | DeepSeek-V3 智能对话生成 |
| **TTS 模块** | `src/tts/` | EmotiVoice 情感语音合成 |
| **Pipeline 模块** | `src/pipeline/` | 多模块协同与流程控制 |

---

## 🚀 快速开始

1. **克隆项目**
   ```bash
   git clone <repository_url>
   cd LearningFriend
   ```

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

3. **安装FunASR**（如果未安装）
   ```bash
   cd FunASR
   pip install -e .
   cd ..
   ```

4. **配置API Key**
   
   **首次使用**：从示例文件创建配置文件
   ```bash
   cp config/config.yaml.example config/config.yaml
   ```
   
   然后编辑 `config/config.yaml`，填入你的 API Key：
   ```yaml
   llm:
     provider: "deepseek"
     deepseek:
       api_key: "sk-your-api-key-here"  # ⚠️ 替换为你的硅基流动API Key
       base_url: "https://api.siliconflow.cn/v1"
       model: "DeepSeek/DeepSeek-V3"
   ```
   
   **重要**：
   - ✅ `config.yaml` 已添加到 `.gitignore`，不会被提交到Git
   - ⚠️ **如果 `config.yaml` 已被 Git 跟踪**：请查看 [SECURITY.md](SECURITY.md) 了解如何安全处理
   
   获取API Key：访问 https://siliconflow.cn/

5. **下载 TTS 模型（EmotiVoice）**

   在首次运行前，请确保 EmotiVoice 模型已正确下载。  
   可参考详细安装说明与下载链接：

   👉 [EmotiVoice/README.md](models/EmotiVoice/README.md)

6. **运行测试**
   ```bash
   python test_pipeline.py
   ```
   
   这个测试会：
   - ✓ 测试ASR模块（FunASR）
   - ✓ 测试LLM模块（DeepSeek-V3）
   - ✓ 测试TTS模块（IndexTTS2占位）
   - ✓ 测试完整对话流程
   
   **首次运行提示**：
   - FunASR模型会自动从ModelScope下载（约1-2GB）
   - 如果配置了API Key，会测试LLM对话
---

7. **开始使用**
   ```bash
   # 交互式模式
   python main.py --mode interactive
   
   # 单文件模式
   python main.py --mode single --input data/audio_input/your_audio.wav
   
   # 批处理模式
   python main.py --mode batch --input audio_directory/
   ```

## ⚙️ 系统配置示例 (Configuration Overview)

项目的主要模块可通过 config/config.yaml 文件进行统一管理。  
以下为核心配置示例：

```yaml
asr:
  provider: "funasr"
  model_name: "iic/speech_paraformer-large-vad-punc-spk_asr_nat-zh-cn"
  device: "cuda"
  sample_rate: 16000
   ```
```yaml
llm:
  provider: "deepseek"
  deepseek:
    api_key: "sk-your-api-key"
    base_url: "https://api.siliconflow.cn/v1"
    model: "DeepSeek/DeepSeek-V3"
    temperature: 0.7
    max_tokens: 2000
   ```

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
---

## 💡 常见问题 (FAQ)

| 问题 | 解决方案 |
|------|-----------|
| **模型未找到或加载失败** | 检查 models/EmotiVoice/outputs/... 路径是否存在且完整 |
| **输出音频为空或异常** | 确认 config.yaml 配置正确，且 voice 在 speaker2id 中存在 |
| **ASR / LLM / TTS 无法联动** | 确保各模块路径正确，可独立运行 |
| **下载速度慢** | 配置 ModelScope 国内镜像或预先下载模型 |
| **CUDA 报错** | 将设备设为 cpu 运行以验证功能 |

---

## 🙏 致谢 (Acknowledgements)

- [FunASR](https://github.com/alibaba-damo-academy/FunASR) – 阿里达摩院语音识别框架  
- [DeepSeek-V3](https://siliconflow.cn/) – 硅基流动大语言模型服务  
- [EmotiVoice](https://github.com/netease-youdao/EmotiVoice) – 网易有道开源情感语音合成系统  

---

## 📄 许可证 (License)

MIT License © 2025 LearningFriend Team  
欢迎贡献与交流 🤝
