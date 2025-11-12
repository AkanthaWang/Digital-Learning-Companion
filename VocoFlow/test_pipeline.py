"""
端到端测试脚本（路径修正版）
- 自动查找项目根目录（包含 config/ 与 src/）
- 统一使用绝对路径，避免被第三方模块 chdir 影响
"""

import os
import sys
import logging
from pathlib import Path

import numpy as np
import soundfile as sf

# ---------- 项目根目录定位 ----------
def _find_project_root(start: Path) -> Path:
    cur = start
    for _ in range(6):
        if (cur / "config").exists() and (cur / "src").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    # 兜底：用脚本所在目录
    return start

HERE = Path(__file__).resolve().parent
ROOT = _find_project_root(HERE)

# 确保可以 import 本项目
sys.path.insert(0, str(ROOT))

from config import load_config
from src.asr import FunASRModule
from src.llm import LLMInterface
from src.tts import EmotiVoiceTTS


def _cfg():
    """显式用绝对路径加载配置"""
    return load_config(str(ROOT / "config" / "config.yaml"))


def test_asr_module():
    print("\n" + "="*60)
    print("测试1: ASR模块 - FunASR语音识别")
    print("="*60)
    try:
        config = _cfg()
        asr = FunASRModule(config['asr'])
        print("✓ ASR模块初始化成功")
        print(f"  模型: {asr.model_name}")
        print(f"  设备: {asr.device}")
        print(f"  采样率: {asr.sample_rate}Hz")

        # 1秒静音
        test_audio = np.zeros(16000, dtype=np.float32)
        print("\n尝试识别测试音频...")
        result = asr.transcribe_array(test_audio)
        print(f"✓ ASR识别完成: '{result}'")
        return True
    except Exception as e:
        print(f"✗ ASR模块测试失败: {e}")
        import traceback; traceback.print_exc()
        return False


def test_llm_module():
    print("\n" + "="*60)
    print("测试2: LLM模块 - DeepSeek-V3对话")
    print("="*60)
    try:
        config = _cfg()
        llm = LLMInterface(config['llm'])
        print("✓ LLM模块初始化成功")
        print(f"  提供商: {llm.provider}")
        print(f"  模型: {llm.model_name}")
        print(f"  基础URL: {llm.client.base_url}")

        api_key = config['llm']['deepseek'].get('api_key', '')
        if not api_key:
            print("⚠ 警告: API Key未配置，跳过LLM对话测试（填写 config/config.yaml 后再测）")
            return None

        print("\n尝试发送测试消息...")
        response = llm.chat("你好，请简单介绍一下自己", use_history=False)
        print(f"✓ LLM回复: {response[:100]}...")
        return True
    except Exception as e:
        print(f"✗ LLM模块测试失败: {e}")
        import traceback; traceback.print_exc()
        return False


def test_tts_module():
    print("\n" + "="*60)
    print("测试3: TTS模块 - EmotiVoice语音合成")
    print("="*60)
    try:
        config = _cfg()
        tts = EmotiVoiceTTS(config['tts'])

        # 关键：EmotiVoice 可能改了 cwd，这里强制切回项目根目录
        os.chdir(str(ROOT))

        print("✓ TTS模块初始化成功")
        print(f"  设备: {tts.device}")
        print(f"  采样率: {tts.sample_rate}Hz")
        print(f"  音色ID: {tts.speaker2id}")

        print("\n尝试合成语音...")
        test_text = "你好，我是数字学伴"
        emotion = "友好的"
        voice = "1050"

        audio_data = tts.synthesize(test_text, voice=voice, emotion=emotion)
        print("✓ TTS合成完成")
        print(f"  音频长度: {len(audio_data)} 样本")
        print(f"  音频时长: {len(audio_data) / tts.sample_rate:.2f} 秒")

        # 统一输出目录：ROOT/data/audio_output
        output_dir = ROOT / "data" / "audio_output"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "test_tts_output.wav"

        sf.write(str(output_path), audio_data, tts.sample_rate)
        print(f"  已保存到: {output_path}")
        return True
    except Exception as e:
        print(f"✗ TTS模块测试失败: {e}")
        import traceback; traceback.print_exc()
        return False


def test_full_pipeline():
    print("\n" + "="*60)
    print("测试4: 完整对话流程")
    print("="*60)
    try:
        # 防御：进入项目根目录，避免内部相对路径飘移
        os.chdir(str(ROOT))

        from src.pipeline import ConversationPipeline
        config = _cfg()
        pipeline = ConversationPipeline(config)
        print("✓ 对话流程控制器初始化成功")

        test_audio = np.zeros(16000, dtype=np.float32)  # 1秒静音
        print("\n尝试处理完整对话流程...")
        result = pipeline.process_audio_array(test_audio, sample_rate=16000)

        if result['success']:
            print("✓ 完整流程测试成功")
            print(f"  ASR识别: {result['asr_text']}")
            msg = result['llm_response']
            print(f"  LLM回复: {msg[:100]}..." if len(msg) > 100 else f"  LLM回复: {msg}")
            if result.get('output_audio_path'):
                print(f"  输出音频: {result['output_audio_path']}")
            print(f"  对话轮数: {pipeline.get_conversation_count()}")
        else:
            print(f"✗ 完整流程测试失败: {result.get('error', '未知错误')}")
        return result['success']
    except Exception as e:
        print(f"✗ 完整流程测试失败: {e}")
        import traceback; traceback.print_exc()
        return False


def main():
    print("\n" + "="*60)
    print("数字学伴系统 - 端到端测试（路径修正版）")
    print("="*60)
    print(f"项目根目录: {ROOT}")
    print("="*60)

    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    results = {
        'asr': test_asr_module(),
        'llm': test_llm_module(),
        'tts': test_tts_module(),
    }

    if results['asr'] and (results['llm'] is not False) and results['tts']:
        results['pipeline'] = test_full_pipeline()

    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    for module, result in results.items():
        status = "✓ 通过" if result is True else ("⚠ 跳过" if result is None else "✗ 失败")
        print(f"  {module.upper():10s}: {status}")
    print("="*60)

    if all(r is True or r is None for r in results.values()):
        print("\n🎉 所有测试通过！系统可以正常运行。")
        print("下一步：准备音频到 data/audio_input/ 并运行 main.py --mode interactive")
        return 0
    else:
        print("\n⚠️  部分测试失败，请检查配置和依赖。")
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
