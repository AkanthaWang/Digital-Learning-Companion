"""
智能学伴系统 - 主程序入口
实现：语音输入 -> ASR -> LLM -> TTS -> 语音输出
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# 项目根路径
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import load_config
from src.pipeline import ConversationPipeline


def setup_logging(config):
    """配置日志系统"""
    log_config = config.get('logging', {})

    # 统一到项目根路径
    log_file = Path(ROOT) / log_config.get('file', 'data/logs/system.log')
    os.makedirs(log_file.parent, exist_ok=True)

    log_level = log_config.get('level', 'INFO')
    log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logging.basicConfig(
        level=getattr(logging, log_level),
        format=log_format,
        handlers=[
            logging.FileHandler(str(log_file), encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return logging.getLogger(__name__)


def interactive_mode(pipeline: ConversationPipeline):
    """交互式模式"""
    print("\n" + "="*60)
    print("智能学伴系统 - 交互式模式")
    print("="*60)
    print("输入音频文件路径（相对路径自动基于项目根目录 data/audio_input）")
    print("输入 'quit' 退出，'reset' 重置对话")
    print("="*60 + "\n")

    input_base = ROOT / "data" / "audio_input"
    output_base = ROOT / "data" / "audio_output"
    input_base.mkdir(parents=True, exist_ok=True)
    output_base.mkdir(parents=True, exist_ok=True)

    while True:
        try:
            user_input = input("\n请输入音频文件名或完整路径: ").strip()

            if user_input.lower() == 'quit':
                print("再见！")
                break
            if user_input.lower() == 'reset':
                pipeline.reset_conversation()
                print("✓ 对话已重置")
                continue
            if not user_input:
                continue

            # 支持只输入文件名
            if not os.path.isabs(user_input):
                user_input = str(input_base / user_input)

            if not os.path.exists(user_input):
                print(f"✗ 文件不存在: {user_input}")
                continue

            print(f"\n处理中... {user_input}")
            result = pipeline.process_audio_file(user_input)

            if result['success']:
                print(f"\n{'─'*60}")
                print(f"👤 用户: {result['asr_text']}")
                print(f"🤖 助手: {result['llm_response']}")
                if result.get('output_audio_path'):
                    print(f"🔊 语音已保存: {result['output_audio_path']}")
                print(f"✓ 完成 (第{pipeline.get_conversation_count()}轮对话)\n")
            else:
                print(f"✗ 处理失败: {result.get('error', '未知错误')}\n")

        except KeyboardInterrupt:
            print("\n\n再见！")
            break
        except Exception as e:
            print(f"\n✗ 错误: {str(e)}\n")


def batch_mode(pipeline: ConversationPipeline, input_dir: str):
    """批处理模式"""
    input_dir = Path(input_dir)
    if not input_dir.is_absolute():
        input_dir = ROOT / input_dir
    input_dir.mkdir(parents=True, exist_ok=True)

    output_dir = ROOT / "data" / "audio_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("智能学伴系统 - 批处理模式")
    print("="*60)
    print(f"输入目录: {input_dir}")
    print("="*60 + "\n")

    audio_files = [p for p in input_dir.glob("*") if p.suffix.lower() in {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}]
    if not audio_files:
        print(f"✗ 未找到音频文件")
        return

    for i, audio_file in enumerate(audio_files, 1):
        print(f"\n[{i}/{len(audio_files)}] 处理: {audio_file.name}")
        try:
            result = pipeline.process_audio_file(str(audio_file))
            if result['success']:
                print(f"  ✓ 成功")
                print(f"  用户: {result['asr_text']}")
                print(f"  助手: {result['llm_response'][:100]}...")
            else:
                print(f"  ✗ 失败: {result.get('error', '未知错误')}")
        except Exception as e:
            print(f"  ✗ 错误: {e}")


def single_file_mode(pipeline: ConversationPipeline, audio_file: str):
    """单文件模式"""
    audio_file = Path(audio_file)
    if not audio_file.is_absolute():
        audio_file = ROOT / audio_file

    print("\n" + "="*60)
    print("智能学伴系统 - 单文件模式")
    print("="*60)
    print(f"输入文件: {audio_file}")
    print("="*60 + "\n")

    if not audio_file.exists():
        print(f"✗ 文件不存在: {audio_file}")
        return

    print("处理中...\n")
    result = pipeline.process_audio_file(str(audio_file))

    if result['success']:
        print(f"{'─'*60}")
        print(f"👤 用户: {result['asr_text']}")
        print(f"🤖 助手: {result['llm_response']}")
        print(f"{'─'*60}")
        if result.get('output_audio_path'):
            print(f"🔊 语音已保存: {result['output_audio_path']}")
    else:
        print(f"✗ 处理失败: {result.get('error', '未知错误')}\n")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='智能学伴系统 - 语音对话助手')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='配置文件路径')
    parser.add_argument('--mode', type=str, choices=['interactive', 'batch', 'single'], default='interactive')
    parser.add_argument('--input', type=str, help='输入文件或目录路径')

    args = parser.parse_args()

    try:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = ROOT / config_path

        print(f"加载配置文件: {config_path}")
        config = load_config(str(config_path))

        logger = setup_logging(config)
        logger.info("="*60)
        logger.info("智能学伴系统启动")
        logger.info("="*60)

        pipeline = ConversationPipeline(config)

        if args.mode == 'interactive':
            interactive_mode(pipeline)
        elif args.mode == 'batch':
            if not args.input:
                print("✗ 批处理模式需要指定 --input 目录")
                return
            batch_mode(pipeline, args.input)
        elif args.mode == 'single':
            if not args.input:
                print("✗ 单文件模式需要指定 --input 文件路径")
                return
            single_file_mode(pipeline, args.input)

        logger.info("智能学伴系统已退出")

    except KeyboardInterrupt:
        print("\n程序已中断")
    except Exception as e:
        import traceback
        print(f"\n✗ 错误: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
