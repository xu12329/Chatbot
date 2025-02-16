from dotenv import load_dotenv
from src.core.stream_work import ConvoManager
import traceback
import logging

# 配置日志输出
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

if __name__ == "__main__":
    try:
        load_dotenv()  # 确保.env文件在项目根目录
        assistant = ConvoManager()
        assistant.init_deepseek("我们正在进行谈话，你不需要生成计划表之类的较多内容，只需要生成正常对话。")
        while True:
            try:
                assistant.begin_recording()
                transcript = assistant.transcribe_audio()
                if transcript:
                    assistant.generate_response()
            except KeyboardInterrupt:
                print("\n程序已退出")
                break
            except Exception as e:
                logging.error(f"运行时异常: {str(e)}", exc_info=True)
    except Exception as e:
        logging.critical(f"初始化失败: {str(e)}", exc_info=True)