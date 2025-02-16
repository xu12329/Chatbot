#录音->whisper转录->deepseek回答->返回结果->回到第一步
# 修复消息格式问题、方法调用错误和日志格式
from ..nlp.API import Deepseek
from ..nlp.whisper_time import whisper_v3_tb
from ..audio.timely_speak import AudioRecorder
#from ..audio.Tacotron_txt2wave import Tacotron2_Txt2Wave
import time  # 避免忙等待
import numpy as np

class ConvoManager:
    def __init__(self):
        self.deepseek = Deepseek()
        self.whisper = whisper_v3_tb()
        self.max_recording_duration = 60  # 更合理的最大录音时长（秒）
        self.audiorecorder = AudioRecorder(16000, max_duration=self.max_recording_duration)
        self.path_wav = "timed_recording.wav"
        #self.to_audio = Tacotron2_Txt2Wave()
        self.log = []  # 统一日志格式为消息字典

    def init_deepseek(self,send_massage):
        self.log.append({"role": "system", "content": send_massage})

    def begin_recording(self):
        """处理录音流程，添加异常处理"""
        try:
            print("开始讲话（说话结束后保持3秒静音将自动停止）")
            self.audiorecorder.start()
            
            # 初始化静音检测变量
            last_sound_time = time.time()
            silence_threshold = 3  # 3秒静音阈值
            
            while self.audiorecorder.stream:
                # 检查是否有新数据
                if self.audiorecorder.recording:
                    # 计算当前音频的RMS值
                    current_data = self.audiorecorder.recording[-1]
                    rms = np.sqrt(np.mean(current_data**2))
                    
                    # 更新最后检测到声音的时间
                    if rms > 0.01:  # 简单的声音阈值
                        last_sound_time = time.time()
                
                # 检查是否达到静音时间
                if time.time() - last_sound_time > silence_threshold:
                    break
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            self.audiorecorder.stop()
            print("\n手动停止录音")
        except Exception as e:
            print(f"录音错误: {str(e)}")
        finally:
            self.audiorecorder.stop()

        if self.audiorecorder.recording is not None:
            duration = len(self.audiorecorder.recording) / self.audiorecorder.samplerate
            self.audiorecorder.save_wav(self.path_wav)
            print(f"实际录制时长: {duration:.2f}秒")

    def transcribe_audio(self):
        """处理语音转文本，添加异常处理"""
        try:
            transcript = self.whisper.w4a2txt(self.path_wav)
            self.log.append({"role": "user", "content": transcript})
            return transcript
        except Exception as e:
            print(f"转录失败: {str(e)}")
            return None

    def generate_response(self):
        """生成回复并处理流式输出"""
        if not self.log:
            print("没有需要回复的内容")
            return

        try:
            full_response = []
            for chunk in self.deepseek.chat_stream(messages=self.log):
                print(chunk, end="", flush=True)
                full_response.append(chunk)
                #audio = self.to_audio.text_to_speech(chunk)
                #self.to_audio.play_audio(audio)
            # 将完整回复加入历史
            self.log.append({"role": "assistant", "content": "".join(full_response)})
        except Exception as e:
            print(f"\n回复生成失败: {str(e)}")