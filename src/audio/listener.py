import sounddevice as sd
import numpy as np
import soundfile as sf
import time
import sys
from queue import Queue

class AudioListener:
    def __init__(self):
        # 配置参数
        self.THRESHOLD = 0.02  # 音量阈值，根据实际情况调整
        self.SAMPLE_RATE = 16000  # 采样率
        self.CHANNELS = 1  # 单声道
        self.DEVICE = None  # 使用默认音频设备
        self.SILENCE_DURATION = 1.5  # 持续静音多少秒后停止
        self.MIN_RECORD_DURATION = 1.0  # 最小录音时长（秒）
        self.MAX_RECORD_DURATION = 30.0  # 最大录音时长（秒）
        self.BLOCK_DURATION = 0.1  # 每次处理的音频块时长（秒）
        self.OUTPUT_PREFIX = "recording"
        
        self.q = Queue()
        self.audio_buffer = []
        self.silence_counter = 0
        self.start_time = 0
        self.is_recording = False

    def _callback(self, indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        self.q.put(indata.copy())

    def _process_audio(self):
        data = self.q.get()
        current_rms = np.sqrt(np.mean(data**2))

        # 检测声音触发
        if not self.is_recording and current_rms >= self.THRESHOLD:
            print("检测到声音输入，开始录音...")
            self.is_recording = True
            self.start_time = time.time()
            self.audio_buffer.append(data)
            return

        if self.is_recording:
            # 添加到缓冲区
            self.audio_buffer.append(data)

            # 检查静音
            if current_rms < self.THRESHOLD:
                self.silence_counter += 1
            else:
                self.silence_counter = 0

            # 检查停止条件
            elapsed = time.time() - self.start_time
            stop_conditions = [
                (self.silence_counter >= int(self.SILENCE_DURATION / self.BLOCK_DURATION)),
                (elapsed >= self.MAX_RECORD_DURATION)
            ]

            if any(stop_conditions):
                if elapsed < self.MIN_RECORD_DURATION:
                    print("录音时间过短，继续监听...")
                    self.audio_buffer = []
                    self.silence_counter = 0
                    self.is_recording = False
                    return
                
                print(f"停止录音，持续静音 {self.SILENCE_DURATION}秒" if stop_conditions[0] 
                      else f"达到最大录音时长 {self.MAX_RECORD_DURATION}秒")
                return True

    def _save_recording(self):
        if self.is_recording and len(self.audio_buffer) > 0:
            audio_data = np.concatenate(self.audio_buffer)
            filename = "record.wav"
            sf.write(filename, audio_data, self.SAMPLE_RATE)
            print(f"文件已保存：{filename}")
            # 这里可以添加调用whisper处理的代码
            # whisper_process(filename)

    def start(self):
        print("启动音频监听循环...")
        while True:
            try:
                with sd.InputStream(
                    samplerate=self.SAMPLE_RATE,
                    channels=self.CHANNELS,
                    device=self.DEVICE,
                    callback=self._callback,
                    blocksize=int(self.SAMPLE_RATE * self.BLOCK_DURATION)
                ):
                    print("\n等待声音输入...")
                    while True:
                        if self._process_audio():
                            break
                    
                    self._save_recording()
                    # 重置状态
                    self.audio_buffer = []
                    self.silence_counter = 0
                    self.is_recording = False

            except KeyboardInterrupt:
                print("\n程序终止")
                break
            except Exception as e:
                print(f"发生错误：{str(e)}")
                time.sleep(1)

if __name__ == "__main__":
    listener = AudioListener()
    listener.start()