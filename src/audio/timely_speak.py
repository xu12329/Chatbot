import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import threading
import os

class AudioRecorder:
    def __init__(self, samplerate=44100, channels=1, device=None, max_duration=None):
        """
        初始化音频录制器
        :param samplerate: 采样率（默认44100Hz）
        :param channels: 声道数（默认1，单声道）
        :param device: 输入设备索引（默认系统默认设备）
        :param max_duration: 最大录音时长（秒），None表示无限制
        """
        self.samplerate = samplerate
        self.channels = channels
        self.device = device
        self.max_duration = max_duration
        self.recording = None
        self.stream = None
        self.timer = None  # 用于超时停止的定时器

    def start(self):
        """开始录音"""
        if self.stream:
            raise RuntimeError("录音已在进行中")
        
        self.recording = []
        self.stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            device=self.device,
            callback=self._callback
        )
        self.stream.start()

        # 启动超时定时器
        if self.max_duration is not None:
            self.timer = threading.Timer(
                interval=self.max_duration,
                function=self.stop  # 超时后自动调用stop
            )
            self.timer.start()

    def _callback(self, indata, frames, time, status):
        """音频数据回调函数"""
        if status:
            print(f"录音警告: {status}")
        self.recording.append(indata.copy())

    def stop(self):
        """停止录音并返回音频数据"""
        # 取消定时器（如果存在）
        if self.timer:
            self.timer.cancel()
            self.timer = None

        # 停止音频流
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

            # 合并录音数据
            if self.recording:
                self.recording = np.concatenate(self.recording, axis=0)
            return self.recording
        return None

    def save_wav(self, filename):
        if self.recording is not None:
            try:
                data = np.int16(self.recording * (32767 / np.max(np.abs(self.recording))))
                write(filename, self.samplerate, data)
                print(f"音频文件已保存到 {os.path.abspath(filename)}")  # 添加路径输出
            except Exception as e:
                print(f"保存失败: {str(e)}")