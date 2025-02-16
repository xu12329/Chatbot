import torch
from speechbrain.inference.TTS import Tacotron2
from speechbrain.inference.vocoders import HIFIGAN

class Tacotron2_Txt2Wave:
    def __init__(self):
        # 从SpeechBrain加载Tacotron2模型
        try:
            # 自动选择设备
            device = "cuda" if torch.cuda.is_available() else "cpu"
            run_opts = {"device": device}
            
            # 加载Tacotron2模型
            self.tacotron2 = Tacotron2.from_hparams(
                source="speechbrain/tts-tacotron2-ljspeech",
                savedir="pretrained_models/tts-tacotron2-ljspeech",
                run_opts=run_opts
            )
            
            # 加载HiFi-GAN声码器
            self.hifi_gan = HIFIGAN.from_hparams(
                source="speechbrain/tts-hifigan-ljspeech",
                savedir="pretrained_models/tts-hifigan-ljspeech",
                run_opts=run_opts
            )
            
        except Exception as e:
            raise RuntimeError(f"模型加载失败: {str(e)}")

    def text_to_speech(self, text):
        # 将文本转换为语音
        with torch.no_grad():
            # 使用Tacotron2生成mel谱
            mel_output, mel_length, alignment = self.tacotron2.encode_text(text)
            
            # 使用HiFi-GAN生成波形
            waveforms = self.hifi_gan.decode_batch(mel_output)
            audio = waveforms.squeeze(1).cpu().numpy()
            
        return audio[0]  # 返回第一个batch的音频

    def save_audio(self, audio, filepath="output.wav", sample_rate=22050):
        """保存生成的音频文件"""
        import torchaudio
        torchaudio.save(filepath, torch.tensor(audio).unsqueeze(0), sample_rate)

    def play_audio(self, audio):
        # 播放生成的音频
        import sounddevice as sd
        sd.play(audio, samplerate=22050)
        sd.wait()