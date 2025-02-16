# 添加异常处理和模型验证
import whisper
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

class whisper_v3_tb:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model_id = "openai/whisper-large-v3-turbo"
        
        try:
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(  
                self.model_id,
                torch_dtype=self.torch_dtype,
                use_safetensors=True
            ).to(self.device)
            
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                device=self.device,
            )
        except Exception as e:
            raise RuntimeError(f"模型加载失败: {str(e)}")

    def w4a2txt(self, w4a_vido_path):
        """添加文件检查和异常处理"""
        try:
            result = self.pipe(
                w4a_vido_path,
                generate_kwargs={"task": "transcribe"},  # 中文支持
                return_timestamps=True
            )
            print("user:"+result["text"])
            return result["text"]
        except FileNotFoundError:
            print(f"错误：文件 {w4a_vido_path} 不存在")
            return ""