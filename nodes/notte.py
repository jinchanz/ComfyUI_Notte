import json

class Notte:
    """
    音频分析节点 - 处理音频并返回说话人数量、片段和语言信息
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_url": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
                "hf_token": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
            }
        }
    
    RETURN_TYPES = ("INT", "STRING", "STRING",)
    RETURN_NAMES = ("speakers_count", "segments", "language",)
    
    FUNCTION = "process"
    CATEGORY = "audio"

    def process(self, audio_url, hf_token):
        from modules.predict import Predictor
        
        try:
            predictor = Predictor()
            predictor.setup(hf_token=hf_token)
            
            output = predictor.predict(file_url=audio_url)
            
            # 将 segments 转换为格式化的 JSON 字符串
            segments_json = json.dumps(output.segments, ensure_ascii=False, indent=2)
            
            return (
                output.num_speakers,
                segments_json,
                output.language,
            )
            
        except Exception as e:
            print(f"处理音频时出错: {str(e)}")
            return (0, "[]", "") 