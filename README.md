# 音频转录与说话人分离系统

这是一个基于 Whisper 和 Pyannote 的音频处理系统，能够实现音频转录、多说话人分离和语言检测等功能。

## 功能特点

- 🎯 **音频转录**：使用 Whisper large-v3 模型进行高精度语音识别
- 👥 **说话人分离**：通过 Pyannote 实现多说话人识别和分离
- 🌍 **多语言支持**：自动检测音频语言
- ⚡ **GPU 加速**：支持 CUDA 加速，提升处理效率
- 🎯 **词级时间戳**：提供精确的词级别时间对齐
- 🔄 **灵活输入**：支持多种音频输入方式（URL、文件、Base64）

## 系统要求

- Python 3.8+
- CUDA 支持（推荐）
- FFmpeg
- HuggingFace 账号和 API Token

## 依赖项

- faster-whisper
- pyannote.audio
- torch
- torchaudio
- ffmpeg-python

## 安装说明

1. 克隆仓库

```
bash
git clone [repository-url]
```
2. 安装依赖

```
bash
pip install -r requirements.txt
```

3. 安装 FFmpeg（如果尚未安装）

## 使用方法

### 基本用法

```
python
from modules.predict import Predictor
predictor = Predictor()
predictor.setup(hf_token="your-huggingface-token")
```

# 处理音频

```
output = predictor.predict(file_url="your-audio-url")
```


### 输出格式

系统输出包含三个主要部分：
1. **说话人数量** (num_speakers)：整数值
2. **语言** (language)：检测到的语言代码
3. **片段信息** (segments)：JSON 格式的详细转录结果，包含：
   - 开始时间 (start)
   - 结束时间 (end)
   - 说话人标识 (speaker)
   - 转录文本 (text)
   - 词级信息 (words)：包含每个词的时间戳和置信度

```json
{
"start": 0.0,
"end": 2.5,
"speaker": "SPEAKER_00",
"text": "转录的文本内容",
"words": [
{
"start": 0.0,
"end": 0.5,
"word": "转录",
"probability": 0.98
},
...
]
}
```


## 高级配置

### 预测参数

- `group_segments`：是否合并相同说话人的相邻片段
- `transcript_output_format`：输出格式选项（"segments_only"/"words_only"/"both"）
- `num_speakers`：预设说话人数量（可选）
- `translate`：是否翻译为英语
- `language`：指定输入音频语言（可选）
- `prompt`：转录提示词（可选）

### 进度回调

系统支持进度回调功能，可以实时监控处理进度：

```python
def progress_callback(info):
print(f"Status: {info['status']}, Progress: {info['progress']}%")
predictor.setup(hf_token="token", progress_callback=progress_callback)
```


## 注意事项

1. 需要有效的 HuggingFace API Token
2. 首次运行时会下载模型，需要稳定的网络连接
3. 音频文件会被临时转换为 WAV 格式处理
4. GPU 模式下需要足够的显存（建议 8GB+）

## 错误处理

系统包含完整的错误处理机制，常见错误包括：
- HuggingFace Token 无效
- 音频文件下载失败
- 格式转换错误
- 内存/显存不足

## License

[指定您的许可证类型]

## 贡献指南

欢迎提交 Issue 和 Pull Request！

## 致谢

- [Whisper](https://github.com/openai/whisper)
- [Pyannote](https://github.com/pyannote/pyannote-audio)
