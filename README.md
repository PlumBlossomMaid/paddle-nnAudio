# paddle-nnAudio

paddle-nnAudio 是基于 PaddlePaddle 卷积神经网络作为后端的音频处理工具箱。通过这种方式， 梅尔频谱可以在神经网络训练过程中实时从音频生成，并且傅里叶核（如 CQT 核）可以被训练。paddle-nnAudio 移植自 [nnAudio](https://github.com/KinWaiCheuk/nnAudio)，旨在为 PaddlePaddle 生态提供类似的音频处理能力。

## 安装
```bash
pip install git+https://github.com/你的用户名/paddle-nnAudio.git#subdirectory=Installation
```
或
```bash
pip install ppaudio
```

## 快速开始
```python
from ppAudio import features
from scipy.io import wavfile
import paddle

sr, song = wavfile.read('./Bach.wav')  # 加载音频
x = song.mean(1)  # 将立体声转换为单声道
x = paddle.to_tensor(x).cast('float32')  # 将数组转换为 PaddlePaddle Tensor

# 初始化模型
spec_layer = features.STFT(n_fft=2048, freq_bins=None, hop_length=512,
                              window='hann', freq_scale='linear', center=True, pad_mode='reflect',
                              fmin=50, fmax=11025, sr=sr)

spec = spec_layer(x)  # 将波形前向传播以获取 spectrogram
```

## 依赖项
- Numpy >= 1.14.5
- Scipy >= 1.2.0
- PaddlePaddle >= 2.0.0
- Python >= 3.6
- librosa = 0.7.0

## 引用
如果您使用了 paddle-nnAudio，请引用原 nnAudio 的论文：

K. W. Cheuk, H. Anderson, K. Agres and D. Herremans, "nnAudio: An on-the-Fly GPU Audio to Spectrogram Conversion Toolbox Using 1D Convolutional Neural Networks," in IEEE Access, vol. 8, pp. 161981-162003, 2020, doi: 10.1109/ACCESS.2020.3019084.

### BibTex
```
@ARTICLE{9174990,
  author={K. W. {Cheuk} and H. {Anderson} and K. {Agres} and D. {Herremans}},
  journal={IEEE Access}, 
  title={nnAudio: An on-the-Fly GPU Audio to Spectrogram Conversion Toolbox Using 1D Convolutional Neural Networks}, 
  year={2020},
  volume={8},
  number={},
  pages={161981-162003},
  doi={10.1109/ACCESS.2020.3019084}}
```

## 许可证
[MIT License](LICENSE)