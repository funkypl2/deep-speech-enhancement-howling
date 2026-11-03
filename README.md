# Deep-Audio-Enhance-Suite: Deep Learning Speech Enhancement & Howling Suppression

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Algorithms: CRN + AFC](https://img.shields.io/badge/Algorithms-CRN%20%7C%20AFC%20%7C%20Notch-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Deep-Audio-Enhance-Suite** 是一个聚焦于**复杂声学环境下深度语音增强（Speech Enhancement）、非平稳噪声消除（Denoising）与声学正反馈啸叫抑制（Acoustic Feedback Cancellation & Howling Suppression, AFC）**的算法工程套件。

---

## 🔬 声学原理与算法体系

### 1. 啸叫形成机理与抑制 (Howling & Feedback Cancellation)
在会议室、直播音响及助听设备中，扬声器播放的声音再次进入麦克风，形成闭环声学反馈传递函数 $G(\omega)$。当系统满足 **奈奎斯特稳定判据**：
$$\left| G(\omega) \right| \ge 1 \quad \text{且} \quad \angle G(\omega) = 2k\pi$$
能量迅速在共振峰处雪崩式聚集，产生刺耳的高幅值啸叫（Whistling）。

本项目采用 **双轨混合啸叫抑制架构**：
- **前段快速检测 (PAPR + 频谱平坦度 SFM)**：毫秒级捕获窄带剧烈突增峰值，精确定位啸叫中心频率。
- **动态自适应陷波器 (Dynamic IIR Notch Filter Bank)**：自动更新零极点分布，精准切除共振尖峰而不伤及临近语音共振峰。
- **归一化最小均方自适应滤波 (NLMS)**：自适应估计主声学房间脉冲响应（RIR）线性路径。

### 2. 深度时频复数神经网络 (CRN / cRM)
- 采用 U-Net 拓扑的时频卷积循环网络（Convolutional Recurrent Network, CRN）。
- 编码器提取实部与虚部特征，中间 2 层双向/单向 LSTM 捕捉语音的时序上下文关联，解码器结合跳跃连接（Skip Connection）进行精准复数比值掩蔽（cRM）重建。

---

## 📊 评测指标 (Objective Metrics)

项目完整内置国际主流语音评测函数：
- **SI-SDR (Scale-Invariant Signal-to-Distortion Ratio)**：尺度不变信噪比，衡量语音保真度。
- **ERLE (Echo Return Loss Enhancement)**：回声/啸叫抑制衰减增益（dB）。
- **SNR (Signal-to-Noise Ratio)**：传统信噪比。

---

## 🚀 快速上手 (Quick Start)

### 1. 安装依赖
```bash
cd 03-deep-speech-enhancement-howling
pip install -r requirements.txt
```

### 2. 运行增强与啸叫抑制评测
脚本将自动合成包含高强度啸叫与低信噪比噪声的测试音频，并实时输出处理前后客观评测指标：
```bash
python infer.py --howling-freq 1250.0 --noise-snr 12.0
```

### 3. 运行自动化单元测试
```bash
pytest tests/ -v
```

---

## 📂 项目结构

```
03-deep-speech-enhancement-howling/
├── dataset/
│   └── synthetic_pipeline.py    # 加噪、啸叫合成仿真流水线
├── models/
│   ├── crn.py                   # 深度卷积循环网络 (CRN)
│   ├── howling_detector.py      # PAPR 啸叫检测器
│   └── afc_neural_filter.py     # NLMS 自适应滤波与动态陷波器
├── metrics/
│   └── evaluator.py             # SI-SDR, ERLE, SNR 评估实现
├── infer.py                     # 端到端推断与效果验证入口
├── requirements.txt
└── README.md
```

---

## 📄 开源许可证
本项目采用 [MIT License](LICENSE) 授权协议。
