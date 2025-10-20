# ReBotNet – Fast Real-Time Video Enhancement

**Project:** Reproduction of the research paper [ReBotNet: Fast Real-time Video Enhancement (WACV 2025)](https://openaccess.thecvf.com/content/WACV2025/papers/Valanarasu_ReBotNet_Fast_Real-Time_Video_Enhancement_WACV_2025_paper.pdf)  

---

## Overview
This repository contains the **reproduction of the ReBotNet model** for real-time video restoration and enhancement. The project demonstrates high-quality enhancement of degraded video frames in real time using a transformer-based architecture.

---

## Dataset
- **Vimeo-90K dataset**  
- **Sequence length:** 7 frames  
- **Resolution:** 448 × 256  
- **Preprocessing:** Applied custom degradations (blur, noise, compression) using the **BasicSR** library to simulate real-world video distortions.

---

## Implementation
- **Model:** Reproduced using the **official architecture code** provided by the authors.  
- **Framework:** PyTorch  
- **Training environment:** Google Colab with **NVIDIA A100 GPU**  
- **Inference:** Real-time frame processing with side-by-side original and enhanced display

---
![Demo](demo.gif)
---

## Evaluation Metrics
- **PSNR:** Peak Signal-to-Noise Ratio  
- **SSIM:** Structural Similarity Index  
- **Latency:** Measured to verify real-time processing capability

---

## Features
- Real-time enhancement of degraded videos  
- Preprocessing pipeline for simulating realistic video distortions  
- GPU-accelerated training and inference  
- Reproducible results aligned with the original paper

---

## Usage
1. Clone the repository:
```bash
git clone https://github.com/hanshanaweerakoon/ReBotNet---Fast-Real-time-Video-Enhancement.git



