# EEG-to-Image-Baseline


Specialized code to reproduce "Visual Decoding and Reconstruction via EEG Embeddings with Guided Diffusion", keeping only the EEG-to-image part under the in-subject setting, removing the retrieval code and diffusion prior code.

Eval results:

| Dataset | Low-level | | High-level | | | | |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| | **PixCorr** ↑ | **SSIM** ↑ | **AlexNet(2)** ↑ | **AlexNet(5)** ↑ | **Inception** ↑ | **CLIP** ↑ | **SwAV** ↓ |
| paper | 0.160 | **0.345** | **0.776** | **0.866** | **0.734** | **0.786** | **0.582** |
| **this Run** | **0.158** | **0.347** | **0.762** | **0.796** | **0.662** | **0.680** | **0.633** |

Environment Setup:

Reference to original repository: https://github.com/dongyangli-del/EEG_Image_decode