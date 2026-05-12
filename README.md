# Diffusion-Based Image Super-Resolution: A Comparative Study

This project presents a comparative evaluation of generative models for Single Image Super-Resolution (SISR). We analyze the performance of two diffusion-based paradigms (SR3-style DDPM and its accelerated variant DDIM) against a state-of-the-art GAN-based baseline (ESRGAN). 

The study focuses on the trade-offs between reconstruction fidelity, perceptual quality, and computational inference costs within a constrained training environment.

## Project Overview

Super-resolution is the task of recovering high-resolution (HR) images from low-resolution (LR) inputs. This project benchmarks three architectures:

1.  **ESRGAN (Enhanced Super-Resolution GAN):** A feed-forward generator optimized for perceptual quality using Residual-in-Residual Dense Blocks (RRDB) and a relativistic discriminator.
2.  **DDPM (SR3-style):** A conditional Denoising Diffusion Probabilistic Model that iteratively refines Gaussian noise into high-resolution outputs guided by the low-resolution input.
3.  **DDIM (Denoising Diffusion Implicit Models):** An accelerated sampler for the DDPM backbone that utilizes deterministic trajectories to reduce sampling steps.

## Key Features

- **Training Pipeline:** Built for Kaggle (T4/P100 GPUs) using PyTorch with Automatic Mixed Precision (AMP).
- **Dataset:** Trained on the 800-image DIV2K dataset. Evaluated on Set5, Set14, and BSD100 benchmarks.
- **Metrics:** Comprehensive evaluation using PSNR, SSIM, LPIPS, and inference latency.
- **Checkpointing:** Native support for epoch-wise checkpointing and resumption to handle Kaggle's session limits.

## Results

### Quantitative Evaluation (4x Upscaling)

| Model | Dataset | PSNR (dB) | SSIM | LPIPS | Time/Img (s) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ESRGAN** | Set5 | 27.76 | 0.8298 | 0.2213 | 0.06 |
| **DDPM** | Set5 | 15.05 | 0.3771 | 0.6390 | 157.62 |
| **DDIM** | Set5 | 11.36 | 0.1978 | 0.9798 | 7.92 |
| | | | | | |
| **ESRGAN** | Set14 | 25.21 | 0.7254 | 0.3164 | 0.12 |
| **DDPM** | Set14 | 15.79 | 0.3508 | 0.6515 | 361.44 |
| **DDIM** | Set14 | 12.89 | 0.2072 | 0.9529 | 18.07 |
| | | | | | |
| **ESRGAN** | BSD100 | 24.96 | 0.6893 | 0.3866 | 0.07 |
| **DDPM** | BSD100 | 16.65 | 0.3392 | 0.6724 | 220.14 |
| **DDIM** | BSD100 | 13.30 | 0.2092 | 0.8908 | 10.98 |

### Summary of Findings

- **Efficiency:** ESRGAN operates in real-time, outperforming diffusion samplers by several orders of magnitude in speed while maintaining superior pixel fidelity (PSNR).
- **Perceptual Quality:** While ESRGAN leads across all metrics, the relative gap in LPIPS is smaller than in PSNR, suggesting diffusion models preserve meaningful perceptual structures despite higher pixel error.
- **Complexity:** The "Dataset Complexity Gradient" shows that the gap between GANs and Diffusion models shrinks as image content becomes more complex and heterogeneous (e.g., BSD100 vs Set5).
- **DDIM Failure Mode:** In the conditional super-resolution setting, DDIM deterministic sampling with low step counts (50 steps) significantly underperforms full DDPM (1000 steps).

## Project Structure

- `main_project.py`: Main execution script containing model definitions, training loops, and evaluation logic.
- `ablation_results.csv`: Data from the diffusion sampler ablation study.

## Installation and Requirements

The project is optimized for execution in a Kaggle environment.

```bash
pip install torch torchvision torchmetrics lpips pandas tqdm matplotlib
```

Dependencies include:
- PyTorch 2.0+
- Torchvision
- LPIPS (Learned Perceptual Image Patch Similarity)
- Torchmetrics

## Team Members

- Shaharyar Rizwan
- Ahmed Ali Zahid
- Naveed Ahmed Bajwa

## Acknowledgments

This project was developed as part of a Generative AI course. Training utilized Kaggle's GPU accelerators.
