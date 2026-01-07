# DDSF-UNet for Underwater Image Enhancement

This repository implements **DDSF_UNet** for enhancing underwater images. The model restores degraded underwater images to visually appealing, high-quality reference images.  


## Requirements

- Python: 3.10.18
- PyTorch: 2.7.1+cu128
- CUDA: 12.8
- cuDNN: 9.7.1
- torchvision
- scikit-image
- numpy
- tqdm
- Pillow

All experiments were conducted on a Linux-based system with NVIDIA GPU support.


Install dependencies via pip

```bash
pip install torch torchvision scikit-image numpy tqdm pillow
```

## Dataset Structure
For training, prepare a paired dataset of degraded and reference images:

```bash
dataset/
├── train/
│   ├── raw/       # degraded images
│   └── ref/       # reference images
```


For testing/inference, provide a folder of raw images and optionally reference images:

```bash
test_data/
├── raw_test/       # degraded test images
├── ref_test/       # ground-truth reference images (optional for metrics)
```

Filenames must match between raw and reference images for paired evaluation.
Training



## Pretrained Checkpoints and Sample Enhanced Images
You can download pretrained models and DDSF-UNet enhanced outputs from Google Drive:

- **Pretrained Checkpoints:** (https://drive.google.com/drive/folders/1Nyr3ZaRoY91pLqAOK5lBSxX11urhlD81?usp=sharing)
- **Sample Enhanced Images:** (https://drive.google.com/drive/folders/1Nyr3ZaRoY91pLqAOK5lBSxX11urhlD81?usp=sharing)



## Run the training script:
```bash
python train.py \
    --raw_dir /path/to/raw_images \
    --ref_dir /path/to/ref_images \
    --epochs 200 \
    --batch_size 2 \
    --crop 256 \
    --save_dir ./checkpoints
```


## Testing / Inference

Run the testing script:

```bash
python test.py \
    --ckpt_path ./checkpoints/best.pth.tar \
    --raw_test_dir /path/to/raw_test \
    --ref_test_dir /path/to/ref_test \
    --out_dir ./test_results \
```


The script will:
- Load the trained model checkpoint.
- Pair raw images with reference images.
- Perform inference and save enhanced images to --out_dir.
- Compute PSNR and SSIM if reference images are provided.


## Evaluation Metrics
- PSNR (Peak Signal-to-Noise Ratio): measures similarity in intensity.
- SSIM (Structural Similarity): measures perceptual similarity in structure.


## Checkpoints
- latest.pth – latest epoch checkpoint.
- best.pth – best performing model checkpoint (lowest training loss).
- Periodic checkpoints (every N epochs) for safety.


## Citation
```bibtex
@article{DDSF-UNet2025,
  title= {DDSF-UNet: A Dual-Domain Spatial–Frequency UNet for Underwater Image Enhancement},
  author={Najmul Hassan, Munsif Ali, Abu Saleh Musa Miah, Jungpil Shin},
  journal={XYZ},
  year={2026},
}
```


## Acknowledgements

We would like to thank the authors of the following projects for their inspiration and resources:

- [AQUA-Net: Adaptive Frequency Fusion and Illumination Aware Network for Underwater Image Enhancement](https://munsifali11.github.io/AQUA-Net_Project/)
- [Ucolor: Underwater Image Enhancement via Medium Transmission-Guided Multi-Color Space Embedding](https://li-chongyi.github.io/Proj_Ucolor.html)  
- [OUNet_JL: UOptimized UNet framework with a joint loss function for underwater image enhancement](https://github.com/WangXin81/OUNet_JL)

