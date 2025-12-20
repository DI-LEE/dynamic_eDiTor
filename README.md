<p align="center">
  <h1 align="center"><strong>Dynamic‑eDiTor: Training‑Free Text‑Driven 4D Scene Editing with Multimodal Diffusion Transformer</strong></h1>
</p>


<p align="center">
  Dong In Lee<sup>1,2*</sup>, Hyungjun Doh<sup>1*</sup>, Seunggeun Chi<sup>1</sup>, Runlin Duan<sup>1</sup>,<br>
  Sangpil Kim<sup>2†</sup>, Karthik Ramani<sup>1†</sup><br><br>
  <sup>1</sup>Purdue University, &nbsp; <sup>2</sup>Korea University
</p>

<div align="center">
  <a href="https://www.arxiv.org/abs/2512.00677">
    <img src="https://img.shields.io/badge/arXiv-2512.00677-red?logo=arxiv" alt="arXiv Badge">
  </a>
  <a href="https://di-lee.github.io/dynamic-eDiTor/">
    <img src="https://img.shields.io/badge/Project-Page-blue?logo=website" alt="Project Page">
  </a>
</div>



## ⚙️ Installation

Tested on Python 3.10 + CUDA 12.1.

```bash
conda create -n editor python=3.10
conda activate editor

# CUDA 12.1
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 xformers --index-url https://download.pytorch.org/whl/cu121
pip install diffusers==0.35.1 transformers==4.55.4 accelerate==1.10.1
pip install "huggingface-hub>=0.34.0,<1.0"
pip install bitsandbytes peft

cd dynamic_editor
# Method-specific dependencies
pip install -r requirements_multiview.txt   # for multi-view scenes
pip install -r requirements_mono.txt       # for monocular scenes
```

If you encounter:

```bash
ImportError: libGL.so.1: cannot open shared object file: No such file or directory
```

Install system dependency:

```bash
sudo apt-get install -y libgl1
```

## 📂 Datasets and Pre‑trained Scenes

- **Multi‑view (DyNeRF)**: Download scenes and reconstruct with official 4DGS.

  - DyNeRF dataset: https://github.com/facebookresearch/Neural_3D_Video/releases/tag/v1.0
  - 4D Gaussian Splatting (4DGS): https://github.com/hustvl/4DGaussians

- **Monocular (DyCheck)**: We provide a pre‑trained scene:

  ```text
  src/Deformable-3D-Gaussians/output/mochi-high-five
  ```

Recommended layout:

```text
data/
 ├─ DyCheck/
 └─ DyNeRF/
dynamic_editor/
```

## 🎨 Editing

1) Update all local paths in `script/run_editor.sh`.  
2) Run editing + optimization + rendering:

```bash
cd script
bash run_editor.sh
```

For ~48GB GPUs (e.g., RTX A6000), enable local caching:

```bash
cd script
bash run_editor_local.sh
```

### Monocular Scenes

Update paths in `src/Deformable-3D-Gaussians/script/run.sh`:

- `DATA_DIR`: path to the data for scene reconstruction
- `BASE_OUTPUT_NAME`: pre‑trained scene name (e.g., "mochi-high-five")
- `BASE_OUTPUT_ROOT`: path to the pre‑trained scene

Run editing + optimization + rendering:

```bash
cd src/Deformable-3D-Gaussians/script
bash run.sh
```

## 🎬 Rendering

All commands above include a visualization/rendering step. After completion, inspect the generated results under the corresponding output directories created by each script. You can render additional views using the rendering utilities provided by 4DGS or the scripts supplied in this repository.

## 🧰 Tips

- Ensure your dataset is correctly pre‑processed with 4DGS (multi‑view) or the provided monocular setup.  
- Use the local‑caching script on 48GB GPUs to avoid OOM.  
- Allocate sufficient local storage if caching is enabled.

## 📜 Citation

If you find this work useful, please cite:

```bibtex
@article{lee2025dynamiceditor,
  title   = {Dynamic-eDiTor: Training-Free Text-Driven 4D Scene Editing with Multimodal Diffusion Transformer},
  author  = {Lee, Dong In and Doh, Hyungjun and Chi, Seunggeun and Duan, Runlin and Kim, Sangpil and Ramani, Karthik},
  journal = {arXiv preprint arXiv:2512.00677},
  year    = {2025}
}
```

## 🙏 Acknowledgements

We thank the authors and contributors of 4DGS, DyNeRF, Diffusers, and related open‑source projects that made this work possible.
