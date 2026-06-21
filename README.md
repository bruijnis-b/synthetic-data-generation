# Synthetic Data Generation for Syringe Detection

## Description
This repository contains the source code and reproduction steps for generating synthetic images of syringes and training object detection models. The project evaluates two generative pipelines, StyleGAN3 and Stable Diffusion (SDXL LoRA), to generate synthetic dataset frames, which are then combined with real-world clinical data to train and optimize a YOLO11 Nano Oriented Bounding Box (OBB) model.


## Reproduction

#### Environment Setup

Because this study spans video pre-processing, generative model fine-tuning, and object detection training, different components use separate environments.

#### 1. Core Pipeline Environment (Data Processing & YOLO Training)
This environment is used for video frame extraction (`extract-frames.py`), dataset merging (`dataset-merger.py`), and YOLO11 OBB training/evaluation (`train.py`).

1. **Python Version:** Python 3.10+ is recommended.
2. **Setup:**
   ```bash
   # Create and activate a virtual environment
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

   # Install dependencies
   pip install -r requirements.txt
   ```

The libraries installed from `requirements.txt` include:
* `ultralytics`: For YOLO11 OBB object detection training and inference.
* `torch`: PyTorch framework (CUDA-enabled version recommended for GPU training).
* `opencv-python`: For image loading and basic processing operations.
* `decord`: For fast, video frame extraction.
* `pyyaml`: YAML parser for parsing and writing YOLO dataset configuration files.
* `matplotlib`: For visualization of training metrics and benchmarks.

#### 2. GAN Environment (StyleGAN3)
Fine-tuning StyleGAN3 requires a custom Conda environment owing to its strict dependency version requirements (e.g., specific PyTorch versions, `ninja` compiler, and CUDA tools).
* **Setup:**
  ```bash
  # Clone the StyleGAN3 repository
  git clone https://github.com/NVlabs/stylegan3.git
  cd stylegan3

  # Create and activate the conda environment
  conda env create -f environment.yml
  conda activate stylegan3

  # Install additional dependencies
  pip install click tqdm requests psutil scipy imageio-ffmpeg==0.4.3 pyspng ninja
  ```

#### 3. Diffusion Environment (LoRA Training & Inference)
Diffusion model training and image generation utilize standalone tools which run in their own environments:
* **LoRA Training (`kohya_ss`):** Set up using the [`kohya_ss`](https://github.com/bmaltais/kohya_ss) repository, which provides installation scripts to set up a python virtual environment and run the GUI interface.
* **Image Generation (`ComfyUI`):** Set up using [`ComfyUI`](https://comfy.org/) which runs either as a portable standalone application or within its own virtual environment.


### Data
Related datasets and configuration files can be found on this [Hugging Face Repository](https://huggingface.co/datasets/bruijnis-b/synthetic-data-generation).

- Raw videos for preprocessing (`\data_repo\real_frames\raw` of the HF repo)
  - Can be placed under a custom local directory (e.g., `data/raw/`).
  - **Note:** Run the frame extraction script `src/data-processing/extract-frames.py` using CLI arguments to point to your video folder and output directory (see [Phase 1: Data Processing](#phase-1-data-processing) below).

- Labeled datasets for dataset preparation
  - Can be placed under `data/`
  - **Note:** The dataset merger tool (`src/model-integration/data/dataset-merger.py`) expects the input datasets to be organized in subfolders within `data/` (e.g., `data/diffusion_v1/`, `data/stylegan_v1/`, `data/baseline_real/`). Each of these subfolders should contain the images and their corresponding YOLO label files. You can download these labeled datasets from the Hugging Face repository linked above (e.g. `\data_repo\stylegan3\v2\processed` or `\data_repo\real_frames\train_frames-filtered-labeled`).

- Datasets for object detection training (`\data_repo\object-detection-datasets`)
  - Can be placed under `src/model-integration/data/`
  - **Note:** The dataset merger tool will create new merged datasets in this directory based on the specified combinations.


### Phase 1: Data Processing
The data pre-processing for this project entails extracting the frames from the original videos. For this, a helper script was made: `src/data-processing/extract-frames.py`.

#### Usage
To extract frames, run the script from the project root using the following command-line arguments:

```bash
python src/data-processing/extract-frames.py --video_folder <path-to-your-video-folder> --frames_dir <path-to-your-frames-output-folder>
```

#### CLI Arguments
| Argument | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `--video_folder` | string | **Yes** | Path to the folder containing the `.mp4` video files to process. |
| `--frames_dir` | string | **Yes** | Path to the folder where extracted frames will be saved. A subfolder is created for each video. |
| `--combined_dir` | string | No | Path to save all frames flattened into a single folder. Defaults to `<frames_dir>_combined`. |
| `--frame_rate` | int | No | Extract every N-th frame. Default is `5` (extracts 1 image every 5 frames). |
| `--overwrite` | flag | No | If specified, overwrites existing frames in the destination directory. |


### Phase 2: Image Generation

#### 2.1 GANs
For GAN-based image generation, the base model StyleGAN3 was used. This model was fine-tuned on the filtered train set. Because this is a computationally expensive process, it was conducted using cloud computing with a GPU containing at least 24GB of VRAM (e.g., NVIDIA RTX 4090).

To optimize performance and reduce computational cost, training was performed on 256x256 cropped images rather than 1024x1024.

The steps for reproducing the GAN training:
1. Use a GPU with at least 12GB of memory (24GB recommended for training).
2. Ensure you have activated the StyleGAN3 conda environment (see [Environment Setup](#environment-setup)).
3. Place the training frames (`\data_repo\real_frames\train_frames-filtered`) in a local folder (e.g., `data/train_frames/`).
4. Clone and set up the StyleGAN3 repository.
5. Commands for training:
   * **Data Processing:** Crop the images to square dimensions to retain aspect ratios while formatting for GAN training:
     ```bash
     python dataset_tool.py --source="/path/to/train_frames" --dest="/path/to/output/syringes_256.zip" --resolution=256x256 --transform=center-crop
     ```
   * **Training:**
     ```bash
     python train.py --outdir="/path/to/output/training-runs" --data="/path/to/output/syringes_256.zip" --gpus=1 --cfg=stylegan3-r --cbase=16384 --mirror=1 --batch=32 --gamma=2 --snap=15 --kimg=15120 --metrics=none --aug=ada --freezed=10 --resume=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhqu-256x256.pkl
     ```
     Various parameters are set:
       * `outdir`: The directory where the training script will save its progress. As it runs, it will create a new subfolder here containing log files, sample images (to visually check progress), and the saved model weights (.pkl files).
       * `data`: The path to the .zip file containing the training data (the cropped images) created in the processing step.
       * `snap`: The snapshot interval. This flag tells the script to save a checkpoint (a .pkl file and a grid of sample images) every 15 ticks (where a tick is 4 `kimg`s, so a tick is 4000 images).
       * `gpus`: Tells the script to use a single GPU for training.
       * `batch`: The global batch size. This means the model will look at 32 images at a time before updating its internal weights.
       * `cfg`: Selects the specific network architecture. StyleGAN3 has two main variants: `stylegan3-t` (translation-equivariant) and `stylegan3-r` (rotation-equivariant). The `-r` version handles unaligned datasets well.
       * `mirror`: Enables x-axis (horizontal) mirroring. During training, the script will randomly flip the images left-to-right.
       * `gamma`: Controls the strength of R1 regularization applied to the discriminator, preventing it from overpowering the generator and causing training to collapse.
       * `resume`: The path or URL to a pre-trained model. Transfer learning is performed starting from NVIDIA's pre-trained model (trained on the FFHQ-Unaligned dataset of human faces at 256x256).
       * `kimg`: The number of thousands of images to process. The v1 model training was stopped early (at ~2,500 kimg) once the visual quality stabilized. For the v2 model, to fully observe the effects of the frozen layers, training was run significantly longer: 24 hours on an NVIDIA RTX 4090 (24 GB VRAM), processing 3,780 ticks, which equates to a total of 15,120 kimg.
       * `metrics`: Disables the FID scoring loop to save GPU time on a small dataset.
       * `aug`: Set to `ada` to enable Adaptive Discriminator Augmentation (ADA).
       * `freezed`: Locks the first 10 resolution blocks of the generator and discriminator, protecting the base model's representation and allowing only the final layers to adapt.
       * `cbase`: Capacity base multiplier. A lower value (e.g., `16384` instead of the default `32768`) reduces the channel count to shrink VRAM usage and training time.

**Inference** is executed using the `gen_images.py` script inside the StyleGAN3 repository:

  ```bash
  python gen_images.py --outdir=/path/to/output/v1 --seeds=0-499 --trunc=0.7 --network=/path/to/models/{model_name.pkl}
  ```
    * **Parameters:**
      * `network`: Points to the .pkl file containing the trained model weights.
      * `outdir`: The destination folder where the script will save the newly generated images.
      * `seeds`: Slices of seeds (e.g. 0-499 to generate 500 unique images). Each seed corresponds to a unique point in latent space.
      * `trunc`: Controls the trade-off between image quality and diversity. A value of 0.7 is the standard default.

The exact models, outputs, training options and generated images for this study can be found in the Hugging Face repository linked above (`\data_repo\stylegan3\v_`).

#### 2.2 Diffusion

##### 2.2.1 LoRA Training
For the LoRA training, the `kohya_ss` tool was used.

**kohya_ss** setup:
1. Organize and copy the images and corresponding `.txt` labels to a local directory (e.g., `/path/to/dataset`).
   * The folder name should follow the structure `n_triggerword_description` (e.g. `20_bbrp_syringes_v1`). The number at the beginning specifies how many times each image in that folder is repeated in an epoch (e.g. 20 times for a small dataset).
2. Download the base model and VAE:
   * Base model (SDXL): `wget https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors`
   * VAE: `wget https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl_vae.safetensors`
   * Place these files under the `models/` directory in the tool structure.
3. Configure `kohya_ss`:
   * Upload the `kohya_config.json` configuration file inside the LoRA training GUI.
   * Click on "Start Training". The outputs (trained weights) will be saved in your specified output directory (e.g., `kohya_ss/output`).

The data used for the LoRA training, the config, and the generated LoRA weights can be found in the Hugging Face repository under `\data_repo\sdxl\LoRA`.

##### 2.2.2 Inference
For generating images using the trained LoRA weights, the **ComfyUI** tool was used.

**ComfyUI** setup:
1. Download/place the base model inside `ComfyUI/models/checkpoints/`
2. Download/place the VAE inside `ComfyUI/models/vae/`
3. Copy the trained LoRA `.safetensors` file into `ComfyUI/models/lora/`
4. Run ComfyUI, load the provided ComfyUI workflow `.json` file, set the batch/generation size, and click **Queue Prompt** (or "Run"). The generated images are saved in `ComfyUI/output/`.

The workflows and outputs are saved in the Hugging Face repository under `\data_repo\sdxl\v_`.

### Phase 3: Object Detection Model Training

#### 3.1 Dataset Preparation (Dataset Merger Tool)
The `dataset-merger.py` tool was written for creating complex, multi-source datasets for YOLO training. It allows you to mix data from different sources with varying ratios easily.

##### Why use the Merger?
- **Domain Mixing:** Train on a combination of different styles/sources to improve model generalization.
- **Controlled Exposure:** Precisely partition what percentage of a source enters the training set versus the evaluation sets.
- **Collision Prevention:** Prefixes merged filenames with their source folder name (e.g., `diffusion_v1_syringe_001.png`) to avoid name collisions.
- **Leakage Protection:** Automatically partitions overlapping sources sequentially to ensure training images do not leak into validation or test sets.

##### Command Line Interface (CLI)
The script is located at `src/model-integration/data/dataset-merger.py`.

| Argument | Type | Description |
| :--- | :--- | :--- |
| `--output` | string | Name of the output folder in `src/model-integration/data/`. |
| `--train` | string... | List of sources in `name:ratio` format (e.g. `diffusion_v1:1.0`). |
| `--val` | string... | List of validation sources in `name:ratio` format. |
| `--test` | string... | List of testing sources in `name:ratio` format. |
| `--seed` | int | Random seed for shuffling and partitioning. Default: `42`. |

###### Usage Examples
**1. The "Master Mix"** (Combine 100% of synthetic sources with 5% of real data for training, reserving the rest of the real data for validation/testing):
```powershell
python src/model-integration/data/dataset-merger.py --output master_mix_v1 `
    --train diffusion_v1:1.0 stylegan_v1:1.0 baseline_real:0.05 `
    --val baseline_real:0.475 `
    --test baseline_real:0.475
```

**2. Comparative Ablation** (Train only on 50% of the StyleGAN data, using real data for validation/testing):
```powershell
python src/model-integration/data/dataset-merger.py --output ablation_gan_50 `
    --train stylegan_v1:0.5 `
    --val baseline_real:0.5 `
    --test baseline_real:0.5
```

##### Technical Features
* **YOLO Directory Structure:** Organized into standard splits:
  ```text
  src/model-integration/data/<output_name>/
  ├── dataset.yaml
  ├── images/ (train/, val/, test/)
  └── labels/ (train/, val/, test/)
  ```
* **File Pairing:** Verifies that every image has a matching `.txt` label file (unpaired files are ignored).
* **Automatic `dataset.yaml`:** Generates a `dataset.yaml` with resolved absolute paths, enabling training execution from any working directory.
* **Coordinate Sanitization:** Clamps YOLO label coordinates to the valid `[0.0, 1.0]` range with 7-decimal precision, preventing training warnings or skipped frames due to boundary anomalies.
* **Non-Overlapping Partitioning:** Shuffles a source once with the seed and slices it sequentially, ensuring zero data leakage when a single source is reused across training and testing splits.


#### 3.2 YOLO Training
The object detection experiments were conducted using the **YOLO11 Nano** model configured for **Oriented Bounding Box (OBB)** detection (`yolo11n-obb.pt`), leveraging the **Ultralytics** framework. YOLO (You Only Look Once) is a state-of-the-art, single-stage object detection architecture known for its speed and accuracy, predicting bounding box coordinates, orientations, and class probabilities directly from full images in a single pass—making it highly suitable for dynamic clinical environments.

##### Training Command
To train the YOLO model using the benchmark suite (which iterates over the configured dataset experiments, registers callbacks, trains the models, and validates them on the test set), run the following command from the project root:

```bash
python src/model-integration/models/yolo/train.py
```

##### Training Configuration
The model was trained iteratively across various dataset combinations, comprising baseline real images and synthetically generated frames from both the StyleGAN and Diffusion pipelines.

* **Training Duration:** Standardized to `50` epochs per dataset configuration.
* **Batch Size:** `32`.
* **Resolution:** Input images were uniformly resized to `640x640` pixels (chosen to balance fine detail retention with manageable VRAM utilization, as training at the native `1024x1024` resolution proved computationally prohibitive).
* **Hardware & Efficiency:**
  * **Precision:** Half-precision (`FP16`) training enabled.
  * **Acceleration:** Executed on a CUDA-capable GPU (GPU is strictly required by the script).
  * **Workers:** `4` data-loading workers.
* **Optimization:** Governed by default Ultralytics hyperparameters (typically an initial learning rate of `0.01` via an auto-selected optimizer).

##### Custom Callback & Data Augmentation
To encourage model robustness and penalize over-reliance on sharp, high-frequency generative artifacts, a targeted data augmentation strategy was implemented using Ultralytics' callback system:
* **Custom Callback:** Injected at the start of every training batch (`on_train_batch_start`).
* **Dynamic Blur:** Dynamically applies a **Gaussian blur** using a kernel size matching the blur strength of `7.0` (which resolves to a `15x15` kernel).

Below is the code snippet from `src/model-integration/models/yolo/train.py` illustrating how the custom callback is defined, registered, and applied to the training batches:

```python
import cv2
import torch
import numpy as np
from ultralytics import YOLO

# 1. Define the Gaussian blur callback generator
def make_blur_callback(strength):
    def on_train_batch_start(trainer):
        if hasattr(trainer, 'batch') and trainer.batch is not None:
            imgs = trainer.batch['img']
            for j in range(len(imgs)):
                # Convert tensor back to numpy image (HWC format, [0, 255])
                img = imgs[j].cpu().numpy().transpose(1, 2, 0)
                img = (img * 255).astype(np.uint8)
                
                # Apply Gaussian Blur (kernel size must be odd)
                k = int(strength) * 2 + 1
                img = cv2.GaussianBlur(img, (k, k), 0)
                
                # Normalize and convert back to PyTorch tensor (CHW format)
                img = img.astype(np.float32) / 255.0
                imgs[j] = torch.from_numpy(img.transpose(2, 0, 1))
            
            trainer.batch['img'] = imgs
    return on_train_batch_start

# 2. Load model and register callback prior to training
model = YOLO("yolo11n-obb.pt")
model.add_callback("on_train_batch_start", make_blur_callback(strength=7.0))

# 3. Launch training
results = model.train(
    data="src/model-integration/data/stylegan_v1_50_50_real/dataset.yaml",
    epochs=50,
    imgsz=640,
    device=0,
    batch=32,
    half=True,
    workers=4
)
```


## Author
This repository was created by Bas Bruijnis ([![Github Badge](https://img.shields.io/badge/-bruijnis--b-24292e?style=flat&logo=Github)](https://github.com/bruijnis-b)), as part of the Bachelor Research Project (CSE3000) at [Delft University of Technology](https://www.tudelft.nl/).

The project was supervised by Nergis Tömen, Xucong Zhang.
