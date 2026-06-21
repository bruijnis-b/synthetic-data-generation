import csv
import gc
import torch
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pathlib import Path

def plot_training_metrics(results_csv_path, save_dir, title_prefix=""):
    """Reads YOLO's results.csv and saves a plot of the training/validation loss and mAP50."""
    if not os.path.exists(results_csv_path):
        return

    epochs = []
    train_loss = []
    val_loss = []
    map50 = []

    try:
        with open(results_csv_path, 'r') as f:
            reader = csv.DictReader(f)
            headers = [h.strip() for h in reader.fieldnames] if reader.fieldnames else []
            reader.fieldnames = headers

            for row in reader:
                try:
                    ep_str = row.get('epoch') or row.get(' epoch')
                    if ep_str is None: continue
                    ep = int(ep_str.strip())
                    
                    # Extract Train Loss
                    t_val = None
                    for key in ['train/box_loss', 'train/box_om', 'train/loss']:
                        val = row.get(key)
                        if val and val.strip():
                            t_val = float(val.strip())
                            break
                    
                    # Extract Validation Loss
                    v_val = None
                    for key in ['val/box_loss', 'val/box_om', 'val/loss']:
                        val = row.get(key)
                        if val and val.strip():
                            v_val = float(val.strip())
                            break

                    # Extract mAP50 (Accuracy)
                    m50_val = None
                    for key in ['metrics/mAP50(B)', 'metrics/mAP50']:
                        val = row.get(key)
                        if val and val.strip():
                            m50_val = float(val.strip())
                            break

                    epochs.append(ep)
                    train_loss.append(t_val)
                    val_loss.append(v_val)
                    map50.append(m50_val)
                except (ValueError, KeyError, AttributeError):
                    continue

        if not epochs:
            return

        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Plot Loss on primary y-axis
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color='tab:red')
        if any(x is not None for x in train_loss):
            ax1.plot(epochs, [x for x in train_loss], label='Train Loss', color='red', marker='o', alpha=0.7)
        if any(x is not None for x in val_loss):
            ax1.plot(epochs, [x for x in val_loss], label='Val Loss', color='orange', marker='x', alpha=0.7)
        ax1.tick_params(axis='y', labelcolor='tab:red')
        ax1.grid(True, linestyle='--', alpha=0.6)

        # Create a second y-axis for mAP50
        if any(x is not None for x in map50):
            ax2 = ax1.twinx()
            ax2.set_ylabel('mAP50 (Accuracy)', color='tab:blue')
            ax2.plot(epochs, [x for x in map50], label='mAP50', color='blue', marker='s', alpha=0.8)
            ax2.tick_params(axis='y', labelcolor='tab:blue')
            ax2.set_ylim(0, 1.05)

        plt.title(f'{title_prefix} - Training Progress')
        fig.tight_layout()
        
        plot_path = os.path.join(save_dir, "training_progress.png")
        plt.savefig(plot_path)
        plt.close()
        # print(f"  [Info] Updated training progress plot at {plot_path}")

    except Exception:
        pass

def make_plotting_callback(ds_name, aug_name):
    """Creates a callback that plots metrics at the end of each epoch."""
    def on_train_epoch_end(trainer):
        internal_csv = os.path.join(trainer.save_dir, "results.csv")
        plot_training_metrics(internal_csv, trainer.save_dir, title_prefix=f"{ds_name} ({aug_name})")
    return on_train_epoch_end

def find_project_root():
    """Finds the project root by looking for the 'data' directory."""
    current = Path(__file__).resolve().parent
    for _ in range(5):
        if (current / "data").exists() and (current / "src" / "model-integration").exists():
            return current
        current = current.parent
    return Path(__file__).resolve().parent.parent.parent.parent.parent

def plot_benchmark_results(csv_path, save_dir):
    """Plots the results from benchmark_results.csv as a line chart."""
    if not os.path.exists(csv_path):
        print(f"[Warning] Cannot plot benchmark results: {csv_path} not found.")
        return
        
    datasets = []
    map50 = []
    map50_95 = []
    
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            # Handle potential spaces in headers
            headers = [h.strip() for h in reader.fieldnames] if reader.fieldnames else []
            reader.fieldnames = headers
            
            for row in reader:
                try:
                    ds = row.get('dataset', 'unknown').strip()
                    m50 = float(row['mAP50'])
                    m95 = float(row['mAP50_95'])
                    datasets.append(ds)
                    map50.append(m50)
                    map50_95.append(m95)
                except (ValueError, KeyError, AttributeError):
                    continue
                    
        if not datasets:
            print("[Warning] No valid data found in benchmark_results.csv to plot.")
            return
            
        plt.figure(figsize=(12, 6))
        plt.plot(datasets, map50, marker='o', linestyle='-', label='mAP50')
        plt.plot(datasets, map50_95, marker='s', linestyle='--', label='mAP50-95')
        
        plt.xlabel('Dataset')
        plt.ylabel('Score')
        plt.title('Benchmark Results: mAP across Datasets')
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plot_path = os.path.join(save_dir, "benchmark_results_plot.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved benchmark results plot to {plot_path}")
    except Exception as e:
        print(f"[Warning] Failed to plot benchmark results: {e}")

if __name__ == '__main__':

    experiments = [
        # {"dataset": "baseline_real", "model": "yolo11n-obb.pt", "augmentation": "blur", "blur": 7.0},
        # {"dataset": "stylegan_v1", "model": "yolo11n-obb.pt", "augmentation": "blur", "blur": 7.0},
        # {"dataset": "stylegan_v2", "model": "yolo11n-obb.pt", "augmentation": "blur", "blur": 7.0},
        # {"dataset": "diffusion_v1", "model": "yolo11n-obb.pt", "augmentation": "blur", "blur": 7.0},
        # {"dataset": "diffusion_v2", "model": "yolo11n-obb.pt", "augmentation": "blur", "blur": 7.0},
        # {"dataset": "diffusion_v3", "model": "yolo11n-obb.pt", "augmentation": "blur", "blur": 7.0},
        # {"dataset": "stylegan_v1_50_real", "model": "yolo11n-obb.pt", "augmentation": "blur", "blur": 7.0},
        # {"dataset": "stylegan_v1_real", "model": "yolo11n-obb.pt", "augmentation": "blur", "blur": 7.0},
        # {"dataset": "diffusion_v3_50_real", "model": "yolo11n-obb.pt", "augmentation": "blur", "blur": 7.0},
        # {"dataset": "diffusion_v3_real", "model": "yolo11n-obb.pt", "augmentation": "blur", "blur": 7.0},
        {"dataset": "stylegan_v1_50_50_real", "model": "yolo11n-obb.pt", "augmentation": "blur", "blur": 7.0},
        {"dataset": "diffusion_v3_50_50_real", "model": "yolo11n-obb.pt", "augmentation": "blur", "blur": 7.0},
    ]

    ROOT = find_project_root()
    # Central results file for the whole run
    RESULTS_CSV = ROOT / "src" / "model-integration" / "models" / "yolo" / "benchmark_results.csv"

    def make_blur_callback(strength):
        def on_train_batch_start(trainer):
            if hasattr(trainer, 'batch') and trainer.batch is not None:
                imgs = trainer.batch['img']
                for j in range(len(imgs)):
                    img = imgs[j].cpu().numpy().transpose(1, 2, 0)
                    img = (img * 255).astype(np.uint8)
                    k = int(strength) * 2 + 1
                    img = cv2.GaussianBlur(img, (k, k), 0)
                    img = img.astype(np.float32) / 255.0
                    imgs[j] = torch.from_numpy(img.transpose(2, 0, 1))
                trainer.batch['img'] = imgs
        return on_train_batch_start

    results = []

    # Enforce GPU usage: exit if CUDA is not available
    if not torch.cuda.is_available():
        print("\n[ERROR] No CUDA-capable GPU detected. This script is configured to require a GPU for training.")
        print("Please ensure you have an NVIDIA GPU and that the CUDA-enabled version of PyTorch is installed.")
        import sys
        sys.exit(1)

    DEVICE = 0
    print(f"Using device: {DEVICE} ({torch.cuda.get_device_name(0)})")

    for i, exp in enumerate(experiments):
        ds_name = exp["dataset"]
        data_yaml = ROOT / "src" / "model-integration" / "data" / ds_name / "dataset.yaml"

        print(f"\n[{i+1}/{len(experiments)}] Dataset: {ds_name} | Model: {exp['model']} | Aug: {exp['augmentation']}")

        if not data_yaml.exists():
            print(f"  [ERROR] Dataset YAML not found: {data_yaml}")
            continue

        model = None
        val_results = None

        try:
            model = YOLO(exp["model"])

            if exp["blur"] > 0:
                model.add_callback("on_train_batch_start", make_blur_callback(exp["blur"]))
            
            # Add real-time plotting callback (updates plot after each epoch)
            model.add_callback("on_train_epoch_end", make_plotting_callback(ds_name, exp["augmentation"]))

            results_obj = model.train(
                data=str(data_yaml),
                epochs=50,
                # imgsz=1024, # this takes too long, since we are using normal RAM as VRAM then...
                imgsz=640,
                device=DEVICE,
                workers=4,
                batch=32,
                cache=False,
                half=True,
                verbose=False
            )

            # Generate final progress plot
            try:
                run_dir = model.trainer.save_dir
                internal_csv = os.path.join(run_dir, "results.csv")
                plot_training_metrics(internal_csv, run_dir, title_prefix=f"{ds_name} ({exp['augmentation']})")
            except Exception as plot_err:
                print(f"  [Warning] Could not trigger final plotting: {plot_err}")

            val_results = model.val(
                data=str(data_yaml),
                split="test",
                device=DEVICE,
                verbose=False
            )

            result = {
                "dataset":      ds_name,
                "model":        exp["model"],
                "augmentation": exp["augmentation"],
                "precision":    round(val_results.box.mp, 3),
                "recall":       round(val_results.box.mr, 3),
                "mAP50":        round(val_results.box.map50, 3),
                "mAP50_95":     round(val_results.box.map, 3),
            }

        except Exception as e:
            print(f"  [ERROR] {e}")
            result = {
                "dataset":      ds_name,
                "model":        exp["model"],
                "augmentation": exp["augmentation"],
                "precision":    "ERROR",
                "recall":       "ERROR",
                "mAP50":        "ERROR",
                "mAP50_95":     "ERROR",
            }

        results.append(result)
        print(f"\n  --- RESULTS ---")
        print(f"  Dataset   : {result['dataset']}")
        print(f"  Precision : {result['precision']}")
        print(f"  Recall    : {result['recall']}")
        print(f"  mAP50     : {result['mAP50']}")
        print(f"  mAP50-95  : {result['mAP50_95']}")

        # Append result to CSV
        file_exists = RESULTS_CSV.exists()
        with open(RESULTS_CSV, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=result.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(result)

        print(f"  Results appended to {RESULTS_CSV}")

        # Safe cleanup
        if model is not None:
            del model
        if val_results is not None:
            del val_results
            
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    print(f"\nAll done! Results saved to {RESULTS_CSV}")
    
    # Plot final benchmark results
    plot_benchmark_results(RESULTS_CSV, RESULTS_CSV.parent)
