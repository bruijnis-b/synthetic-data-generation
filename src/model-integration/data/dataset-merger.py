import os
import shutil
import random
import argparse
import yaml
from pathlib import Path
from collections import defaultdict

def find_project_root():
    """Finds the project root by looking for the 'data' directory."""
    current = Path(__file__).resolve().parent
    for _ in range(5):
        if (current / "data").exists() and (current / "src" / "model-integration").exists():
            return current
        current = current.parent
    return Path(__file__).resolve().parent.parent

def get_valid_pairs(images_dir, labels_dir):
    """Returns a list of (image_name, label_name) tuples for matching pairs."""
    if not images_dir.exists() or not labels_dir.exists():
        return []
    
    image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    valid_pairs = []
    for img_name in image_files:
        label_name = Path(img_name).with_suffix('.txt').name
        if (labels_dir / label_name).exists():
            valid_pairs.append((img_name, label_name))
    return valid_pairs

def create_structure(output_dir):
    """Creates the standard YOLO directory structure."""
    for split in ['train', 'val', 'test']:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

def write_yaml(output_dir):
    """Generates the dataset.yaml file."""
    yaml_content = {
        'path': str(output_dir.resolve()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': {0: 'syringe'}
    }
    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
    return yaml_path

def sanitize_and_copy_label(src_path, dst_path):
    """Reads a YOLO label file, clamps coordinates to [0, 1], and writes to destination."""
    try:
        with open(src_path, 'r') as f:
            lines = f.readlines()
        
        cleaned_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            
            # First part is class ID (integer)
            class_id = parts[0]
            # Rest are coordinates (floats)
            try:
                coords = [float(x) for x in parts[1:]]
                # Clamp to [0.0, 1.0]
                clamped_coords = [max(0.0, min(1.0, x)) for x in coords]
                # Reconstruct line
                cleaned_line = f"{class_id} " + " ".join([f"{x:.7f}" for x in clamped_coords])
                cleaned_lines.append(cleaned_line)
            except ValueError:
                print(f"Warning: Skipping malformed line in {src_path}")
            
        with open(dst_path, 'w') as f:
            f.write("\n".join(cleaned_lines) + "\n")
    except Exception as e:
        print(f"Warning: Failed to sanitize {src_path}: {e}. Falling back to copy.")
        shutil.copy2(src_path, dst_path)

def parse_source_arg(arg_list):
    """Parses list of 'source:ratio' strings into a dict."""
    result = []
    if not arg_list:
        return result
    for item in arg_list:
        if ':' in item:
            name, ratio = item.split(':')
            result.append((name, float(ratio)))
        else:
            result.append((item, 1.0))
    return result

def merge_datasets(args):
    base_dir = find_project_root()
    output_dir = base_dir / "src" / "model-integration" / "data" / args.output
    
    # Map sources and their requested ratios for each split
    # {source_name: {split_name: ratio}}
    source_map = defaultdict(dict)
    
    for name, ratio in parse_source_arg(args.train):
        source_map[name]['train'] = ratio
    for name, ratio in parse_source_arg(args.val):
        source_map[name]['val'] = ratio
    for name, ratio in parse_source_arg(args.test):
        source_map[name]['test'] = ratio

    # Check for over-allocation (sum of ratios > 1.0 for a single source)
    for name, splits in source_map.items():
        total_ratio = sum(splits.values())
        if total_ratio > 1.0001:
            print(f"Error: Source '{name}' is over-allocated (Total ratio: {total_ratio})")
            return

    # Load and partition each source
    final_splits = defaultdict(list) # {split: [(source_name, img, lbl), ...]}
    
    random.seed(args.seed)
    
    for name, requested_splits in source_map.items():
        # Try finding source in data/ or as a raw path
        src_path = base_dir / "data" / name
        if not src_path.exists():
            src_path = Path(name)
            if not src_path.is_absolute():
                src_path = base_dir / src_path
        
        if not src_path.exists():
            print(f"Error: Could not find source '{name}' at {src_path}")
            continue

        pairs = get_valid_pairs(src_path / "images", src_path / "labels")
        if not pairs:
            print(f"Warning: No valid pairs found in {src_path}")
            continue

        random.shuffle(pairs)
        total = len(pairs)
        
        # Partition the shuffled pairs
        current_idx = 0
        for split in ['train', 'val', 'test']:
            ratio = requested_splits.get(split, 0.0)
            if ratio <= 0:
                continue
            
            count = int(total * ratio)
            end_idx = current_idx + count
            selected = pairs[current_idx:end_idx]
            for img, lbl in selected:
                final_splits[split].append((src_path, img, lbl))
            current_idx = end_idx

    # Create structure and copy files
    create_structure(output_dir)
    print(f"Merging datasets into {args.output}...")
    
    for split in ['train', 'val', 'test']:
        items = final_splits[split]
        if not items:
            continue
        
        print(f"  - {split}: {len(items)} items")
        for src_path, img, lbl in items:
            src_folder_name = src_path.name
            target_img = f"{src_folder_name}_{img}"
            target_lbl = f"{src_folder_name}_{lbl}"
            
            shutil.copy2(src_path / "images" / img, output_dir / "images" / split / target_img)
            sanitize_and_copy_label(src_path / "labels" / lbl, output_dir / "labels" / split / target_lbl)

    write_yaml(output_dir)
    print(f"\nDone! Dataset merged successfully at {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset Merger (Multi-Source)")
    parser.add_argument("--output", type=str, required=True, help="Output folder name in model-integration/data/")
    parser.add_argument("--train", type=str, nargs='+', default=[], help="Sources for train (format: name:ratio)")
    parser.add_argument("--val", type=str, nargs='+', default=[], help="Sources for val (format: name:ratio)")
    parser.add_argument("--test", type=str, nargs='+', default=[], help="Sources for test (format: name:ratio)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    merge_datasets(args)
