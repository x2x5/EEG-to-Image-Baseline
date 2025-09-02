import os
import re
import sys
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from diffusers import AutoPipelineForText2Image
from transformers import CLIPVisionModelWithProjection
from PIL import Image
from torchvision import transforms
import matplotlib.font_manager as fm

plt.switch_backend("agg")


class Config:
    def __init__(self, config_dict):
        # 从字典中设置属性
        for key, value in config_dict.items():
            setattr(self, key, value)


def load_config(config_path="config.yaml"):
    """从 YAML 文件加载配置"""
    with open(config_path, "r", encoding="utf-8") as file:
        config_dict = yaml.safe_load(file)
    return Config(config_dict)


def load_labels(labels_file="data/labels_chinese.txt"):
    """加载中文标签文件"""
    if os.path.exists(labels_file):
        with open(labels_file, "r", encoding="utf-8") as f:
            return [line.strip() for line in f.readlines()]
    return None


def display_samples(
    gt_images, recon_images, labels, num_samples, imsize, output_path, use_chinese=True
):
    """Creates and saves a grid comparing ground truth and reconstructed images with labels."""
    print("Generating comparison image grid...")

    # 随机选择样本
    total_samples = len(gt_images)
    if num_samples > total_samples:
        num_samples = total_samples

    indices = np.random.choice(total_samples, num_samples, replace=False)
    indices = np.sort(indices)  # 保持顺序

    # 加载中文标签
    labels = None
    if use_chinese:
        labels = load_labels("config/labels_chinese.txt")
        font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
        font_prop = fm.FontProperties(fname=font_path)
        print(f"使用中文字体: {font_path}")
    else:
        labels = load_labels("config/labels_english.txt")
        font_prop = None

    # 创建图形
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 2, 4))
    if num_samples == 1:
        axes = axes.reshape(2, 1)

    # 在最左边添加行标签
    if font_prop and use_chinese:
        fig.text(
            0.06,
            0.75,
            "原图",
            fontproperties=font_prop,
            fontsize=14,
            ha="center",
            va="center",
            rotation=0,
        )
        fig.text(
            0.06,
            0.25,
            "重构图",
            fontproperties=font_prop,
            fontsize=14,
            ha="center",
            va="center",
            rotation=0,
        )
    else:
        fig.text(0.06, 0.75, "Seen", fontsize=14, ha="center", va="center", rotation=0)
        fig.text(
            0.06,
            0.25,
            "Recon",
            fontsize=14,
            ha="center",
            va="center",
            rotation=0,
        )

    for i, idx in enumerate(indices):
        # 原图
        gt_img = gt_images[idx].permute(1, 2, 0).cpu().numpy()
        axes[0, i].imshow(gt_img)
        axes[0, i].axis("off")

        # 重构图
        if len(recon_images.shape) == 4:
            recon_img = recon_images[idx].permute(1, 2, 0).cpu().numpy()
        else:
            recon_img = recon_images[0][idx].permute(1, 2, 0).cpu().numpy()
        axes[1, i].imshow(recon_img)
        axes[1, i].axis("off")

        # 添加标签（只在第一行显示类别标签）
        if use_chinese and labels and idx < len(labels):
            label = labels[idx]
        else:
            label = labels[idx] if idx < len(labels) else f"Sample {idx}"

        # 设置标签字体
        if font_prop and use_chinese and labels and idx < len(labels):
            axes[0, i].set_title(label, fontproperties=font_prop, fontsize=10, pad=5)
        else:
            axes[0, i].set_title(label, fontsize=10, pad=5)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85, left=0.08)  # 为左侧行标签留出空间
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Sample comparison grid saved to {output_path}")
    print(f"Selected indices: {indices.tolist()}")


def extract_id_from_string(s):
    """从字符串末尾提取数字ID"""
    match = re.search(r"\d+$", s)
    if match:
        return int(match.group())
    return None


def extract_timestamp_from_path(path):
    """从路径中提取时间戳"""
    # 使用正则表达式匹配时间戳格式 (MMDD_HHMMSS)
    match = re.search(r"(\d{4}_\d{6})", path)
    if match:
        return match.group(1)
    else:
        # 如果没有找到时间戳，返回None，让函数自动生成
        return None


def load_data(train=False, classes=None, pictures=None):
    data_list = []
    label_list = []
    texts = []
    images = []

    if train:
        data_directory = "/data1/share_data/EEG_Image_decode/training_images"
    else:
        data_directory = "/data1/share_data/EEG_Image_decode/test_images"

    dirnames = [
        d
        for d in os.listdir(data_directory)
        if os.path.isdir(os.path.join(data_directory, d))
    ]
    dirnames.sort()

    if classes is not None:
        dirnames = [dirnames[i] for i in classes]

    for dir in dirnames:

        try:
            idx = dir.index("_")
            description = dir[idx + 1 :]
        except ValueError:
            print(f"Skipped: {dir} due to no '_' found.")
            continue

        new_description = f"{description}"
        texts.append(new_description)

    if classes is not None and pictures is not None:
        images = []
        for i in range(len(classes)):
            class_idx = classes[i]
            pic_idx = pictures[i]
            if class_idx < len(dirnames):
                folder = dirnames[class_idx]
                folder_path = os.path.join(data_directory, folder)
                all_images = [
                    img
                    for img in os.listdir(folder_path)
                    if img.lower().endswith((".png", ".jpg", ".jpeg"))
                ]
                all_images.sort()
                if pic_idx < len(all_images):
                    images.append(os.path.join(folder_path, all_images[pic_idx]))
    elif classes is not None and pictures is None:
        images = []
        for i in range(len(classes)):
            class_idx = classes[i]
            if class_idx < len(dirnames):
                folder = dirnames[class_idx]
                folder_path = os.path.join(data_directory, folder)
                all_images = [
                    img
                    for img in os.listdir(folder_path)
                    if img.lower().endswith((".png", ".jpg", ".jpeg"))
                ]
                all_images.sort()
                images.extend(os.path.join(folder_path, img) for img in all_images)
    elif classes is None:
        images = []
        for folder in dirnames:
            folder_path = os.path.join(data_directory, folder)
            all_images = [
                img
                for img in os.listdir(folder_path)
                if img.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
            all_images.sort()
            images.extend(os.path.join(folder_path, img) for img in all_images)
    else:

        print("Error")
    return texts, images


class IPAdapterGenerator:
    def __init__(self, device="cuda", dtype=torch.float16):
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "h94/IP-Adapter",
            subfolder="models/image_encoder",
            torch_dtype=dtype,
        )
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo",
            image_encoder=image_encoder,
            torch_dtype=dtype,
            variant="fp16",
        )
        self.pipe.to(device)
        self.pipe.load_ip_adapter(
            "h94/IP-Adapter",
            subfolder="sdxl_models",
            weight_name="ip-adapter_sdxl_vit-h.safetensors",
            torch_dtype=dtype,
        )
        self.pipe.set_ip_adapter_scale(1)

    def generate(self, image_embeds, num_images_per_prompt=1):
        images = self.pipe(
            prompt="",
            ip_adapter_image_embeds=[image_embeds],
            num_inference_steps=4,
            guidance_scale=0.0,
            num_images_per_prompt=num_images_per_prompt,
        ).images
        return images


def process_images(source_dir, imsize):
    """
    Scans a directory of images, preprocesses them (resize, to_tensor),
    and saves them as a single stacked PyTorch tensor.
    Note: It takes only the first image from each sub-folder, as in the original notebook.
    """
    # if os.path.exists(target_path) and not force_preprocess:
    #     print(f"Tensor file already exists at {target_path}. Skipping preprocessing.")
    #     tensor_data = torch.load(target_path)
    #     if isinstance(tensor_data, dict):
    #         return tensor_data["tensors"], tensor_data["labels"]
    #     else:
    #         # 兼容旧格式，重新生成标签
    #         labels = get_folder_labels(source_dir)
    #         return tensor_data, labels

    print(f"Processing images from '{source_dir}'...")

    transform = transforms.Compose(
        [
            transforms.Resize(
                (imsize, imsize), interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.ToTensor(),  # Converts to [C, H, W] FloatTensor and scales to [0, 1]
        ]
    )

    tensor_list = []
    labels = []
    # Use sorted() for consistent order
    for folder_name in sorted(os.listdir(source_dir)):
        folder_path = os.path.join(source_dir, folder_name)
        if os.path.isdir(folder_path):
            image_files = sorted(os.listdir(folder_path))
            if not image_files:
                continue
            # Process multiple images in the sub-directory
            for image_name in image_files:
                image_path = os.path.join(folder_path, image_name)
                try:
                    with Image.open(image_path) as img:
                        img_rgb = img.convert("RGB")
                        tensor = transform(img_rgb)
                        tensor_list.append(tensor)
                        labels.append(folder_name)
                except Exception as e:
                    print(
                        f"Skipping file {image_path} due to error: {e}", file=sys.stderr
                    )

    if not tensor_list:
        raise FileNotFoundError(
            f"No images were processed from {source_dir}. Check the directory structure."
        )

    all_tensors = torch.stack(tensor_list, dim=0)
    print(f"Final tensor shape: {all_tensors.shape}")

    # 保存张量和标签 (Removed saving to file)
    # tensor_data = {"tensors": all_tensors, "labels": labels}
    # torch.save(tensor_data, target_path)
    return all_tensors, labels


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == "type1":
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == "type2":
        lr_adjust = {2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 10: 5e-7, 15: 1e-7, 20: 5e-8}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        print("Updating learning rate to {}".format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), path + "/" + "checkpoint.pth")
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name="./pic/test.pdf"):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label="GroundTruth", linewidth=2)
    if preds is not None:
        plt.plot(preds, label="Prediction", linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches="tight")


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)
