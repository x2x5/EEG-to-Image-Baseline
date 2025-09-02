import os

import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from itertools import combinations

import clip
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
from tqdm import tqdm
from eegdatasets_leaveone import EEGDataset

from einops.layers.torch import Rearrange, Reduce

from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
import random
from braindecode.models import EEGNetv4, ATCNet, EEGConformer, EEGITNet, ShallowFBCSPNet
import csv
from torch import Tensor
import itertools
import math
import re
from subject_layers.Transformer_EncDec import Encoder, EncoderLayer
from subject_layers.SelfAttention_Family import FullAttention, AttentionLayer
from subject_layers.Embed import DataEmbedding
import numpy as np
from loss import ClipLoss
import yaml
from torch import nn
from torch.optim import AdamW
import logging


class TransformerConfig:
    def __init__(self):
        self.task_name = "classification"  # Example task name
        self.seq_len = 250  # Sequence length
        self.pred_len = 250  # Prediction length
        self.output_attention = False  # Whether to output attention weights
        self.d_model = 250  # Model dimension
        self.embed = "timeF"  # Time encoding method
        self.freq = "h"  # Time frequency
        self.dropout = 0.25  # Dropout rate
        self.factor = 1  # Attention scaling factor
        self.n_heads = 4  # Number of attention heads
        self.e_layers = 1  # Number of encoder layers
        self.d_ff = 256  # Dimension of the feedforward network
        self.activation = "gelu"  # Activation function
        self.enc_in = 63  # Encoder input dimension (example value)


class iTransformer(nn.Module):
    def __init__(self, configs, joint_train=False, num_subjects=10):
        super(iTransformer, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        # Embedding
        self.enc_embedding = DataEmbedding(
            configs.seq_len,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
            joint_train=False,
            num_subjects=num_subjects,
        )
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )

    def forward(self, x_enc, x_mark_enc, subject_ids=None):
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc, subject_ids)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        enc_out = enc_out[:, :63, :]
        # print("enc_out", enc_out.shape)
        return enc_out


class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        super().__init__()
        # Revised from ShallowNet
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), stride=(1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (63, 1), stride=(1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),
            Rearrange("b e (h) (w) -> b (h w) e"),
        )

    def forward(self, x: Tensor) -> Tensor:
        # b, _, _, _ = x.shape
        x = x.unsqueeze(1)
        # print("x", x.shape)
        x = self.tsconv(x)
        # print("tsconv", x.shape)
        x = self.projection(x)
        # print("projection", x.shape)
        return x


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FlattenHead(nn.Sequential):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        return x


class Enc_eeg(nn.Sequential):
    def __init__(self, emb_size=40, **kwargs):
        super().__init__(PatchEmbedding(emb_size), FlattenHead())


class Proj_eeg(nn.Sequential):
    def __init__(self, embedding_dim=1440, proj_dim=1024, drop_proj=0.5):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(
                nn.Sequential(
                    nn.GELU(),
                    nn.Linear(proj_dim, proj_dim),
                    nn.Dropout(drop_proj),
                )
            ),
            nn.LayerNorm(proj_dim),
        )


class ATMS(nn.Module):
    def __init__(
        self,
        num_channels=63,
        sequence_length=250,
        num_subjects=2,
        num_features=64,
        num_latents=1024,
        num_blocks=1,
    ):
        super(ATMS, self).__init__()
        default_config = TransformerConfig()
        self.encoder = iTransformer(default_config)
        self.subject_wise_linear = nn.ModuleList(
            [
                nn.Linear(default_config.d_model, sequence_length)
                for _ in range(num_subjects)
            ]
        )
        self.enc_eeg = Enc_eeg()
        self.proj_eeg = Proj_eeg()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = ClipLoss()

    def forward(self, x, subject_ids):
        x = self.encoder(x, None, subject_ids)
        # print(f'After attention shape: {x.shape}')
        # print("x", x.shape)
        # x = self.subject_wise_linear[0](x)
        # print(f'After subject-specific linear transformation shape: {x.shape}')
        eeg_embedding = self.enc_eeg(x)

        out = self.proj_eeg(eeg_embedding)
        return out


def extract_id_from_string(s):
    match = re.search(r"\d+$", s)
    if match:
        return int(match.group())
    return None


def train_model(
    sub,
    eeg_model,
    dataloader,
    optimizer,
    device,
    text_features_all,
    img_features_all,
    config,
):
    eeg_model.train()
    text_features_all = text_features_all.to(device).float()  # (n_cls, d)
    img_features_all = (img_features_all[::10]).to(device).float()
    total_loss = 0
    correct = 0
    total = 0
    alpha = 0.90
    features_list = []  # List to store features
    save_features = True
    mse_loss_fn = nn.MSELoss()
    for batch_idx, (
        eeg_data,
        labels,
        text,
        text_features,
        img,
        img_features,
    ) in enumerate(dataloader):
        eeg_data = eeg_data.to(device)
        text_features = text_features.to(device).float()
        img_features = img_features.to(device).float()
        labels = labels.to(device)

        optimizer.zero_grad()

        batch_size = eeg_data.size(0)  # Assume the first element is the data tensor
        subject_id = extract_id_from_string(sub)
        # eeg_data = eeg_data.permute(0, 2, 1)
        subject_ids = torch.full((batch_size,), subject_id, dtype=torch.long).to(device)
        # if not config.insubject:
        #     subject_ids = torch.full((batch_size,), -1, dtype=torch.long).to(device)
        eeg_features = eeg_model(eeg_data, subject_ids).float()

        features_list.append(eeg_features)
        logit_scale = eeg_model.logit_scale

        img_loss = eeg_model.loss_func(eeg_features, img_features, logit_scale)
        text_loss = eeg_model.loss_func(eeg_features, text_features, logit_scale)
        # loss = img_loss + text_loss
        # print("text_loss", text_loss)
        # print("img_loss", img_loss)
        regress_loss = mse_loss_fn(eeg_features, img_features)
        loss = alpha * regress_loss * 10 + (1 - alpha) * img_loss * 10
        loss.backward()

        optimizer.step()
        total_loss += loss.item()

        # logits = logit_scale * eeg_features @ text_features_all.T # (n_batch, n_cls)
        # Compute corresponding logits
        logits_img = logit_scale * eeg_features @ img_features_all.T
        # logits_text = logit_scale * eeg_features @ text_features_all.T
        # logits_single = (logits_text + logits_img) / 2.0
        # logits_text = logit_scale * eeg_features @ text_features_all.T
        logits_single = logits_img
        predicted = torch.argmax(
            logits_single, dim=1
        )  # (n_batch, ) ∈ {0, 1, ..., n_cls-1}

        batch_size = predicted.shape[0]
        total += batch_size
        correct += (predicted == labels).sum().item()
        del eeg_data, eeg_features, img_features
    average_loss = total_loss / (batch_idx + 1)
    accuracy = correct / total
    return average_loss, accuracy, torch.cat(features_list, dim=0)


def evaluate_model(
    sub, eeg_model, dataloader, device, text_features_all, img_features_all, k, config
):
    eeg_model.eval()

    text_features_all = text_features_all.to(device).float()
    img_features_all = img_features_all.to(device).float()
    total_loss = 0
    correct = 0
    total = 0
    alpha = 0.99
    top5_correct = 0
    top5_correct_count = 0
    all_labels = set(range(text_features_all.size(0)))
    top5_acc = 0
    mse_loss_fn = nn.MSELoss()
    with torch.no_grad():
        for batch_idx, (
            eeg_data,
            labels,
            text,
            text_features,
            img,
            img_features,
        ) in enumerate(dataloader):
            eeg_data = eeg_data.to(device)
            text_features = text_features.to(device).float()
            labels = labels.to(device)
            img_features = img_features.to(device).float()

            batch_size = eeg_data.size(0)  # Assume the first element is the data tensor
            subject_id = extract_id_from_string(sub)
            # eeg_data = eeg_data.permute(0, 2, 1)
            subject_ids = torch.full((batch_size,), subject_id, dtype=torch.long).to(
                device
            )
            # if not config.insubject:
            #     subject_ids = torch.full((batch_size,), -1, dtype=torch.long).to(device)
            eeg_features = eeg_model(eeg_data, subject_ids)

            logit_scale = eeg_model.logit_scale
            # print(eeg_features.type, text_features.type, img_features.type)
            img_loss = eeg_model.loss_func(eeg_features, img_features, logit_scale)
            text_loss = eeg_model.loss_func(eeg_features, text_features, logit_scale)
            regress_loss = mse_loss_fn(eeg_features, img_features)
            loss = alpha * regress_loss * 10 + (1 - alpha) * img_loss * 10

            total_loss += loss.item()

            for idx, label in enumerate(labels):
                # First select k-1 classes excluding the correct class
                possible_classes = list(all_labels - {label.item()})
                selected_classes = random.sample(possible_classes, k - 1) + [
                    label.item()
                ]
                selected_img_features = img_features_all[selected_classes]
                selected_text_features = text_features_all[selected_classes]

                if k == 200:
                    # Compute corresponding logits
                    logits_img = (
                        logit_scale * eeg_features[idx] @ selected_img_features.T
                    )
                    logits_single = logits_img
                    # print("logits_single", logits_single.shape)
                    # Get predicted class
                    # predicted_label = selected_classes[torch.argmax(logits_single).item()]
                    predicted_label = selected_classes[
                        torch.argmax(logits_single).item()
                    ]  # (n_batch, ) ∈ {0, 1, ..., n_cls-1}
                    if predicted_label == label.item():
                        # print("predicted_label", predicted_label)
                        correct += 1

                    # logits_single is the model output, assumed to be shape (n_batch, n_classes)
                    # label is the true label, shape (n_batch,)
                    # Get top-5 predicted indices
                    # print("logits_single", logits_single)
                    _, top5_indices = torch.topk(logits_single, 5, largest=True)

                    # Check if true label is in top-5 predictions
                    if label.item() in [
                        selected_classes[i] for i in top5_indices.tolist()
                    ]:
                        top5_correct_count += 1
                    total += 1
                elif k == 50 or k == 100:
                    # For k=50 or 100, select k classes for evaluation
                    selected_classes = random.sample(possible_classes, k - 1) + [
                        label.item()
                    ]

                    logits_img = (
                        logit_scale * eeg_features[idx] @ selected_img_features.T
                    )
                    logits_single = logits_img

                    predicted_label = selected_classes[
                        torch.argmax(logits_single).item()
                    ]
                    if predicted_label == label.item():
                        correct += 1
                    _, top5_indices = torch.topk(logits_single, 5, largest=True)

                    # Check if true label is in top-5 predictions
                    if label.item() in [
                        selected_classes[i] for i in top5_indices.tolist()
                    ]:
                        top5_correct_count += 1
                    total += 1
                elif k == 2 or k == 4 or k == 10:
                    selected_classes = random.sample(possible_classes, k - 1) + [
                        label.item()
                    ]
                    # Compute corresponding logits
                    logits_img = (
                        logit_scale * eeg_features[idx] @ selected_img_features.T
                    )
                    # logits_text = logit_scale * eeg_features[idx] @ selected_text_features.T
                    # logits_single = (logits_text + logits_img) / 2.0
                    logits_single = logits_img
                    # print("logits_single", logits_single.shape)
                    # Get predicted class
                    # predicted_label = selected_classes[torch.argmax(logits_single).item()]
                    predicted_label = selected_classes[
                        torch.argmax(logits_single).item()
                    ]  # (n_batch, ) ∈ {0, 1, ..., n_cls-1}
                    if predicted_label == label.item():
                        correct += 1
                    total += 1
                else:
                    print("Error.")
            del eeg_data, eeg_features, img_features
    average_loss = total_loss / (batch_idx + 1)
    accuracy = correct / total
    top5_acc = top5_correct_count / total
    return average_loss, accuracy, top5_acc


def main_train_loop(
    sub,
    current_time,
    eeg_model,
    train_dataloader,
    test_dataloader,
    optimizer,
    device,
    text_features_train_all,
    text_features_test_all,
    img_features_train_all,
    img_features_test_all,
    config,
):

    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []
    v2_accs = []
    v4_accs = []
    v10_accs = []

    best_accuracy = 0.0
    best_model_weights = None
    best_epoch_info = {}
    results = []  # List to store results for each epoch

    # 创建日志目录和文件
    log_dir = os.path.join(config.output_dir, config.encoder_type, sub, current_time)
    os.makedirs(log_dir, exist_ok=True)

    # 设置高效的日志记录器
    log_file = os.path.join(log_dir, "training_log.txt")

    # 创建专用的训练日志记录器
    train_logger = logging.getLogger(f'training_{sub}_{current_time.replace("/", "_")}')
    train_logger.setLevel(logging.INFO)

    # 避免重复添加handler
    if not train_logger.handlers:
        # 文件处理器（追加模式）
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(message)s")
        file_handler.setFormatter(formatter)
        train_logger.addHandler(file_handler)

    # 写入日志头部信息
    train_logger.info(f"开始训练 - 受试者: {sub}, 编码器: {config.encoder_type}")
    train_logger.info(
        "Epoch,TrainLoss,TrainAcc,TestLoss,TestAcc,Top5Acc,V2Acc,V4Acc,V10Acc,V50Acc,V100Acc"
    )

    for epoch in tqdm(range(config.epochs)):
        # Train the model
        train_loss, train_accuracy, features_tensor = train_model(
            sub,
            eeg_model,
            train_dataloader,
            optimizer,
            device,
            text_features_train_all,
            img_features_train_all,
            config=config,
        )

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Evaluate the model
        test_loss, test_accuracy, top5_acc = evaluate_model(
            sub,
            eeg_model,
            test_dataloader,
            device,
            text_features_test_all,
            img_features_test_all,
            k=200,
            config=config,
        )
        _, v2_acc, _ = evaluate_model(
            sub,
            eeg_model,
            test_dataloader,
            device,
            text_features_test_all,
            img_features_test_all,
            k=2,
            config=config,
        )
        _, v4_acc, _ = evaluate_model(
            sub,
            eeg_model,
            test_dataloader,
            device,
            text_features_test_all,
            img_features_test_all,
            k=4,
            config=config,
        )
        _, v10_acc, _ = evaluate_model(
            sub,
            eeg_model,
            test_dataloader,
            device,
            text_features_test_all,
            img_features_test_all,
            k=10,
            config=config,
        )
        _, v50_acc, v50_top5_acc = evaluate_model(
            sub,
            eeg_model,
            test_dataloader,
            device,
            text_features_test_all,
            img_features_test_all,
            k=50,
            config=config,
        )
        _, v100_acc, v100_top5_acc = evaluate_model(
            sub,
            eeg_model,
            test_dataloader,
            device,
            text_features_test_all,
            img_features_test_all,
            k=100,
            config=config,
        )
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        v2_accs.append(v2_acc)
        v4_accs.append(v4_acc)
        v10_accs.append(v10_acc)

        # 高效记录每轮指标到日志文件
        log_line = f"{epoch+1},{train_loss:.4f},{train_accuracy*100:.2f},{test_loss:.4f},{test_accuracy*100:.2f},{top5_acc*100:.2f},{v2_acc*100:.2f},{v4_acc*100:.2f},{v10_acc*100:.2f},{v50_acc*100:.2f},{v100_acc*100:.2f}"
        train_logger.info(log_line)

        # Append results for this epoch
        epoch_results = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "v2_acc": v2_acc,
            "v4_acc": v4_acc,
            "v10_acc": v10_acc,
            "top5_acc": top5_acc,
            "v50_acc": v50_acc,
            "v100_acc": v100_acc,
            "v50_top5_acc": v50_top5_acc,
            "v100_top5_acc": v100_top5_acc,
        }

        results.append(epoch_results)

        # 保存最佳测试准确率的模型权重
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_model_weights = eeg_model.state_dict().copy()

            # 保存最佳模型
            best_model_path = os.path.join(log_dir, "best_model.pth")
            torch.save(best_model_weights, best_model_path)

            best_epoch_info = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "v2_acc": v2_acc,
                "v4_acc": v4_acc,
                "v10_acc": v10_acc,
            }
            train_logger.info(
                f"新的最佳模型! Epoch {epoch+1}, Test Acc: {test_accuracy*100:.2f}%"
            )

        print(
            f"Epoch {epoch + 1}/{config.epochs} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy*100:.2f}%, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy*100:.2f}%, Top5 Accuracy: {top5_acc*100:.2f}%"
        )
        print(
            f"Epoch {epoch + 1}/{config.epochs} - v2 Accuracy: {v2_acc*100:.2f}% - v4 Accuracy: {v4_acc*100:.2f}% - v10 Accuracy: {v10_acc*100:.2f}% - v50 Accuracy: {v50_acc*100:.2f}% - v100 Accuracy: {v100_acc*100:.2f}%"
        )

    # 记录最佳模型信息
    train_logger.info(
        f"训练完成! 最佳测试准确率: {best_accuracy*100:.2f}% (Epoch {best_epoch_info['epoch']})"
    )
    train_logger.info(f"最佳模型已保存至: {os.path.join(log_dir, 'best_model.pth')}")

    # # Load best model weights
    # model.load_state_dict(best_model_weights)

    # # # Save best model
    # torch.save(model.state_dict(), '{train_pos_img_text}.pth')

    # Create 5 subplots
    fig, axs = plt.subplots(3, 2, figsize=(10, 15))

    # Loss plot
    axs[0, 0].plot(train_losses, label="Train Loss")
    axs[0, 0].plot(test_losses, label="Test Loss")
    axs[0, 0].legend()
    axs[0, 0].set_title("Loss Curve")

    # Overall accuracy plot
    axs[0, 1].plot(train_accuracies, label="Train Accuracy")
    axs[0, 1].plot(test_accuracies, label="Test Accuracy")
    axs[0, 1].legend()
    axs[0, 1].set_title("Accuracy Curve")

    # The following are the three new plots you added, assuming you have calculated the corresponding accuracies
    # 2-class accuracy plot
    axs[1, 0].plot(v2_accs, label="2-class Accuracy")
    axs[1, 0].legend()
    axs[1, 0].set_title("2-Class Accuracy Curve")

    # 4-class accuracy plot
    axs[1, 1].plot(v4_accs, label="4-class Accuracy")
    axs[1, 1].legend()
    axs[1, 1].set_title("4-Class Accuracy Curve")

    # 10-class accuracy plot
    axs[2, 0].plot(v10_accs, label="10-class Accuracy")
    axs[2, 0].legend()
    axs[2, 0].set_title("10-Class Accuracy Curve")

    # Construct the string information you want to annotate
    info_text = (
        f"Best Model Info (from Epoch {best_epoch_info['epoch']}):\n"
        f"Train Loss: {best_epoch_info['train_loss']:.4f}\n"
        f"Train Accuracy: {best_epoch_info['train_accuracy']*100:.2f}%\n"
        f"Test Loss: {best_epoch_info['test_loss']:.4f}\n"
        f"Test Accuracy: {best_epoch_info['test_accuracy']*100:.2f}%\n"
        f"v2_acc: {best_epoch_info['v2_acc']*100:.2f}%\n"
        f"v4_acc: {best_epoch_info['v4_acc']*100:.2f}%\n"
        f"v10_acc: {best_epoch_info['v10_acc']*100:.2f}%"
    )

    axs[2, 1].axis("off")
    axs[2, 1].text(
        0.5,
        0.5,
        info_text,
        fontsize=10,
        ha="center",
        va="center",
        transform=axs[2, 1].transAxes,
    )

    plt.tight_layout()

    # Add main title
    plt.suptitle("pos_img_text", fontsize=16, y=1.05)

    # 保存图表到正确的输出目录
    plot_path = os.path.join(log_dir, "training_curves.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    train_logger.info(f"训练曲线图已保存至: {plot_path}")

    return results


import datetime


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


def main():
    # 从 YAML 配置文件加载配置
    try:
        config = load_config("config/train.yaml")
    except FileNotFoundError:
        print("配置文件 config.yaml 未找到，请确保文件存在")
        return
    except yaml.YAMLError as e:
        print(f"配置文件解析错误: {e}")
        return

    # 设置设备
    device = torch.device(config.device)

    subjects = config.subjects
    # 修改时间格式为月日/时分秒
    now = datetime.datetime.now()
    date_str = now.strftime("%m%d")  # 月日
    time_str = now.strftime("%H%M%S")  # 时分秒
    current_time = f"{date_str}/{time_str}"

    print(f"开始训练，配置信息：")
    print(f"编码器类型: {config.encoder_type}")
    print(f"设备: {device}")
    print(f"学习率: {config.lr}")
    print(f"训练轮数: {config.epochs}")
    print(f"批次大小: {config.batch_size}")
    print(f"受试者内模式: {config.insubject}")

    for sub in subjects:
        print(f"\n正在训练受试者: {sub}")
        eeg_model = globals()[config.encoder_type]()
        eeg_model.to(device)

        optimizer = AdamW(itertools.chain(eeg_model.parameters()), lr=config.lr)

        if config.insubject:
            train_dataset = EEGDataset(config, subjects=[sub], train=True)
            test_dataset = EEGDataset(config, subjects=[sub], train=False)
        else:
            train_dataset = EEGDataset(
                config, exclude_subject=sub, subjects=subjects, train=True
            )
            test_dataset = EEGDataset(
                config, exclude_subject=sub, subjects=subjects, train=False
            )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=6,
            drop_last=True,
        )
        test_loader = DataLoader(
            test_dataset, batch_size=200, shuffle=False, num_workers=3, drop_last=True
        )

        text_features_train_all = train_dataset.text_features
        text_features_test_all = test_dataset.text_features
        img_features_train_all = train_dataset.img_features
        img_features_test_all = test_dataset.img_features

        results = main_train_loop(
            sub,
            current_time,
            eeg_model,
            train_loader,
            test_loader,
            optimizer,
            device,
            text_features_train_all,
            text_features_test_all,
            img_features_train_all,
            img_features_test_all,
            config=config,
        )


if __name__ == "__main__":
    main()
