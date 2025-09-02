import torch
import numpy as np
from torchvision import transforms
from torchvision.models import (
    alexnet,
    AlexNet_Weights,
    inception_v3,
    Inception_V3_Weights,
    EfficientNet_B1_Weights,
    efficientnet_b1,
)
import clip
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import scipy as sp

from torchvision.models.feature_extraction import create_feature_extractor


@torch.no_grad()
def calculate_identification_accuracy(
    all_brain_recons, all_images, model, preprocess, device, feature_layer=None
):
    """Core function for identification accuracy."""
    preds = model(
        torch.stack([preprocess(recon) for recon in all_brain_recons]).to(device)
    )
    reals = model(torch.stack([preprocess(indiv) for indiv in all_images]).to(device))

    if feature_layer is not None:
        preds, reals = preds[feature_layer], reals[feature_layer]

    preds = preds.float().flatten(1).cpu().numpy()
    reals = reals.float().flatten(1).cpu().numpy()

    r = np.corrcoef(reals, preds)
    r = r[: len(all_images), len(all_images) :]

    congruents = np.diag(r)
    # 按照notebook中的逻辑计算成功率
    success = r < congruents
    success_cnt = np.sum(success, 0)
    perf = np.mean(success_cnt) / (len(all_images) - 1)

    return perf


def calculate_pixel_correlation(recons, gts):
    """Calculates pixel-wise correlation."""
    print("Calculating Pixel Correlation (PixCorr)...")
    preprocess = transforms.Compose(
        [
            transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR),
        ]
    )

    # 使用与notebook一致的方法：view而不是reshape，保持tensor格式
    gts_flat = preprocess(gts).view(len(gts), -1).cpu()
    recons_flat = preprocess(recons).view(len(recons), -1).cpu()
    corr = [
        np.corrcoef(gts_flat[i], recons_flat[i])[0, 1]
        for i in tqdm(range(len(gts)), desc="PixCorr")
    ]
    pixcorr = np.mean(corr)
    return {"PixCorr": pixcorr}


def calculate_structural_similarity(recons, gts):
    """Calculates Structural Similarity Index (SSIM)."""

    print("Calculating Structural Similarity Index (SSIM)...")
    preprocess = transforms.Compose(
        [
            transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR),
        ]
    )

    gts_gray = rgb2gray(preprocess(gts).permute(0, 2, 3, 1).cpu().numpy())
    recons_gray = rgb2gray(preprocess(recons).permute(0, 2, 3, 1).cpu().numpy())

    ssim_scores = [
        ssim(
            gts_gray[i],
            recons_gray[i],
            multichannel=True,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
            data_range=1.0,
        )
        for i in tqdm(range(len(gts)), desc="SSIM")
    ]
    return {"SSIM": np.mean(ssim_scores)}


def calculate_alexnet_metrics(recons, gts, device):
    """Calculates identification metrics using AlexNet features."""
    print("Calculating AlexNet metrics...")
    weights = AlexNet_Weights.IMAGENET1K_V1
    model = create_feature_extractor(
        alexnet(weights=weights), return_nodes=["features.4", "features.11"]
    ).to(device)
    model.eval()

    # 使用与notebook一致的预处理设置
    preprocess = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    acc_early = calculate_identification_accuracy(
        recons, gts, model, preprocess, device, "features.4"
    )
    acc_mid = calculate_identification_accuracy(
        recons, gts, model, preprocess, device, "features.11"
    )

    return {"AlexNet-2": acc_early, "AlexNet-5": acc_mid}


def calculate_inception_accuracy(recons, gts, device):
    """Calculates identification accuracy using InceptionV3 features."""
    print("Calculating InceptionV3 metric...")
    weights = Inception_V3_Weights.DEFAULT
    model = create_feature_extractor(
        inception_v3(weights=weights), return_nodes=["avgpool"]
    ).to(device)
    model.eval()

    # 使用与notebook一致的预处理设置
    preprocess = transforms.Compose(
        [
            transforms.Resize(342, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    accuracy = calculate_identification_accuracy(
        recons, gts, model, preprocess, device, "avgpool"
    )
    return {"Inception": accuracy}


def calculate_clip_accuracy(recons, gts, device):
    """Calculates identification accuracy using CLIP image embeddings."""
    print("Calculating CLIP metric...")
    model, _ = clip.load("ViT-L/14", device=device)
    model.eval()

    # 使用与notebook一致的预处理方式
    preprocess = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )

    accuracy = calculate_identification_accuracy(
        recons, gts, model.encode_image, preprocess, device, None
    )
    return {"CLIP": accuracy}


@torch.no_grad()
def calculate_effnet_distance(recons, gts, device):
    """Calculates correlation distance using EfficientNet-B1 features."""
    print("Calculating EfficientNet-B1 Distance...")
    weights = EfficientNet_B1_Weights.DEFAULT
    model = create_feature_extractor(
        efficientnet_b1(weights=weights), return_nodes=["avgpool"]
    ).to(device)
    model.eval()

    # 使用与notebook一致的预处理设置
    preprocess = transforms.Compose(
        [
            transforms.Resize(255, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    gt_feats = model(preprocess(gts))["avgpool"].reshape(len(gts), -1).cpu().numpy()
    recon_feats = (
        model(preprocess(recons))["avgpool"].reshape(len(recons), -1).cpu().numpy()
    )

    distances = [
        sp.spatial.distance.correlation(gt_feats[i], recon_feats[i])
        for i in range(len(gt_feats))
    ]
    return {"EffNet-B": np.mean(distances)}


@torch.no_grad()
def calculate_swav_distance(recons, gts, device):
    """Calculates correlation distance using SwAV (ResNet50) features."""
    print("Calculating SwAV Distance...")
    model = torch.hub.load("facebookresearch/swav:main", "resnet50")
    model = create_feature_extractor(model, return_nodes=["avgpool"]).to(device)
    model.eval()

    preprocess = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    gt_feats = model(preprocess(gts))["avgpool"].reshape(len(gts), -1).cpu().numpy()
    recon_feats = (
        model(preprocess(recons))["avgpool"].reshape(len(recons), -1).cpu().numpy()
    )

    distances = [
        sp.spatial.distance.correlation(gt_feats[i], recon_feats[i])
        for i in range(len(gt_feats))
    ]
    return {"SwAV": np.mean(distances)}
