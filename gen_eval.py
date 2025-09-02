import os

# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["HF_HUB_OFFLINE"] = "1"

import yaml
import datetime
from torch.utils.data import DataLoader
from tqdm import tqdm
from ATMS_reconstruction import ATMS
from eegdatasets_leaveone import EEGDataset
import torch
import sys
import pandas as pd
from utils.metrics import (
    calculate_alexnet_metrics,
    calculate_inception_accuracy,
    calculate_pixel_correlation,
    calculate_structural_similarity,
    calculate_clip_accuracy,
    calculate_swav_distance,
)
from utils.tools import (
    Config,
    IPAdapterGenerator,
    display_samples,
    extract_id_from_string,
    load_config,
    load_data,
    process_images,
)


@torch.no_grad()
def extract_eeg_features(config):
    device = torch.device(config.device)
    print(f"Using device: {device}")

    eeg_model = ATMS()
    eeg_model.load_state_dict(
        torch.load(config.eeg_encoder_path, map_location=device, weights_only=True)
    )
    eeg_model = eeg_model.to(device)

    # 处理测试集
    print("\nProcessing test set...")
    test_dataset = EEGDataset(subjects=[config.sub], train=False)
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2
    )

    eeg_features_test = []

    subject_id = extract_id_from_string(config.sub)

    for eegs in test_loader:
        eeg_data, labels, texts, text_features, imgs, img_features = eegs
        eeg_data = eeg_data.to(device)
        subject_ids = torch.full((eeg_data.shape[0],), subject_id, dtype=torch.long).to(
            device
        )
        eeg_features = eeg_model(eeg_data, subject_ids)
        eeg_features_test.append(eeg_features)
    del eeg_model
    torch.cuda.empty_cache()
    return torch.cat(eeg_features_test, dim=0)


def gen_images(config, eeg_features_test, date_time):
    """执行图像生成任务"""
    device = config.device
    dtype = torch.float16

    ip_adapter_generator = IPAdapterGenerator(device=device, dtype=dtype)
    texts, _ = load_data()  # 加载用于命名的文本标签
    
    # 获取当前项目目录名
    current_dir = os.path.basename(os.getcwd())
    
    # 解析 date_time (格式: MMDD_HHMMSS) 为 月日 和 时分秒
    date_part = date_time.split('_')[0]  # MMDD
    time_part = date_time.split('_')[1]  # HHMMSS
    
    # 构建目录结构：../Gen/项目名/MMDD/HHMMSS/
    gen_base_dir = os.path.join("..", "Gen", current_dir, date_part, time_part)
    gen_dir = gen_base_dir
    os.makedirs(gen_dir, exist_ok=True)
    
    # 生成图像
    eeg_features_test = eeg_features_test.unsqueeze(1)
    for k in tqdm(range(len(eeg_features_test))):
        eeg_embeds = eeg_features_test[k : k + 1].to(device)
        path = f"{gen_dir}/{texts[k]}"
        os.makedirs(path, exist_ok=True)
        images = ip_adapter_generator.generate(eeg_embeds, config.gen_num)
        for i, image in enumerate(images):
            image.save(f"{path}/{i}.png")
            print(f"图像已保存到 {path}/{i}.png")

    print("\n" + "=" * 50)
    print("✅ 图像生成完成！")
    print("=" * 50)
    del ip_adapter_generator
    torch.cuda.empty_cache()
    return gen_dir


def eval_images(config, gen_dir, date_time):
    device = config.device
    eval_dir = config.eval_dir
    eval_dir = os.path.join(eval_dir, date_time)
    os.makedirs(eval_dir, exist_ok=True)
    # 更新配置中的输出目录路径
    config.results_csv_path = os.path.join(eval_dir, "metrics.csv")
    config.comparison_image_path = os.path.join(
        eval_dir,
        "samples_comparison.png",
    )

    # Step 1: Preprocess or load image tensors
    gt_images, gt_labels = process_images(
        source_dir=config.gt_source_dir,
        imsize=config.imsize,
    )
    gt_images = gt_images.to(device)

    recon_images, recon_labels = process_images(
        source_dir=gen_dir,
        imsize=config.imsize,
    )
    if len(recon_images) != len(gt_images):
        # group into K groups
        recon_images = recon_images.view(
            len(gt_images), config.gen_num, *gt_images.shape[1:]
        ).permute(1, 0, 2, 3, 4)

    recon_images = recon_images.to(device)

    print(f"Loaded Ground Truth images tensor with shape: {gt_images.shape}")
    print(f"Loaded Reconstructed images tensor with shape: {recon_images.shape}")

    # Step 2:Display sample comparisons
    num_samples = config.comparison_num_samples
    display_samples(
        gt_images.cpu(),
        recon_images[0].cpu(),
        gt_labels,
        num_samples,
        config.imsize,
        config.comparison_image_path,
        not config.use_english,
    )

    # Step 3: Calculate all metrics
    print(f"\n=== 使用单次指标计算 ===")
    all_metrics = [{} for _ in range(config.gen_num)]
    metric_functions = [
        (calculate_pixel_correlation, False),  # (function, needs_device)
        (calculate_structural_similarity, False),
        (calculate_alexnet_metrics, True),
        (calculate_inception_accuracy, True),
        (calculate_clip_accuracy, True),
        (calculate_swav_distance, True),
        # (calculate_effnet_distance, True),
    ]
    for i in range(config.gen_num):
        for func, needs_device in metric_functions:
            try:
                if needs_device:
                    results = func(recon_images[i], gt_images, device)
                else:
                    results = func(recon_images[i], gt_images)
                all_metrics[i].update(results)
            except Exception as e:
                print(
                    f"Error calculating metric with {func.__name__}: {e}",
                    file=sys.stderr,
                )

    results_df = pd.DataFrame(all_metrics)

    # (可选但推荐) 为了清晰，给每一行（每一组）一个明确的名称
    run_labels = [f"Run {i}" for i in range(config.gen_num)]
    results_df.index = run_labels

    # 检查是否有足够的数据来计算平均值
    if len(results_df) > 0:
        # 计算每一列（每个指标）的平均值
        average_metrics = results_df.mean()
        # 使用 .loc 添加一个名为 'Average' 的新行
        results_df.loc["Average"] = average_metrics

    # Step 4: Report and save results
    print("\n--- 评估结果 ---")
    # 使用 to_string() 打印完整的、格式优美的表格
    print(results_df.to_string(float_format="%.3f"))

    # 保存结果到文件时，我们现在希望把行名（'Run 0', 'Run 1', 'Average'）也保存进去
    # 所以将 index=False 改为 index=True
    results_df.to_csv(
        config.results_csv_path,
        sep="\t",
        index=True,  # <-- 注意这里的变化
        float_format="%.3f",
    )
    print(f"\n结果已包含平均值，并保存到: {config.results_csv_path}")


if __name__ == "__main__":

    config = load_config("config/gen_eval.yaml")
    date_time = datetime.datetime.now().strftime("%m%d_%H%M%S")
    if not config.only_eval:
        # step1: extract eeg features
        eeg_features_test = extract_eeg_features(config)

        # step2: generate images
        gen_dir = gen_images(config, eeg_features_test, date_time)
    else:
        # 在only_eval模式下，使用配置文件中指定的gen_dir目录
        gen_dir = config.gen_dir

    # step3: evaluate images
    eval_images(config, gen_dir, date_time)
