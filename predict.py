from torch.utils.data import dataset
from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np

from torch.utils import data
from datasets import VOCSegmentation, Cityscapes, cityscapes, MyDataset
from torchvision import transforms as T
from metrics import StreamSegMetrics

import torch
import torch.nn as nn

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from glob import glob


def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="path to a single image or image directory",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        choices=["voc", "cityscapes"],
        help="Name of training set",
    )

    # Deeplab Options
    available_models = sorted(
        name
        for name in network.modeling.__dict__
        if name.islower()
        and not (name.startswith("__") or name.startswith("_"))
        and callable(network.modeling.__dict__[name])
    )

    parser.add_argument(
        "--model",
        type=str,
        default="deeplabv3plus_mobilenet",
        choices=available_models,
        help="model name",
    )
    parser.add_argument(
        "--separable_conv",
        action="store_true",
        default=False,
        help="apply separable conv to decoder and aspp",
    )
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument(
        "--save_val_results_to",
        default=None,
        help="save segmentation results to the specified dir",
    )

    parser.add_argument(
        "--crop_val",
        action="store_true",
        default=False,
        help="crop validation (default: False)",
    )
    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=4,
        help="batch size for validation (default: 4)",
    )
    parser.add_argument("--crop_size", type=int, default=513)

    parser.add_argument("--ckpt", default=None, type=str, help="resume from checkpoint")
    parser.add_argument("--gpu_id", type=str, default="0", help="GPU ID")
    return parser


def overlay_mask(image, mask, color=(255, 0, 0), alpha=0.5):
    """
    将掩码以指定颜色和透明度叠加到原图上
    Args:
    image (PIL.Image): 原始图像（RGB模式）
    mask (np.ndarray): 预测的掩码（0-1二值或多类索引）
    color (tuple/str): 掩码颜色，如 (255,0,0) 或 "red"
    alpha (float): 透明度 (0~1)
    Returns:
    PIL.Image: 叠加后的图像
    """
    # 将原图转换为RGBA模式以便添加透明度
    overlay = image.convert("RGBA")

    # 创建颜色层
    color_layer = Image.new("RGBA", overlay.size, color=color)

    # 根据掩码生成透明度层：掩码区域为alpha，其他区域为0
    if mask.ndim == 2:  # 二值掩码
        mask_array = (mask > 0).astype(np.uint8) * int(255 * alpha)
    else:  # 多分类掩码（假设mask为类别索引）
        mask_array = (mask > 0).astype(np.uint8) * int(255 * alpha)

    mask_image = Image.fromarray(mask_array, mode="L")

    # 将颜色层与原图混合
    overlay = Image.composite(color_layer, overlay, mask_image)

    # 合并图层并转换回RGB模式
    return overlay.convert("RGB")


def main():
    opts = get_argparser().parse_args()
    if opts.dataset.lower() == "voc":
        opts.num_classes = 21
        decode_fn = VOCSegmentation.decode_target
    elif opts.dataset.lower() == "cityscapes":
        opts.num_classes = 19
        decode_fn = Cityscapes.decode_target
    else:
        opts.num_classes = 2
        decode_fn = lambda pred: (pred > 0).astype(np.uint8) * 255  # 同上

    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: %s" % device)

    # Setup dataloader
    image_files = []
    if os.path.isdir(opts.input):
        for ext in ["png", "jpeg", "jpg", "JPEG",'tif']:
            files = glob(os.path.join(opts.input, "**/*.%s" % (ext)), recursive=True)
            if len(files) > 0:
                image_files.extend(files)
    elif os.path.isfile(opts.input):
        image_files.append(opts.input)

    # Set up model (all models are 'constructed at network.modeling)
    model = network.modeling.__dict__[opts.model](
        num_classes=opts.num_classes, output_stride=opts.output_stride
    )
    if opts.separable_conv and "plus" in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(opts.ckpt, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        print("Resume model from %s" % opts.ckpt)
        del checkpoint
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    # denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

    if opts.crop_val:
        transform = T.Compose(
            [
                T.Resize(opts.crop_size),
                T.CenterCrop(opts.crop_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    else:
        transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    if opts.save_val_results_to is not None:
        os.makedirs(opts.save_val_results_to, exist_ok=True)
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))  # 4x4 网格，显示 16 张图像
    axes = axes.ravel()  # 将二维数组展平为一维，方便遍历
    with torch.no_grad():
        model = model.eval()
        for i,img_path in enumerate(image_files[:16]):
            ext = os.path.basename(img_path).split(".")[-1]
            img_name = os.path.basename(img_path)[: -len(ext) - 1]
            img_orgin = Image.open(img_path).convert("RGB")
            img = img_orgin
            img = transform(img).unsqueeze(0)  # To tensor of NCHW
            img = img.to(device)

            pred = model(img).max(1)[1].cpu().numpy()[0]  # HW
            colorized_preds = decode_fn(pred).astype("uint8")
            # colorized_preds = Image.fromarray(colorized_preds).convert('L')
            result = overlay_mask(img_orgin, colorized_preds, color="red", alpha=1)
            axes[i].imshow(result)
            axes[i].set_title("test")  # 显示文件名作为标题
            axes[i].axis("off")  # 不显示坐标轴

            if opts.save_val_results_to:
                colorized_preds.save(
                    os.path.join(opts.save_val_results_to, img_name + ".png")
                )
                # 显示掩码图

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
