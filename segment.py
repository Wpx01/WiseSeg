"""WiseSeg inference for 147 anatomical structures encoded by 146 labels.

Copyright 2026 WiseSeg authors.
SPDX-License-Identifier: Apache-2.0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
from monai.transforms import (
    Compose, EnsureChannelFirstd, LoadImaged, NormalizeIntensityd,
    Resized, ThresholdIntensityd, ToTensord,
)
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
LOGGER = logging.getLogger("wiseseg")


def read_json(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def load_configuration():
    config = read_json(ROOT / "config.json")
    labels = config["labels"]
    offset = 0
    for group in config["groups"]:
        if group["label_offset"] != offset:
            raise ValueError(f"Invalid label offset for {group['name']}.")
        if group["unet_params"]["out_channels"] != group["num_labels"] + 1:
            raise ValueError(f"Invalid output channel count for {group['name']}.")
        expected = list(range(offset + 1, offset + group["num_labels"] + 1))
        assigned = [item["id"] for item in labels if item["group"] == group["name"]]
        if assigned != expected:
            raise ValueError(f"Label mapping does not match {group['name']}.")
        offset += group["num_labels"]
    if offset != 146 or [item["id"] for item in labels] != list(range(147)):
        raise ValueError("Expected background 0 and foreground labels 1 through 146.")
    if labels[138]["anatomical_names"] != ["prostate", "uterus"]:
        raise ValueError("Label 138 must represent the shared prostate/uterus class.")
    return config, labels


def verify_weight_files(weights_dir: Path, config):
    for group in config["groups"]:
        path = weights_dir / group["weights_file"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {path}")
        if path.stat().st_size != group["weights_size_bytes"]:
            raise ValueError(f"Checkpoint size mismatch: {path.name}")
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != group["weights_sha256"]:
            raise ValueError(f"Checkpoint SHA256 mismatch: {path.name}")


def build_model(group, weights_dir: Path, device: torch.device):
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=group["unet_params"]["out_channels"],
        channels=tuple(group["unet_params"]["channels"]),
        strides=tuple(group["unet_params"]["strides"]),
        num_res_units=group["unet_params"]["num_res_units"],
        dropout=group["unet_params"]["dropout"],
    )
    state = torch.load(
        weights_dir / group["weights_file"], map_location="cpu", weights_only=True
    )
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def preprocess(ct_file: Path, group, resize_shape):
    """Use the original group-specific intensity and fixed-size transforms."""
    transform = Compose([
        LoadImaged(keys=["image"], image_only=True, reader="NibabelReader"),
        EnsureChannelFirstd(keys=["image"]),
        ThresholdIntensityd(
            keys=["image"], threshold=group["threshold_low"],
            above=True, cval=group["threshold_low"],
        ),
        ThresholdIntensityd(
            keys=["image"], threshold=group["threshold_high"],
            above=False, cval=group["threshold_high"],
        ),
        NormalizeIntensityd(
            keys=["image"], subtrahend=group["norm_sub"],
            divisor=group["norm_div"],
        ),
        Resized(keys=["image"], spatial_size=resize_shape, mode="trilinear"),
        ToTensord(keys=["image"]),
    ])
    image = transform({"image": str(ct_file)})["image"]
    if hasattr(image, "as_tensor"):
        image = image.as_tensor()
    return image.unsqueeze(0)


def logits_to_labels(logits: torch.Tensor) -> np.ndarray:
    """Return labels in SimpleITK array order (Z, Y, X), using small slabs.

    Argmax of logits equals argmax of softmax. Omitting softmax avoids a
    second full-volume floating-point allocation.
    """
    if logits.ndim != 5 or logits.shape[0] != 1:
        raise ValueError("Expected logits with shape (1, classes, X, Y, Z).")
    _, _, nx, ny, nz = logits.shape
    result = np.empty((nz, ny, nx), dtype=np.uint8)
    for start in range(0, nz, 16):
        end = min(start + 16, nz)
        slab = logits[:, :, :, :, start:end].argmax(dim=1)[0]
        result[start:end] = slab.to(dtype=torch.uint8, device="cpu").numpy().transpose(2, 1, 0)
    return result


def restore_original_grid(labels: np.ndarray, reference: sitk.Image) -> sitk.Image:
    """Preserve the original nearest-neighbor index-grid restoration."""
    image = sitk.GetImageFromArray(labels)
    old_size = image.GetSize()
    new_size = reference.GetSize()
    spacing = [old_size[i] / new_size[i] for i in range(3)]
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(spacing)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    restored = resampler.Execute(image)
    restored.CopyInformation(reference)
    return restored


def infer_group(ct_file, reference, group, config, weights_dir, device,
                output_device, sw_batch_size=None):
    model = build_model(group, weights_dir, device)
    try:
        inputs = preprocess(ct_file, group, config["resize_shape"])
        with torch.inference_mode():
            logits = sliding_window_inference(
                inputs=inputs,
                roi_size=group["patch_size"],
                sw_batch_size=sw_batch_size or group["sw_batch_size"],
                predictor=model,
                overlap=group["sw_overlap"],
                mode="gaussian",
                sw_device=device,
                device=output_device,
            )
            labels = logits_to_labels(logits)
        return restore_original_grid(labels, reference)
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def merge_group(merged: np.ndarray, image: sitk.Image, group):
    """Apply fixed global IDs; later nonzero predictions take precedence."""
    local = sitk.GetArrayFromImage(image)
    if local.shape != merged.shape or int(local.max()) > group["num_labels"]:
        raise ValueError(f"Invalid prediction shape or label range for {group['name']}.")
    foreground = local > 0
    merged[foreground] = local[foreground].astype(np.uint16) + group["label_offset"]


def read_reference(ct_file: Path):
    """Read NIfTI geometry with Unicode-path support on Windows."""
    template = nib.load(str(ct_file))
    if len(template.shape) != 3:
        raise ValueError(f"Expected a scalar 3D CT image: {ct_file.name}")
    affine_lps = np.diag([-1.0, -1.0, 1.0, 1.0]) @ template.affine
    spacing = np.linalg.norm(affine_lps[:3, :3], axis=0)
    if not np.isfinite(affine_lps).all() or np.any(spacing <= 0):
        raise ValueError("The input NIfTI affine must have finite, nonzero voxel spacing.")
    direction = affine_lps[:3, :3] / spacing
    if not np.allclose(direction.T @ direction, np.eye(3), atol=1e-4):
        raise ValueError("Sheared NIfTI affines are unsupported; use an orthogonal CT grid.")
    reference = sitk.Image([int(size) for size in template.shape], sitk.sitkUInt8)
    reference.SetSpacing(spacing.tolist())
    reference.SetOrigin(affine_lps[:3, 3].tolist())
    reference.SetDirection(direction.ravel().tolist())
    return reference, template


def write_mask(labels: np.ndarray, template, output_path: Path):
    """Write integer labels on the input grid, retaining both NIfTI forms."""
    header = template.header.copy()
    header.set_data_dtype(np.uint16)
    header.set_slope_inter(1.0, 0.0)
    header.set_intent("label")
    result = template.__class__(
        labels.transpose(2, 1, 0), template.affine, header=header
    )
    result.set_qform(template.get_qform(), int(template.header["qform_code"]))
    result.set_sform(template.get_sform(), int(template.header["sform_code"]))
    result.header.set_slope_inter(1.0, 0.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(result, str(output_path))


def segment_case(ct_file, output_path, config, weights_dir, device,
                 output_device, sw_batch_size=None):
    reference, template = read_reference(ct_file)
    merged = np.zeros(tuple(reversed(reference.GetSize())), dtype=np.uint16)
    for group in tqdm(config["groups"], desc=ct_file.name, unit="group"):
        image = infer_group(
            ct_file, reference, group, config, weights_dir, device,
            output_device, sw_batch_size,
        )
        merge_group(merged, image, group)
        del image
    write_mask(merged, template, output_path)
    LOGGER.info("Saved %s", output_path)


def is_nifti(path: Path):
    return path.name.lower().endswith((".nii", ".nii.gz"))


def case_stem(path: Path):
    return path.name[:-7] if path.name.lower().endswith(".nii.gz") else path.name[:-4]


def collect_inputs(input_path: Path):
    if input_path.is_file() and is_nifti(input_path):
        return [input_path]
    if input_path.is_dir():
        files = sorted(p for p in input_path.iterdir() if p.is_file() and is_nifti(p))
        if files:
            return files
    raise ValueError("Input must be a .nii/.nii.gz file or a directory containing such files.")


def resolve_device(name: str):
    if name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Supported devices are auto, cpu, cuda, and cuda:N.")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable. Install CUDA-enabled PyTorch or use --device cpu.")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device {device.index} is unavailable.")
    return device


def positive_integer(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer.")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="WiseSeg: 147 anatomical structures represented by 146 foreground labels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--input", type=Path, help="CT NIfTI file or directory.")
    parser.add_argument("-o", "--output", type=Path, help="Output directory for *_mask.nii.gz files.")
    parser.add_argument("--weights-dir", type=Path, default=ROOT / "group_weight",
                        help="Directory containing the nine released checkpoints.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N.")
    parser.add_argument("--output-device", choices=["cpu", "cuda"], default="cpu",
                        help="Device for full-volume sliding-window accumulation.")
    parser.add_argument("--sw-batch-size", type=positive_integer,
                        help="Override the number of windows per forward pass; use 1 to reduce memory.")
    parser.add_argument("--num-threads", type=positive_integer, default=min(8, os.cpu_count() or 1),
                        help="Number of PyTorch CPU threads.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output masks.")
    parser.add_argument("--check-weights", action="store_true",
                        help="Check SHA256 and strict CPU loading of all nine checkpoints, then exit.")
    parser.add_argument("--list-labels", action="store_true",
                        help="Print the complete output label table, then exit.")
    args = parser.parse_args(argv)
    if not (args.check_weights or args.list_labels) and (args.input is None or args.output is None):
        parser.error("--input and --output are required for inference.")
    return args


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        config, labels = load_configuration()
        if args.list_labels:
            print("label_id\tname\tgroup")
            for item in labels:
                print(f"{item['id']}\t{item['name']}\t{item['group'] or '-'}")
            return 0
        torch.set_num_threads(args.num_threads)
        verify_weight_files(args.weights_dir, config)
        if args.check_weights:
            for group in config["groups"]:
                model = build_model(group, args.weights_dir, torch.device("cpu"))
                del model
                LOGGER.info("%s: checksum and strict checkpoint loading passed.", group["name"])
            return 0
        device = resolve_device(args.device)
        if args.output_device == "cuda" and device.type != "cuda":
            raise ValueError("--output-device cuda requires a CUDA inference device.")
        output_device = device if args.output_device == "cuda" else torch.device("cpu")
        input_path, output_dir = args.input.resolve(), args.output.resolve()
        files = collect_inputs(input_path)
        if input_path.is_dir() and output_dir == input_path:
            raise ValueError("Use a different output directory to avoid including old masks as input.")
        outputs = [output_dir / f"{case_stem(path)}_mask.nii.gz" for path in files]
        names = [os.path.normcase(str(path)) for path in outputs]
        if len(set(names)) != len(names):
            raise ValueError("Two input files would produce the same output name.")
        for output in outputs:
            if output.exists() and not args.overwrite:
                raise FileExistsError(f"Output already exists: {output}. Use --overwrite to replace it.")
        LOGGER.info("Inference device: %s; accumulation device: %s; cases: %d",
                    device, output_device, len(files))
        for ct_file, output in zip(files, outputs):
            segment_case(ct_file, output, config, args.weights_dir, device,
                         output_device, args.sw_batch_size)
        return 0
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        LOGGER.error("%s", error)
        return 1


if __name__ == "__main__":
    sys.exit(main())

