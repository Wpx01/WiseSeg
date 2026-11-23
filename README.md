# WiseSeg

WiseSeg segments the CT component of PET/CT examinations using nine pretrained 3D residual U-Net models. It covers **147 anatomical structures represented by 146 foreground label IDs** in one multi-label NIfTI mask.

**Why 146 labels for 147 structures?** Prostate and uterus share **label 138** as a sex-specific class. All subsequent G7 labels follow consecutively from 139 to 146. Label 0 is background. This is intentional; no output channel or checkpoint is missing.

This repository provides inference code, pretrained weights, label definitions, and training parameter documentation. Training and evaluation scripts are not included.

**Paper:** Wen P, He J, Qiao L, et al. *WiseSeg: weakly supervised whole-body PET/CT segmentation of 147 anatomical structures*. EJNMMI Physics (2026). [DOI: 10.1186/s40658-026-00934-9](https://doi.org/10.1186/s40658-026-00934-9).

## Installation

Use a 64-bit Python **3.11 or 3.12** environment. The release dependencies are pinned in [requirements.txt](requirements.txt). Checkpoint loading and compact synthetic inference have been tested on Windows with Python 3.12 and CPU PyTorch. The commands below also provide a CUDA installation option.

Clone the repository and create an environment:

~~~bash
git clone https://github.com/Wpx01/WiseSeg.git
cd WiseSeg
python -m venv .venv
~~~

Activate it on Windows PowerShell:

~~~powershell
.\.venv\Scripts\Activate.ps1
~~~

Or on Linux:

~~~bash
source .venv/bin/activate
~~~

Install **one** PyTorch build. For an NVIDIA GPU compatible with CUDA 12.4:

~~~bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
~~~

For CPU execution:

~~~bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
~~~

Then install the remaining dependencies:

~~~bash
python -m pip install -r requirements.txt
python segment.py --check-weights
~~~

PyTorch installation details are available in the [official previous-version guide](https://pytorch.org/get-started/previous-versions/). The nine checkpoint files are included in <code>group_weight/</code>; no separate model download or registration is required.

## Quick start

Segment a single CT:

~~~bash
python segment.py -i /path/to/ct.nii.gz -o /path/to/output
~~~

Segment every .nii or .nii.gz file directly inside a directory:

~~~bash
python segment.py -i /path/to/ct_directory -o /path/to/output
~~~

Windows example:

~~~powershell
python segment.py -i "D:\data\ct.nii.gz" -o "D:\results\WiseSeg"
~~~

The default device is CUDA when available, otherwise CPU. Select a device explicitly if needed:

~~~bash
python segment.py -i ct.nii.gz -o output --device cuda:0
python segment.py -i ct.nii.gz -o output --device cpu
~~~

Additional controls:

| Option | Meaning |
| --- | --- |
| --weights-dir PATH | Use another directory containing the same released checkpoints. |
| --output-device cpu | Accumulate full-volume window predictions in host memory; default. |
| --output-device cuda | Accumulate on the selected CUDA device; requires more GPU memory. |
| --sw-batch-size 1 | Process fewer windows at a time to reduce inference memory usage. |
| --num-threads N | Set the PyTorch CPU thread count. |
| --overwrite | Replace existing output masks; otherwise the program stops before inference. |
| --check-weights | Verify SHA256 and strict loading of all nine checkpoints, then exit. |
| --list-labels | Print the output label mapping, then exit. |
| --help | Show the complete command-line interface. |

Full-volume inference is memory intensive. For a 25-channel group on the fixed 512 x 512 x 448 grid, the float32 accumulation tensor alone occupies about 11 GiB. Leave additional memory for inputs, window predictions, and resampling. Reducing --sw-batch-size lowers window-processing memory, but does not shrink that accumulation tensor. CPU execution is supported but can be slow.

## Input and output

- **Input:** one scalar 3D CT volume in NIfTI format (.nii or .nii.gz), with CT intensities in Hounsfield units and valid spatial metadata. Use the CT component of PET/CT, not a PET/SUV volume or a fused display image.
- **Coverage:** the model was developed for whole-body PET/CT CT images. Keep the acquisition's anatomical axis order; the pipeline does not perform automatic reorientation or registration.
- **Directory mode:** reads only the immediate input directory, without recursion. Use a separate output directory.
- **Output:** <code>&lt;input_name&gt;_mask.nii.gz</code>, a single unsigned 16-bit label image with the original CT size, spacing, origin, and direction. Values are 0 through 146.
- **Shared class:** label 138 is <code>prostate_or_uterus</code>. Interpret it using known patient anatomy or metadata. The model does not infer patient sex and does not produce separate prostate and uterus masks.
- **Other labels:** not every label is expected in every scan. Anatomical absence and limited field of view can lead to absent labels.
- DICOM conversion, PET resampling, SUV calculation, and clinical evaluation are outside this inference package.

## Anatomical groups

| Model group | Paper group | Coverage | Foreground IDs | Label count | Anatomical structure count |
| --- | --- | --- | --- | ---: | ---: |
| G1 | G1 | Vertebrae | 1-24 | 24 | 24 |
| G2 | G2 | Ribs | 25-48 | 24 | 24 |
| G3 | G3 | Other bones and selected muscles | 49-72 | 24 | 24 |
| G4_1 | G4_1 | Abdominal organs | 73-85 | 13 | 13 |
| G4_2 | G4_2 | Thoracic, cervical, and neural structures | 86-95 | 10 | 10 |
| G5_LR | G5_1 | Paired vessels | 96-105 | 10 | 10 |
| G5_NoLR | G5_2 | Cardiac structures and other vessels | 106-118 | 13 | 13 |
| G6 | G6 | Head and neck structures | 119-137 | 19 | 19 |
| G7 | G7 | Supplementary structures, including the shared prostate/uterus class | 138-146 | 9 | 10 |
| Total | | | 1-146 | 146 | 147 |

G7 has **10 network output channels: background plus nine foreground labels**. The shared class is counted as two anatomical structures in the paper and one foreground ID in the output.

### G7 label interpretation

| Global ID | Local G7 ID | Structure |
| --- | --- | --- |
| 138 | 1 | Prostate or uterus |
| 139 | 2 | Left sternocleidomastoid muscle |
| 140 | 3 | Right sternocleidomastoid muscle |
| 141 | 4 | Left internal jugular vein |
| 142 | 5 | Right internal jugular vein |
| 143 | 6 | Left renal vein |
| 144 | 7 | Right renal vein |
| 145 | 8 | Left testis |
| 146 | 9 | Right testis |

### Complete label list

The complete machine-readable mapping is included in the labels field of [config.json](config.json), together with local group IDs and the two anatomical names associated with label 138.

<details>
<summary>Show all 146 foreground labels and background</summary>

| Label ID | Name | Model group | Local ID |
| ---: | --- | --- | ---: |
| 0 | background | - | 0 |
| 1 | vertebrae_C1 | G1 | 1 |
| 2 | vertebrae_C2 | G1 | 2 |
| 3 | vertebrae_C3 | G1 | 3 |
| 4 | vertebrae_C4 | G1 | 4 |
| 5 | vertebrae_C5 | G1 | 5 |
| 6 | vertebrae_C6 | G1 | 6 |
| 7 | vertebrae_C7 | G1 | 7 |
| 8 | vertebrae_T1 | G1 | 8 |
| 9 | vertebrae_T2 | G1 | 9 |
| 10 | vertebrae_T3 | G1 | 10 |
| 11 | vertebrae_T4 | G1 | 11 |
| 12 | vertebrae_T5 | G1 | 12 |
| 13 | vertebrae_T6 | G1 | 13 |
| 14 | vertebrae_T7 | G1 | 14 |
| 15 | vertebrae_T8 | G1 | 15 |
| 16 | vertebrae_T9 | G1 | 16 |
| 17 | vertebrae_T10 | G1 | 17 |
| 18 | vertebrae_T11 | G1 | 18 |
| 19 | vertebrae_T12 | G1 | 19 |
| 20 | vertebrae_L1 | G1 | 20 |
| 21 | vertebrae_L2 | G1 | 21 |
| 22 | vertebrae_L3 | G1 | 22 |
| 23 | vertebrae_L4 | G1 | 23 |
| 24 | vertebrae_L5 | G1 | 24 |
| 25 | rib_right_1 | G2 | 1 |
| 26 | rib_right_2 | G2 | 2 |
| 27 | rib_right_3 | G2 | 3 |
| 28 | rib_right_4 | G2 | 4 |
| 29 | rib_right_5 | G2 | 5 |
| 30 | rib_right_6 | G2 | 6 |
| 31 | rib_right_7 | G2 | 7 |
| 32 | rib_right_8 | G2 | 8 |
| 33 | rib_right_9 | G2 | 9 |
| 34 | rib_right_10 | G2 | 10 |
| 35 | rib_right_11 | G2 | 11 |
| 36 | rib_right_12 | G2 | 12 |
| 37 | rib_left_1 | G2 | 13 |
| 38 | rib_left_2 | G2 | 14 |
| 39 | rib_left_3 | G2 | 15 |
| 40 | rib_left_4 | G2 | 16 |
| 41 | rib_left_5 | G2 | 17 |
| 42 | rib_left_6 | G2 | 18 |
| 43 | rib_left_7 | G2 | 19 |
| 44 | rib_left_8 | G2 | 20 |
| 45 | rib_left_9 | G2 | 21 |
| 46 | rib_left_10 | G2 | 22 |
| 47 | rib_left_11 | G2 | 23 |
| 48 | rib_left_12 | G2 | 24 |
| 49 | skull | G3 | 1 |
| 50 | sternum | G3 | 2 |
| 51 | clavicle_right | G3 | 3 |
| 52 | clavicle_left | G3 | 4 |
| 53 | scapula_right | G3 | 5 |
| 54 | scapula_left | G3 | 6 |
| 55 | costal_cartilage | G3 | 7 |
| 56 | sacrum | G3 | 8 |
| 57 | humerus_right | G3 | 9 |
| 58 | humerus_left | G3 | 10 |
| 59 | femur_right | G3 | 11 |
| 60 | femur_left | G3 | 12 |
| 61 | pelvic_bone_right | G3 | 13 |
| 62 | pelvic_bone_left | G3 | 14 |
| 63 | gluteus_maximus_right | G3 | 15 |
| 64 | gluteus_maximus_left | G3 | 16 |
| 65 | gluteus_medius_right | G3 | 17 |
| 66 | gluteus_medius_left | G3 | 18 |
| 67 | gluteus_minimus_right | G3 | 19 |
| 68 | gluteus_minimus_left | G3 | 20 |
| 69 | erector_spinae_right | G3 | 21 |
| 70 | erector_spinae_left | G3 | 22 |
| 71 | iliopsoas_right | G3 | 23 |
| 72 | iliopsoas_left | G3 | 24 |
| 73 | stomach | G4_1 | 1 |
| 74 | liver | G4_1 | 2 |
| 75 | gallbladder | G4_1 | 3 |
| 76 | pancreas | G4_1 | 4 |
| 77 | duodenum | G4_1 | 5 |
| 78 | small_intestine | G4_1 | 6 |
| 79 | large_intestine | G4_1 | 7 |
| 80 | kidney_left | G4_1 | 8 |
| 81 | kidney_right | G4_1 | 9 |
| 82 | urinary_bladder | G4_1 | 10 |
| 83 | adrenal_gland_left | G4_1 | 11 |
| 84 | adrenal_gland_right | G4_1 | 12 |
| 85 | spleen | G4_1 | 13 |
| 86 | esophagus | G4_2 | 1 |
| 87 | lung_upper_lobe_left | G4_2 | 2 |
| 88 | lung_upper_lobe_right | G4_2 | 3 |
| 89 | lung_lower_lobe_left | G4_2 | 4 |
| 90 | lung_middle_lobe_right | G4_2 | 5 |
| 91 | lung_lower_lobe_right | G4_2 | 6 |
| 92 | trachea | G4_2 | 7 |
| 93 | thyroid | G4_2 | 8 |
| 94 | brain | G4_2 | 9 |
| 95 | spinal_cord | G4_2 | 10 |
| 96 | common_carotid_artery_left | G5_LR | 1 |
| 97 | common_carotid_artery_right | G5_LR | 2 |
| 98 | subclavian_artery_left | G5_LR | 3 |
| 99 | subclavian_artery_right | G5_LR | 4 |
| 100 | common_iliac_artery_left | G5_LR | 5 |
| 101 | common_iliac_artery_right | G5_LR | 6 |
| 102 | brachiocephalic_vein_left | G5_LR | 7 |
| 103 | brachiocephalic_vein_right | G5_LR | 8 |
| 104 | common_iliac_vein_left | G5_LR | 9 |
| 105 | common_iliac_vein_right | G5_LR | 10 |
| 106 | myocardium | G5_NoLR | 1 |
| 107 | atrium_left | G5_NoLR | 2 |
| 108 | atrium_right | G5_NoLR | 3 |
| 109 | ventricle_left | G5_NoLR | 4 |
| 110 | ventricle_right | G5_NoLR | 5 |
| 111 | pulmonary_artery | G5_NoLR | 6 |
| 112 | left_atrial_appendage | G5_NoLR | 7 |
| 113 | aorta | G5_NoLR | 8 |
| 114 | brachiocephalic_trunk | G5_NoLR | 9 |
| 115 | superior_vena_cava | G5_NoLR | 10 |
| 116 | inferior_vena_cava | G5_NoLR | 11 |
| 117 | splenic_and_portal_vein | G5_NoLR | 12 |
| 118 | pulmonary_vein | G5_NoLR | 13 |
| 119 | brainstem | G6 | 1 |
| 120 | lens_left | G6 | 2 |
| 121 | lens_right | G6 | 3 |
| 122 | vitreous_body_left | G6 | 4 |
| 123 | vitreous_body_right | G6 | 5 |
| 124 | optic_nerve_left | G6 | 6 |
| 125 | optic_nerve_right | G6 | 7 |
| 126 | optic_chiasm | G6 | 8 |
| 127 | parotid_gland_left | G6 | 9 |
| 128 | parotid_gland_right | G6 | 10 |
| 129 | submandibular_gland_left | G6 | 11 |
| 130 | submandibular_gland_right | G6 | 12 |
| 131 | laryngeal_glottis | G6 | 13 |
| 132 | supraglottic_larynx | G6 | 14 |
| 133 | arytenoid_muscle | G6 | 15 |
| 134 | oral_cavity | G6 | 16 |
| 135 | lips | G6 | 17 |
| 136 | cricopharyngeal_inlet | G6 | 18 |
| 137 | mandible | G6 | 19 |
| 138 | prostate_or_uterus | G7 | 1 |
| 139 | sternocleidomastoid_muscle_left | G7 | 2 |
| 140 | sternocleidomastoid_muscle_right | G7 | 3 |
| 141 | internal_jugular_vein_left | G7 | 4 |
| 142 | internal_jugular_vein_right | G7 | 5 |
| 143 | renal_vein_left | G7 | 6 |
| 144 | renal_vein_right | G7 | 7 |
| 145 | testis_left | G7 | 8 |
| 146 | testis_right | G7 | 9 |

</details>

## Inference pipeline

1. Load the CT without changing its anatomical axis order.
2. Apply group-specific HU clipping and fixed normalization.
3. Resize the complete CT matrix to 512 x 512 x 448 using trilinear interpolation.
4. Run each residual U-Net using Gaussian-weighted sliding-window inference.
5. Take the class with the highest merged logit, then restore the original matrix size with nearest-neighbor interpolation.
6. Map local classes to fixed global IDs and merge groups in configuration order. Later nonzero group predictions overwrite earlier predictions where they overlap.

All network definitions, HU thresholds, normalization values, patch sizes, overlap values, and offsets are stored in [config.json](config.json). The script reads this file directly. The original checkpoints and inference settings are retained. CPU accumulation is the default to reduce GPU memory use; --output-device cuda keeps accumulation on the GPU. Argmax is computed in slabs to avoid allocating an additional full-volume softmax tensor.

## Training parameters

The following settings describe the training procedure reported in the WiseSeg paper. They document the pretrained models; this repository contains inference code and weights, without the training pipeline.

### Reported training setup

| Setting | Value |
| --- | --- |
| Segmentation input | CT component of PET/CT |
| Anatomical coverage | 147 structures encoded by 146 foreground labels |
| Grouping | Seven major groups, implemented as nine subgroup models |
| Pseudo-label generator | nnU-Net |
| Pseudo-label generator source data | 1,204 TotalSegmentator CTs; 500 AMOS CTs; 42 HaN-Seg CTs; 100 supplementary annotated PET/CT scans |
| Weakly supervised training data | 2,000 PET/CT scans, using CT images and generated pseudo-labels |
| Independent test set | 110 PET/CT scans |
| Healthy reference cohort | 98 PET/CT scans |
| Main network | MONAI 3D residual U-Net |
| Reported training patch size | 256 x 256 x 128 voxels |
| Training batch size | 2 |
| Epochs | 2,000 per subgroup |
| Optimizer | Adam |
| Initial learning rate | 0.01 |
| Internal validation | Five-fold cross-validation |
| Training strategy | Uncertainty-guided pseudo-label learning |
| Loss components | Uncertainty-weighted cross-entropy and Dice |

The uncertainty mechanism compares predictions from an original CT and a mildly Gaussian-noise-perturbed CT, with dropout contributing prediction variability. Voxel-wise KL divergence estimates uncertainty, which reduces unreliable contributions to cross-entropy; image-level average uncertainty modulates the Dice term.

This summary is not a complete executable training specification. Details not recovered from the original training environment are not assigned assumed values.

### Released network architectures

All models use monai.networks.nets.UNet, with three spatial dimensions, one input channel, and dropout 0.1. Output channels include background.

| Group | Foreground labels | Anatomical structures | Output channels | Channels | Strides | Residual units |
| --- | ---: | ---: | ---: | --- | --- | ---: |
| G1 | 24 | 24 | 25 | 4, 8, 16, 32, 64 | 2, 2, 2, 2 | 2 |
| G2 | 24 | 24 | 25 | 4, 8, 16, 32, 64, 128 | 2, 2, 2, 2, 2 | 4 |
| G3 | 24 | 24 | 25 | 4, 8, 16, 32, 64 | 2, 2, 2, 2 | 2 |
| G4_1 | 13 | 13 | 14 | 4, 8, 16, 32, 64, 128 | 2, 2, 2, 2, 2 | 4 |
| G4_2 | 10 | 10 | 11 | 4, 8, 16, 32, 64, 128 | 2, 2, 2, 2, 2 | 4 |
| G5_LR | 10 | 10 | 11 | 4, 8, 16, 32, 64, 128 | 2, 2, 2, 2, 2 | 4 |
| G5_NoLR | 13 | 13 | 14 | 4, 8, 16, 32, 64 | 2, 2, 2, 2 | 2 |
| G6 | 19 | 19 | 20 | 4, 8, 16, 32, 64 | 2, 2, 2, 2 | 2 |
| G7 | 9 | 10 | 10 | 4, 8, 16, 32, 64 | 2, 2, 2, 2 | 2 |

The shared prostate/uterus class is G7 local ID 1 and global ID 138. G7 therefore needs 10 output channels: one background channel and nine foreground channels.

### Released inference configuration

The complete configuration in [config.json](config.json) is read by the inference script. Intensities are clipped to the group's HU interval and transformed as (clipped_HU - norm_sub) / norm_div. Every group uses a fixed resized matrix of 512 x 512 x 448.

| Group | HU minimum | HU maximum | norm_sub | norm_div | Inference patch | Overlap | Window batch size |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| G1 | 8.0 | 1027.0 | 299.42 | 193.56 | 256 x 256 x 128 | 0.5 | 2 |
| G2 | -23.0 | 863.0 | 301.49 | 175.73 | 256 x 256 x 128 | 0.5 | 2 |
| G3 | -86.0 | 1499.0 | 212.21 | 328.92 | 256 x 256 x 128 | 0.5 | 2 |
| G4_1 | -990.0 | 76.0 | -31.53 | 200.7 | 256 x 256 x 128 | 0.7 | 2 |
| G4_2 | -946.0 | 92.0 | -519.78 | 361.34 | 256 x 256 x 128 | 0.5 | 2 |
| G5_LR | -40.0 | 227.0 | 37.623 | 42.989 | 256 x 256 x 128 | 0.7 | 2 |
| G5_NoLR | -224.0 | 81.0 | 27.98 | 63.44 | 256 x 256 x 128 | 0.5 | 2 |
| G6 | -132.0 | 1547.0 | 213.018 | 394.37 | 256 x 256 x 128 | 0.5 | 2 |
| G7 | -83.0 | 101.0 | 33.95 | 41.03 | 192 x 192 x 128 | 0.5 | 4 |

Window batch size controls inference windows per forward pass; it is independent of the training batch size. The dependency pins in requirements.txt define the release inference environment, not the historical training environment.

Reference: [WiseSeg paper](https://doi.org/10.1186/s40658-026-00934-9).

## Repository contents

This folder is ready to upload as the repository contents.

| Item | Purpose |
| --- | --- |
| README.md | Installation, usage, complete structure list, training parameters, and citation. |
| segment.py | Command-line inference. |
| config.json | Model settings, label definitions, and checkpoint checksums; required by the script. |
| requirements.txt | Python dependency versions. |
| LICENSE | Apache-2.0 license for the source code. |
| MODEL_LICENSE | CC BY 4.0 license for the pretrained models, including the full terms. |
| group_weight/ | Nine pretrained checkpoint files; keep all of them. |

## Validation and limitations

Release checks cover dependency installation, SHA256 integrity, strict loading and small-volume forward passes for all nine models, the shared label mapping, and compact synthetic end-to-end NIfTI inference. They do not establish clinical accuracy or reproduce the paper's full-resolution runtime and accuracy. GPU inference and full-size clinical scans require validation in the target environment.

Small vessels, head and neck structures, altered anatomy, and incomplete scan coverage can be challenging. Inspect outputs before downstream use. This research software has not been validated as a standalone clinical device.

## Citation

If you use WiseSeg, please cite:

~~~bibtex
@article{wen2026wiseseg,
  title   = {WiseSeg: weakly supervised whole-body PET/CT segmentation of 147 anatomical structures},
  author  = {Wen, Pengxu and He, Jian and Qiao, Lei and Chen, Jing and Tang, Lijun and Li, Aimei and Nie, Ziwei and Gui, Luying and Yang, Xiaoping},
  journal = {EJNMMI Physics},
  year    = {2026},
  doi     = {10.1186/s40658-026-00934-9}
}
~~~

## License

- **Source code:** [Apache License 2.0](LICENSE).
- **Pretrained weights in group_weight/:** [Creative Commons Attribution 4.0 International](MODEL_LICENSE).

Copyright 2026 WiseSeg authors. These licenses permit commercial use subject to their respective terms. The model attribution and full CC BY 4.0 terms are included in MODEL_LICENSE. Dependencies and referenced datasets retain their own licenses.

## Acknowledgments and support

WiseSeg builds on [MONAI](https://github.com/Project-MONAI/MONAI) and [PyTorch](https://pytorch.org/). The study used [TotalSegmentator](https://github.com/wasserth/TotalSegmentator), [AMOS](https://github.com/JunMa11/AMOS), and [HaN-Seg](https://han-seg2023.grand-challenge.org/) datasets, and nnU-Net for pseudo-label generation. These external tools and datasets are not bundled here.

For usage questions or issues, use the [GitHub issue tracker](https://github.com/Wpx01/WiseSeg/issues).
