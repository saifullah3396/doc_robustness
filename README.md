# Are Deep Models Robust against Real Distortions? A Case Study on Document Image Classification
This repository contains the datasets and code for the paper "Are Deep Models Robust against Real Distortions? A Case Study on Document Image Classification" by Saifullah, Shoaib Ahmed Siddiqui, Stefan Agne, Andreas Dengel, and Sheraz Ahmed.

Requires Python 3+. For evaluation, please download the data from the links below.

## Example Distortions:
<img align="center" src="assets/example_distortions.png" width="2000">

## Mean Corruption Error (mCE) and Relative mCE values:
<img align="center" src="assets/comparison.jpg" width="2000">

# RVL-CDIP-D:
[Download RVL-CDIP-D here.](link)

RVL-CDIP-D has 16 classes with images of size 1000xW just like the original RVL-CDIP dataset. Each sample image has 21 different augmentations of 5 severity levels.
<img align="center" src="assets/rvlcdip_results.png" width="2000">


## Tobacco3482-D:
[Download Tobacco3482-D here.](link)

Tobacco3482-D has 10 classes with images of size 1000xW. Each sample image has 21 different augmentations of 5 severity levels.

<img align="center" src="assets/tobacco3482_results.png" width="2000">

# Generating the datasets from scratch
Install the project dependencies.
```
pip install -r requirements.txt
```

## RVL-CDIP-D:
Set the DATASET_DIR to the root path of original RVL-CDIP dataset. 
```
export DATASET_DIR=</path/to/RVL-CDIP>
```
Set the DATASET_OUTPUT_DIR to the output path where the distorted dataset RVL-CDIP-D will be generated.
```
export DATASET_OUTPUT_DIR=</path/to/RVL-CDIP-D>
```
Run the augmentation script with the RVL-CDIP config.
```
 ./scripts/augment.sh --cfg ./cfg/rvlcdip-aug.yaml
```
## Tobacco3482-D:
Set the DATASET_DIR to the root path of original Tobacco3482 dataset. 
```
export DATASET_DIR=</path/to/Tobacco3482>
```

Set the DATASET_OUTPUT_DIR to the output path where the distorted dataset Tobacco3482-D will be generated. 
```
export DATASET_OUTPUT_DIR=</path/to/Tobacco3482-D>
```

Run the augmentation script with the Tobacco3482 config.
```
 ./scripts/augment.sh --cfg ./cfg/tobacco-aug.yaml
```

# Citation
If you find this useful in your research, please consider citing:
```
@article{saifullah2022doc-robustness,
  title={Are Deep Models Robust against Real Distortions? A Case Study on Document Image Classification},
  author={Saifullah, S. A. Siddiqui, s. Agne, A. Dengel, S. Ahmed},
  journal={ArXiv},
  year={2022}
}
```

# License
This repository is released under the Apache 2.0 license as found in the LICENSE file.
