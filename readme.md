This project is the official code of the paper [DPDTRN: A Dynamic Pixel-level Difficulty-aware Texture Reconstruction Network for Document Super-Resolution].

## Folder structure

The project folder for this section should be placed in the following structure:

```
DPDTRN
├── data
├── dataset
├── loss
├── model	
├── models
├── results	
├── utils	
├── readme.md
├── requirements.txt
├── test_DPDTRN.py	
├── train_DPDTRN_x2.py 	
├── train_DPDTRN_x4.py	  
├── train_DPDTRN_x8.py
```

## Requirements

Required environment

1. Python   3.9.19
2. torch    2.1.2
3. numpy    1.26.3
4. cuda     12.4

Install the environment with following commands.

```
conda create -n dpdtrn_env python==3.9.19
conda activate dpdtrn_env
pip install -r requirements.txt
```

## prepare data

1.Download the original Text330 datastet from https://github.com/t2mhanh/LapSRN_TextImages_git

2.Download the cropped Text330 datastet from https://pan.baidu.com/s/1RWla_jNNYqSCHgmfapLvOA with code 8l3d. This is our
training datatset. Please place the downloaded dataset folder in the path of this project [DPDTRN/data/Text330/train].

## train

Run following commands to train

```
python train_DPDTRN_x2.py
python train_DPDTRN_x4.py
python train_DPDTRN_x8.py
NOTE：Please find trained models in DPDTRN/results/.
```

## test

Run following commands to test and verify

```
python test_DPDTRN.py  
NOTE：Please find testing results in DPDTRN/results/.
```

