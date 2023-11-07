# HIDRO-VQA : High Dynamic Range Oracle for Video Quality Assessment

Shreshth Saini*, Avinab Saha*, and Alan C. Bovik

(* denotes equal contribution)

This is the official repository of the paper [HIDRO-VQA : High Dynamic Range Oracle for Video Quality Assessment](https://arxiv.org/abs/2110.13266)

## Usage
The code has been tested on Linux systems with python 3.9. Please refer to [requirements.txt](requirements.txt) for installing dependent packages.

### Running HIDRO-VQA
In order to obtain quality-aware features or to start the HDR quality-aware fine-tuning, checkpoints need to be downloaded.  Download the checkpoint folder from the [link](https://drive.google.com/drive/folders/1wuakzvupOxwVv9Sa3Ta0IKjkSBPqG8MG?usp=sharing) and save them to the checkpoints folder.


### Obtaining HIDRO-VQA Features on the LIVE HDR Database
For obtaining HIDRO-VQA features, the following command can be used. The features are saved in '.npy' format. It assumes that the videos are stored raw YUV 10-bit format, upscaled to 4K. Please change path of the videos in line 97.
```
python demo_hidro_vqa_feats.py
```

## Training HIDRO-VQA
### Data Processing (TODO)


### Training Model

Training with multiple GPUs using Distributed training

Run the following commands on different terminals concurrently. Please update folder location of HDR Frames in data_loader file located in modules/data_loader.py (Lines 119,137).
```
CUDA_VISIBLE_DEVICES=0 python train.py --nodes 3 --nr 0 --batch_size 256 --lr 0.1 --epochs 25
CUDA_VISIBLE_DEVICES=1 python train.py --nodes 3 --nr 1 --batch_size 256 --lr 0.1 --epochs 25
CUDA_VISIBLE_DEVICES=2 python train.py --nodes 3 --nr 2 --batch_size 256 --lr 0.1 --epochs 25

```
Note that in distributed training, ```batch_size``` value will be the number of images to be loaded on each GPU. 

### Training Regressor
After HIDRO_VQA model training is complete, a SVR is trained using HIDRO_VQA features and corresponding ground truth quality scores from LIVE-HDR using the following command. It assumes features from each video is extracted (per frame) and stored in a numpy files in a folder. 

```
python train_svr.py --score_file <score_csv_file> --feature_folder <feature_folder_path> --train_and_test
```

## Contact
Correspondence to : Avinab Saha (avinab.saha@utexas.edu) and Shreshth Saini (saini.2@utexas.edu)

## Citation
```

```
