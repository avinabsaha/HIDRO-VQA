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
### Download Training Data
Create a directory ```mkdir training_data``` to store images used for training CONTRIQUE.
1. KADIS-700k : Download [KADIS-700k](http://database.mmsp-kn.de/kadid-10k-database.html) dataset and execute the supllied codes to generate synthetically distorted images. Store this data in the ```training_data/kadis700k``` directory.
2. AVA : Download [AVA](https://github.com/mtobeiyf/ava_downloader) dataset and store in the ```training_data/UGC_images/AVA_Dataset``` directory.
3. COCO : [COCO](https://cocodataset.org/#download) dataset contains 330k images spread across multiple competitions. We used 4 folders ```training_data/UGC_images/test2015, training_data/UGC_images/train2017, training_data/UGC_images/val2017, training_data/UGC_images/unlabeled2017``` for training.
4. CERTH-Blur : [Blur](https://mklab.iti.gr/results/certh-image-blur-dataset/) dataset images are stored in the ```training_data/UGC_images/blur_image``` directory.
5. VOC : [VOC](http://host.robots.ox.ac.uk:8080/pascal/VOC/voc2012/) images are stored in the ```training_data/UGC_images/VOC2012``` directory.

### Training Model

Training with multiple GPUs using Distributed training (Recommended)

Run the following commands on different terminals concurrently. Please update root location of HDR Frames in data_loader file located in modules/data_loader.py (Lines 119,137).
```
CUDA_VISIBLE_DEVICES=0 python train.py --nodes 3 --nr 0 --batch_size 256 --lr 0.1 --epochs 25
CUDA_VISIBLE_DEVICES=1 python train.py --nodes 3 --nr 1 --batch_size 256 --lr 0.1 --epochs 25
CUDA_VISIBLE_DEVICES=2 python train.py --nodes 3 --nr 2 --batch_size 256 --lr 0.1 --epochs 25

```
Note that in distributed training, ```batch_size``` value will be the number of images to be loaded on each GPU. 

### Training Linear Regressor
After CONTRIQUE model training is complete, a linear regressor is trained using CONTRIQUE features and corresponding ground truth quality scores using the following command.

```
python3 train_regressor.py --feat_path feat.npy --ground_truth_path scores.npy --alpha 0.1
```

## Contact
Correspondence to : Avinab Saha (avinab.saha@utexas.edu) and Shreshth Saini (saini.2@utexas.edu)

## Citation
```
@article{madhusudana2021st,
  title={Image Quality Assessment using Contrastive Learning},
  author={Madhusudana, Pavan C and Birkbeck, Neil and Wang, Yilin and Adsumilli, Balu and Bovik, Alan C},
  journal={arXiv:2110.13266},
  year={2021}
}
```
