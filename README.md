# HIDRO-VQA: High Dynamic Range Oracle for Video Quality Assessment

This is the official repository of the paper [HIDRO-VQA](https://arxiv.org/abs/2311.11059)

[![arXiv](https://img.shields.io/badge/arXiv-2311.11059-b31b1b.svg)](https://arxiv.org/abs/2311.11059)

> HIDRO-VQA: High Dynamic Range Oracle for Video Quality Assessment  
> Shreshth Saini*, Avinab Saha*, and Alan C. Bovik  
> WACV 2024


## Usage
The code has been tested on Linux systems with Python 3.9. Please refer to [requirements.txt](requirements.txt) for installing dependent packages.

### Running HIDRO-VQA
In order to obtain quality-aware features or to start HDR quality-aware fine-tuning, checkpoints need to be downloaded.  Download the checkpoint folder from the [link](https://drive.google.com/drive/folders/1wuakzvupOxwVv9Sa3Ta0IKjkSBPqG8MG?usp=sharing) and save them to the checkpoints folder.


### Obtaining HIDRO-VQA Features on the LIVE HDR Database
For obtaining HIDRO-VQA features, the following command can be used. The features are saved in '.npy' format. It assumes the videos are stored in raw YUV 10-bit format, upscaled to 4K. Please change the path of the videos in line 97.
```
python demo_hidro_vqa_feats.py
```

## Training HIDRO-VQA
### Data Processing 

Please take a look at the DATA folder for steps on how to prepare data for HIDRO-VQA pre-training. 


### Training Model

Training with multiple GPUs using Distributed training

Run the following commands on different terminals concurrently. Please update the folder location of HDR Frames in the data_loader file located in modules/data_loader.py (Lines 119,137).
```
CUDA_VISIBLE_DEVICES=0 python train.py --nodes 3 --nr 0 --batch_size 256 --lr 0.1 --epochs 25
CUDA_VISIBLE_DEVICES=1 python train.py --nodes 3 --nr 1 --batch_size 256 --lr 0.1 --epochs 25
CUDA_VISIBLE_DEVICES=2 python train.py --nodes 3 --nr 2 --batch_size 256 --lr 0.1 --epochs 25

```
Note that in distributed training, ```batch_size``` value will be the number of images to be loaded on each GPU. 

### Training Regressor
After HIDRO-VQA model pre-training, an SVR is trained using HIDRO_VQA features and corresponding ground truth quality scores from LIVE-HDR using the following command. It assumes features from each video is extracted (per frame) and stored using individual numpy files in a folder. 

```
python train_svr.py --score_file <score_csv_file> --feature_folder <feature_folder_path> --train_and_test
```

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## Citation 
    @article{saini2023hidro,
    title={HIDRO-VQA: High Dynamic Range Oracle for Video Quality Assessment},
    author={Saini, Shreshth and Saha, Avinab and Bovik, Alan C},
    journal={arXiv preprint arXiv:2311.11059},
    year={2023}
    }


## License

This project is licensed under the [MIT License](LICENSE).
