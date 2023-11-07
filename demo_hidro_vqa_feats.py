from glob import glob
import numpy as np
import cv2

import torch
from modules.network import get_network
from modules.CONTRIQUE_model import CONTRIQUE_model
from torchvision import transforms
import numpy as np

import os
import argparse
import pickle

def fread(fid, nelements, dtype):
     if dtype is np.str_:
         dt = np.uint8  # WARNING: assuming 8-bit ASCII for np.str!
     else:
         dt = dtype

     data_array = np.fromfile(fid, dt, nelements)
     data_array.shape = (nelements, 1)

     return data_array


def yuv_read(filename,tot_frames,height,width):

    file_object = open(filename)

    rgb_data = np.zeros([tot_frames,height,width,3],dtype=np.float32)

    for frame_num in range(0,tot_frames):

        file_object.seek(frame_num*height*width*3)
        y1 = fread(file_object,height*width,np.uint16)
        u1 = fread(file_object,height*width//4,np.uint16)
        v1 = fread(file_object,height*width//4,np.uint16)

        y = np.reshape(y1,(height,width))
        u = np.reshape(u1,(height//2,width//2)).repeat(2,axis=0).repeat(2,axis=1)
        v = np.reshape(v1,(height//2,width//2)).repeat(2,axis=0).repeat(2,axis=1)

        rgb_data[frame_num] = yuv2rgb_bt2020(y,u,v)
    
    file_object.close()
        
    return rgb_data

def yuv2rgb_bt2020(y,u,v):
    # cast to float32 for yuv2rgb in BT2020

    y = y.astype(np.float32)
    u = u.astype(np.float32)
    v = v.astype(np.float32)

    cb = u - 512
    cr = v - 512

    r = y+1.4747*cr
    g = y-0.1645*cb-0.5719*cr
    b = y+1.8814*cb

    r = (r-64)/(940-64)
    g = (g-64)/(940-64)
    b = (b-64)/(940-64)

    r = np.clip(r,0,1)
    g = np.clip(g,0,1)
    b = np.clip(b,0,1)

    frame = np.stack((r,g,b),2)

    return frame

def read_hdr_yuv(hdr_file):

    with open(hdr_file, 'rb') as f:
        yuv_data = f.read()
    f.close()

    width = 3840
    height = 2160

    
    # Calculate the size of a single frame in bytes
    frame_size = int((width * height * 3 / 2) * 2)  # Each pixel is 10 bits (2 bytes)

    # Split the YUV420p10le data into frames
    frames = [yuv_data[i:i + frame_size] for i in range(0, len(yuv_data), frame_size)]

    rgb_data = yuv_read(hdr_file,len(frames),height,width)

    return rgb_data
   

files = sorted(glob("/scratch/09032/saini_2/HDR_LIVE/fall2021_hdr_upscaled_yuv/*.yuv"))

# load CONTRIQUE Model
encoder = get_network('resnet50', pretrained=False)
model = CONTRIQUE_model(encoder, 2048)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load('checkpoints/checkpoint_HDR_FT_SDR_PT.tar', map_location=device.type))

model = model.to(device)
model.eval()

for file in files:

    print(file)

    rgb_numpy_array = read_hdr_yuv(file)

    feats = np.zeros([rgb_numpy_array.shape[0],4096])

    with torch.no_grad():

        for loop_idx in range(0,rgb_numpy_array.shape[0]):

            image = rgb_numpy_array[loop_idx]
            image_2 = cv2.resize(rgb_numpy_array[loop_idx] , dsize = None, fx=0.5,fy=0.5,interpolation=cv2.INTER_CUBIC)

            image = torch.from_numpy(image).permute(2,0,1).unsqueeze(0).cuda()
            image_2 = torch.from_numpy(image_2).permute(2,0,1).unsqueeze(0).cuda()

            _,_, _, _, model_feat, model_feat_2, _, _ = model(image, image_2)
            feat = np.hstack((model_feat.detach().cpu().numpy(),\
                                    model_feat_2.detach().cpu().numpy()))

            feats[loop_idx] = feat 
    
    
    np.save("./CONTRIQUE_feats_HDR_FineTuned_Pretrained_SDR/" + file[file.rfind("/")+1:-3]+"npy",feats)




    

