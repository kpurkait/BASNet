import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms  # , utils
# import torch.optim as optim

import numpy as np
from PIL import Image
import glob
import argparse
import cv2

from data_loader import RescaleT
from data_loader import CenterCrop
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import BASNet


def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn


def save_output(image_name, pred, d_dir):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()
    predict_np = predict_np*255

    mask = cv2.cvtColor(predict_np, cv2.COLOR_GRAY2BGR).astype(np.uint8)
    orig_image = cv2.imread(image_name)
    mask = cv2.resize(
        mask, (orig_image.shape[1], orig_image.shape[0]), interpolation=cv2.INTER_LINEAR)

    masked_image = cv2.bitwise_and(orig_image, mask)
    masked_white_bg = cv2.bitwise_or(orig_image, 255-mask)

    img_tile = [[orig_image, mask],
                [masked_image, masked_white_bg]]

    img_tile = cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in img_tile])

    # pb_np = np.array(imo)
    img_name = image_name.split("/")[-1].rsplit(".", 1)[0]
    cv2.imwrite(d_dir+img_name+'.png', img_tile)


if __name__ == '__main__':
    # --------- 1. get image path and name ---------
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", help="input directory",
                        default='./test_data/test_images/')
    parser.add_argument("-o", "--output_dir", help="output directory",
                        default='./test_data/test_results/')
    args = parser.parse_args()

    image_dir = args.input_dir
    prediction_dir = args.output_dir
    os.makedirs(prediction_dir, exist_ok=True)

    model_dir = './saved_models/basnet_bsi/basnet.pth'

    img_name_list = glob.glob(image_dir + '*.jpg')

    # --------- 2. dataloader ---------
    # 1. dataload
    test_salobj_dataset = SalObjDataset(img_name_list=img_name_list, lbl_name_list=[
    ], transform=transforms.Compose([RescaleT(256), ToTensorLab(flag=0)]))
    test_salobj_dataloader = DataLoader(
        test_salobj_dataset, batch_size=1, shuffle=False, num_workers=1)

    # --------- 3. model define ---------
    print("...load BASNet...")
    net = BASNet(3, 1)
    net.load_state_dict(torch.load(model_dir))
    if torch.cuda.is_available():
        net.cuda()
    net.eval()

    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):

        print("inferencing:", img_name_list[i_test].split("/")[-1])

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1, d2, d3, d4, d5, d6, d7, d8 = net(inputs_test)

        # normalization
        pred = d1[:, 0, :, :]
        pred = normPRED(pred)

        # save results to test_results folder
        save_output(img_name_list[i_test], pred, prediction_dir)

        del d1, d2, d3, d4, d5, d6, d7, d8
