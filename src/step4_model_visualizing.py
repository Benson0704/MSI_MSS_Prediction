'''
Description:
    This script provides functions to visualize model using GradCAM
Author:
    Benson0704@outlook.com
Version:
    origin
WARNING:
    If model re-training is needed, dataset must be added manually as comments below
'''
import cv2
import os
import numpy as np
import torch
import torchvision.transforms as transforms
import json


def visualize(out_dir, name):
    HYPER_PARAMETERS = open('../hyper_parameters.json', 'r')
    HYPER_PARAMETERS = json.load(HYPER_PARAMETERS)

    def img_preprocess(img_in):
        img = img_in.copy()
        img = img[:, :, ::-1]
        img = np.ascontiguousarray(img)
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        img = transform(img)
        img = img.unsqueeze(0)
        return img

    def backward_hook(module, grad_in, grad_out):
        grad_block.append(grad_out[0].detach())

    def farward_hook(module, input, output):
        fmap_block.append(output)

    def cam_show_img(img, feature_map, grads, out_dir, name):
        H, W, _ = img.shape
        cam = np.zeros(feature_map.shape[1:], dtype=np.float32)
        grads = grads.reshape([grads.shape[0], -1])
        weights = np.mean(grads, axis=1)
        for i, w in enumerate(weights):
            cam += w * feature_map[i, :, :]
        cam = np.maximum(cam, 0)
        cam = cam / cam.max()
        cam = cv2.resize(cam, (W, H))
        cv2.imwrite(os.path.join(out_dir,
                                 name.split('/')[-1]), img)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        cam_img = 0.3 * heatmap + 0.7 * img

        path_cam_img = os.path.join(out_dir, "Edited-" + name.split('/')[-1])
        cv2.imwrite(path_cam_img, cam_img)
    path_img = name
    output_dir = '../results/cam/{}'.format(out_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(HYPER_PARAMETERS['GPU'])
    fmap_block = list()
    grad_block = list()
    img = cv2.imread(path_img, 1)

    img_input = img_preprocess(img)

    net = torch.load("../models/" + '' + '.model')  # load your model here
    net.eval()
    net.layer4.register_forward_hook(farward_hook)
    net.layer4.register_backward_hook(backward_hook)

    output = net(img_input.to(device))
    idx = np.argmax(output.cpu().data.numpy())
    net.zero_grad()
    class_loss = output[0, idx]
    class_loss.backward()

    grads_val = grad_block[0].cpu().data.numpy().squeeze()
    fmap = fmap_block[0].cpu().data.numpy().squeeze()

    cam_show_img(img, fmap, grads_val, output_dir, name)


if __name__ == '__main__':
    '''
    Now you can add your dataset
    this python list should append the tile's path
        like: data=['1.png','2.png']
    '''
    data = []
    for line in data:
        visualize('test', line)
    print('done!')
