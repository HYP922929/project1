import argparse
import cv2
import numpy as np
import torch
from torch.autograd import Function
import torch.nn.functional as F

import torch.nn as nn
from network_files.faster_rcnn_framework import FasterRCNN, FastRCNNPredictor
from backbone.resnet50_model import resnet50_backbone


class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        # print(self.model._modules.keys())
        # print(self.model._modules["backbone"]._modules.keys())
        # print(self.model._modules["backbone"]._modules["body"]._modules.keys())
        # print(self.model._modules["backbone"]._modules["body"]._modules["layer4"]._modules.keys())
        # print(self.model._modules["backbone"]._modules["fpn"]._modules.keys())
        # print(self.model._modules["backbone"]._modules["fpn"]._modules["layer_blocks"]._modules.keys())
        x = x.squeeze(0)
        target_index = None
        x, target_index = self.model.transform.resize(x,target_index)
        x = x.unsqueeze(0)
        x.requires_grad_(True)
        x = self.model.backbone.body.conv1(x)
        x = self.model.backbone.body.bn1(x)
        x = self.model.backbone.body.relu(x)
        x = self.model.backbone.body.maxpool(x)
        x = self.model.backbone.body.layer1(x)
        x1 = x

        x = self.model.backbone.body.layer2(x)
        x2 = x

        x = self.model.backbone.body.layer3(x)
        x3 = x

        x = self.model.backbone.body.layer4(x)
        x4 = x

        # x1 = nn.Conv2d(256, 256, 1)(x1)
        # # print(x1.shape)
        # x1 = x1 + F.interpolate(x1, size=[200,200], mode="nearest")
        # # print(x1.shape)
        # x1 = nn.Conv2d(256, 256, 3, padding=1)(x1)
        # # print(x1.shape)
        # x2 = nn.Conv2d(512, 256, 1)(x2)
        # # print(x2.shape)
        # x_2 = F.interpolate(x1, size=[100, 100], mode="nearest")
        # # x_2 = F.interpolate(x1, size=[200, 200], mode="nearest")
        # # print(x_2.shape)
        # # x2 = x2 + x_2
        # # print(x2.shape)
        # x2 = F.interpolate(x1, size=[200, 200], mode="nearest")
        # x2 = nn.Conv2d(256, 256, 3, padding=1)(x2)
        # # print(x2.shape)
        # x3 = nn.Conv2d(1024, 256, 1)(x3)
        # x3 = x3 + F.interpolate(x1, size=[50,50], mode="nearest")
        # x3 = F.interpolate(x1, size=[200, 200], mode="nearest")
        # x3 = nn.Conv2d(256, 256, 3, padding=1)(x3)
        # x4 = nn.Conv2d(2048, 256, 1)(x4)
        # x4 = x4 + F.interpolate(x1, size=[25,25], mode="nearest")
        # x4 =  F.interpolate(x1, size=[200, 200], mode="nearest")
        # x4 = nn.Conv2d(256, 256, 3, padding=1)(x4)

        # x = x4 + x3 + x2 + x1
        x = x1
        x.requires_grad_(True)
        x.register_hook(self.save_gradient)
        outputs += [x]

        # for name, module in self.model._modules.items():
        # for name, module in self.model._modules["backbone"]._modules["body"]._modules["layer4"]._modules.items():
        #     print(x.shape)
        #     print(name)
        #     # print(module)
        #
        #     x = module(x)
        #     # print(x)
        #     if name in self.target_layers:
        #         print(name)
        #         x.register_hook(self.save_gradient)
        #         outputs += [x]
        return outputs, x


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        # self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)
        self.feature_extractor = FeatureExtractor(self.model, target_layers)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        target_activations, x = self.feature_extractor(x)
        # print(self.model._modules.keys())

        # # for name, module in self.model._modules.items():
        # for name, module in self.model._modules["backbone"]._modules.items():
        #     if module == self.feature_module:
        #         target_activations, x = self.feature_extractor(x)
        #     elif "avgpool" in name.lower():
        #         x = module(x)
        #         x = x.view(x.size(0),-1)
        #     else:
        #         x = module(x)
        x = self.maxpool(x)
        # x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # print(x.shape)

        # print("这里应该有梯度")
        # print(x)

        return target_activations, x


def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = preprocessed_img.requires_grad_(True)
    # print(input.shape)
    return input


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite("cam_test_res1.jpg", np.uint8(255 * cam))


class GradCam:
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        # print(input)
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        # print("here")
        # print(output)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        # print(output.cpu().data.numpy())
        # print(np.argmax(output.cpu().data.numpy()))
        # print(index)
        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        # print(one_hot)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        # print(one_hot)
        # print(output)
        # if self.cuda:
        #     one_hot = torch.sum(one_hot.cuda() * output)
        # else:
        #     one_hot = torch.sum(one_hot * output)
        if self.cuda:
            one_hot = one_hot.cuda()

        one_hot = torch.sum(one_hot * output)

        # print(one_hot)
        self.feature_module.zero_grad()
        self.model.zero_grad()

        # print(one_hot)
        # one_hot.requires_grad_()
        one_hot.backward(retain_graph=True)

        # print(one_hot)
        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, input.shape[2:])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam


class GuidedBackpropReLU(Function):

    @staticmethod
    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input),
                                   torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output,
                                                 positive_mask_1), positive_mask_2)

        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        if self.cuda:
            self.model = model.cuda()

        def recursive_relu_apply(module_top):
            for idx, module in module_top._modules.items():
                recursive_relu_apply(module)
                if module.__class__.__name__ == 'ReLU':
                    module_top._modules[idx] = GuidedBackpropReLU.apply

        # replace ReLU with GuidedBackpropReLU
        recursive_relu_apply(self.model)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        # if self.cuda:
        #     output = self.forward(input.cuda())
        # else:
        #     output = self.forward(input)
        # print("这里多余requires_grad=True")
        if self.cuda:
            input = input.cuda()

        input = input.requires_grad_(True)

        # print("输入的梯度项为requires_grad=True")
        x = input.squeeze(0)
        # print(x.shape)
        target_index = None
        x, target_index = self.model.transform.resize(x, target_index)
        # print(x.shape)
        x = x.unsqueeze(0)
        # print("这里没梯度")
        x.requires_grad_(True)
        # print(x)
        x = self.model.backbone.body.conv1(x)
        # print(x.shape)
        x = self.model.backbone.body.bn1(x)
        # print(x.shape)
        x = self.model.backbone.body.relu(x)
        # print(x)
        x = self.model.backbone.body.maxpool(x)
        # print("这里没有梯度项")
        x = self.model.backbone.body.layer1(x)
        x1 = x

        x = self.model.backbone.body.layer2(x)
        x2 = x

        x = self.model.backbone.body.layer3(x)
        x3 = x

        x = self.model.backbone.body.layer4(x)
        x4 = x

        # x1 = nn.Conv2d(256, 256, 1)(x1)
        # # print(x1.shape)
        # x1 = x1 + F.interpolate(x1, size=[200,200], mode="nearest")
        # # print(x1.shape)
        # x1 = nn.Conv2d(256, 256, 3, padding=1)(x1)
        # # print(x1.shape)
        # x2 = nn.Conv2d(512, 256, 1)(x2)
        # # print(x2.shape)
        # x_2 = F.interpolate(x1, size=[100, 100], mode="nearest")
        # # x_2 = F.interpolate(x1, size=[200, 200], mode="nearest")
        # # print(x_2.shape)
        # x2 = x2 + x_2
        # # print(x2.shape)
        # x2 = F.interpolate(x1, size=[200, 200], mode="nearest")
        # x2 = nn.Conv2d(256, 256, 3, padding=1)(x2)
        # # print(x2.shape)
        # x3 = nn.Conv2d(1024, 256, 1)(x3)
        # x3 = x3 + F.interpolate(x1, size=[50,50], mode="nearest")
        # x3 = F.interpolate(x1, size=[200, 200], mode="nearest")
        # x3 = nn.Conv2d(256, 256, 3, padding=1)(x3)
        # x4 = nn.Conv2d(2048, 256, 1)(x4)
        # x4 = x4 + F.interpolate(x1, size=[25,25], mode="nearest")
        # x4 =  F.interpolate(x1, size=[200, 200], mode="nearest")
        # x4 = nn.Conv2d(256, 256, 3, padding=1)(x4)

        # x = x4 + x3 + x2 + x1
        x = x1
        # x = self.avgpool(x)
        x = self.maxpool(x)
        output = x.view(x.size(0), -1)

        # output = self.forward(input)
        #
        # print(output.shape)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())
            print(index)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        # self.model.features.zero_grad()
        # self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)

        output = input.grad.cpu().data.numpy()
        output = output[0, :, :, :]

        return output


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='./test.png',
                        help='Input image path')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args


def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img * 255)


def create_model(num_classes):
    backbone = resnet50_backbone()
    model = FasterRCNN(backbone=backbone, num_classes=num_classes)
    return model


if __name__ == '__main__':
    """ python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for VGG19 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """

    args = get_args()

    # Can work with any model, but it assumes that the model has a
    # feature method, and a classifier method,
    # as in the VGG models in torchvision.
    # model = models.resnet50(pretrained=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = create_model(num_classes=2)
    # print(model)
    train_weights = "./save_weights/resNet-model-9.pth"
    # # assert os.path.exists(train_weights), "{} file dose not exist.".format(train_weights)
    model.load_state_dict(torch.load(train_weights, map_location=device)["model"], False)
    model.to(device)
    # model.eval()
    # print(model)
    # with torch.no_grad():
        # print(model.backbone.body.layer4)
        # grad_cam = GradCam(model=model, feature_module=model.layer4, \
        #                    target_layer_names=["2"], use_cuda=args.use_cuda)
    grad_cam = GradCam(model=model, feature_module=model.backbone.body.layer4, \
                        target_layer_names=["2"], use_cuda=args.use_cuda)

    img = cv2.imread(args.image_path, 1)
    # print(img)
    img = np.float32(cv2.resize(img, (224, 224))) / 255
    # print("处理前")
    # print(img)
    input = preprocess_image(img)
    # print("处理后输入图片")
    # print(input)

        # If None, returns the map for the highest scoring category.
        # Otherwise, targets the requested index.
    target_index = None
    mask = grad_cam(input, target_index)
        # mask = grad_cam(input2, target_index)

    show_cam_on_image(img, mask)

    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
    # print(model._modules.items())
    gb = gb_model(input, index=target_index)
    gb = gb.transpose((1, 2, 0))
    cam_mask = cv2.merge([mask, mask, mask])
    cam_gb = deprocess_image(cam_mask * gb)
    gb = deprocess_image(gb)

    cv2.imwrite('gb_test_res1.jpg', gb)
    cv2.imwrite('cam_gb_test_res1.jpg', cam_gb)
