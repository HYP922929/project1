"""
该脚本用于调用训练好的模型权重去计算验证集/测试集的COCO指标
以及每个类别的mAP(IoU=0.5)
"""

import os
import json

import torch
from tqdm import tqdm
import numpy as np

import transforms
from network_files.faster_rcnn_framework import FasterRCNN
from backbone.resnet50_fpn_model import resnet50_fpn_backbone
from my_dataset import VOC
from train_utils.coco_utils import get_coco_api_from_dataset
from train_utils.coco_eval import CocoEvaluator


def summarize(self, catId=None):
    """
    Compute and display summary metrics for evaluation results.
    Note this functin can *only* be applied on the default parameter setting
    """

    def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
        p = self.params
        iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
        titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
        typeStr = '(AP)' if ap == 1 else '(AR)'
        iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
            if iouThr is None else '{:0.2f}'.format(iouThr)

        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

        if ap == 1:
            # dimension of precision: [TxRxKxAxM]
            s = self.eval['precision']
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                # 输出满足条件的坐标 【0】x坐标
                s = s[t]

            if isinstance(catId, int):
                s = s[:, :, catId, aind, mind]
            else:
                s = s[:, :, :, aind, mind]

        else:
            # dimension of recall: [TxKxAxM]
            s = self.eval['recall']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]

            if isinstance(catId, int):
                s = s[:, catId, aind, mind]
            else:
                s = s[:, :, aind, mind]

        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])

        print_string = iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s)
        return mean_s, print_string

    stats, print_list = [0] * 12, [""] * 12
    stats[0], print_list[0] = _summarize(1)
    stats[1], print_list[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[0])
    stats[2], print_list[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[0])
    stats[3], print_list[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[0])
    stats[4], print_list[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[0])
    stats[5], print_list[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[0])
    stats[6], print_list[6] = _summarize(0, maxDets=self.params.maxDets[0])
    stats[7], print_list[7] = _summarize(0, maxDets=self.params.maxDets[0])
    stats[8], print_list[8] = _summarize(0, maxDets=self.params.maxDets[0])
    stats[9], print_list[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[0])
    stats[10], print_list[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[0])
    stats[11], print_list[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[0])

    print_info = "\n".join(print_list)

    if not self.eval:
        raise Exception('Please run accumulate() first')

    return stats, print_info


def main(parser_data):
    device = torch.device(parser_data.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))

    data_transform = {
        "val": transforms.Compose([transforms.ToTensor()])
    }

    # read class_indict
    label_json_path = './pascal_voc_classes.json'
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    json_file = open(label_json_path, 'r')
    class_dict = json.load(json_file)
    category_index = {v: k for k, v in class_dict.items()}

    VOC_root = parser_data.data_path
    # check voc root
    if os.path.exists(os.path.join(VOC_root, "VOC")) is False:
        raise FileNotFoundError("VOC dose not in path:'{}'.".format(VOC_root))

    # 注意这里的collate_fn是自定义的，因为读取的数据包括image和targets，不能直接使用默认的方法合成batch
    batch_size = parser_data.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using %g dataloader workers' % nw)

    # load validation data set
    val_data_set = VOC(VOC_root, data_transform["val"], "val.txt")
    val_data_set_loader = torch.utils.data.DataLoader(val_data_set,
                                                      batch_size=batch_size,
                                                      shuffle=False,
                                                      num_workers=nw,
                                                      collate_fn=val_data_set.collate_fn)

    # create model num_classes equal background + 20 classes
    backbone = resnet50_fpn_backbone()
    model = FasterRCNN(backbone=backbone, num_classes=parser_data.num_classes + 1)
    # model = FasterRCNN(backbone=backbone, num_classes=parser_data.num_classes)

    # 载入你自己训练好的模型权重
    weights_path = parser_data.weights
    assert os.path.exists(weights_path), "not found {} file.".format(weights_path)
    weights_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(weights_dict['model'])
    # print(model)

    model.to(device)

    # evaluate on the test dataset
    coco = get_coco_api_from_dataset(val_data_set)
    #print(dir(coco))
    iou_types = ["bbox"]
    coco_evaluator = CocoEvaluator(coco, iou_types)
    cpu_device = torch.device("cpu")

    model.eval()
    with torch.no_grad():
        for image, targets in tqdm(val_data_set_loader, desc="validation..."):
            # 将图片传入指定设备device tqdm 进度条
            image = list(img.to(device) for img in image)

            # inference
            outputs = model(image)
            # print(outputs)

            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
            # print(outputs)
            res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
            # print(res)
            coco_evaluator.update(res)
            # 输入到update就是四个以上的bbox

    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    coco_eval = coco_evaluator.coco_eval["bbox"]
    # calculate COCO info for all classes
    coco_stats, print_coco = summarize(coco_eval)

    # calculate voc info for every classes(IoU=0.5)
    voc_map_info_list = []
    for i in range(len(category_index)):
        stats, _ = summarize(coco_eval, catId=i)
        voc_map_info_list.append(" {:15}: {}".format(category_index[i + 1], stats[1]))

    print_voc = "\n".join(voc_map_info_list)
    # print(print_voc)

    # 将验证结果保存至txt文件中
    with open("record_mAP.txt", "w") as f:
        record_lines = ["COCO results:",
                        print_coco,
                        "",
                        "mAP(IoU=0.5) for each category:",
                        print_voc]
        f.write("\n".join(record_lines))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    # 使用设备类型
    parser.add_argument('--device', default='cuda', help='device')

    # 检测目标类别数
    parser.add_argument('--num-classes', type=int, default='1', help='number of classes')

    # 数据集的根目录(VOC2012根目录)
    parser.add_argument('--data-path', default='./', help='dataset root')

    # 训练好的权重文件
    parser.add_argument('--weights', default='./save_weights/resNetFpn-model-9.pth', type=str, help='training weights')

    # batch size
    parser.add_argument('--batch_size', default=4, type=int, metavar='N',
                        help='batch size when training.')

    args = parser.parse_args()

    main(args)
