import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from tools.cfg import py2cfg
import os
import torch
from torch import nn
import cv2
import numpy as np
import argparse
from pathlib import Path
from tools.metric import Evaluator
from pytorch_lightning.loggers import CSVLogger
import random

from pynvml import *
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
nvmlInit()
handle = nvmlDeviceGetHandleByIndex(int(os.environ["CUDA_VISIBLE_DEVICES"]))
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, help="Path to the config.", required=True)
    return parser.parse_args()

class Supervision_Train(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.net = config.net
        self.loss = config.loss
        #创建一个名为 metrics_train 的评估器对象，用于训练阶段的指标评估。num_class 参数来自配置对象，用于指定类别数量。
        self.metrics_train = Evaluator(num_class=config.num_classes)
        #创建一个名为 metrics_val 的评估器对象，用于验证阶段的指标评估。同样，num_class 参数来自配置对象。
        self.metrics_val = Evaluator(num_class=config.num_classes)
    def forward(self, x):
        #调用网络模型 net，对输入 x 进行前向传播，得到预测结果 seg_pre。
        seg_pre = self.net(x)
        return seg_pre
    def training_step(self, batch, batch_idx):#定义了训练阶段的操作，接受一个批次的数据 batch 和批次索引 batch_idx。
        img, mask = batch['img'], batch['gt_semantic_seg']#从批次数据中提取图像 img 和语义分割标签 mask。
        prediction = self.net(img)#利用网络模型 net 对图像进行预测，得到预测结果 prediction。
        loss = self.loss(prediction, mask)#计算预测结果 prediction 和真实标签 mask 之间的损失值，使用配置中指定的损失函数 loss。
        if self.config.use_aux_loss:#检查配置中是否启用辅助损失。
            pre_mask = nn.Softmax(dim=1)(prediction[0])
        else:
            pre_mask = nn.Softmax(dim=1)(prediction)
        pre_mask = pre_mask.argmax(dim=1)#取预测结果中概率最高的类别作为预测的分割结果。
        for i in range(mask.shape[0]):#遍历批次中的每个样本。
            #将真实标签和预测结果添加到训练阶段的评估器中，用于后续指标评估。
            self.metrics_train.add_batch(mask[i].cpu().numpy(), pre_mask[i].cpu().numpy())
        return {"loss": loss}
    def on_train_epoch_end(self):#定义了一个在每个训练epoch结束时执行的函数。
        if 'vaihingen' in self.config.log_name:#如果日志名称中包含“vaihingen”，则执行以下代码块。
            #计算Intersection over Union的平均值，但是将最后一个值（假设是一个阈值）排除在外，然后赋值给mIoU。
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            #计算F1分数的平均值，同样将最后一个值排除在外，然后赋值给F1。
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        elif 'potsdam' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        elif 'whubuilding' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        elif 'massbuilding' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        elif 'cropland' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        else:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union())
            F1 = np.nanmean(self.metrics_train.F1())
        #计算Overall Accuracy的平均值，并赋值给OA。
        OA = np.nanmean(self.metrics_train.OA())
        #获取每个类别的Intersection over Union的值。
        iou_per_class = self.metrics_train.Intersection_over_Union()
        #创建一个包含mIoU、F1和OA的字典。
        eval_value = {'mIoU': mIoU,'F1': F1,'OA': OA}
        print('train_images:', eval_value)
        iou_value = {}#创建一个空字典用于存储每个类别的Intersection over Union。
        for class_name, iou in zip(self.config.classes, iou_per_class):#使用zip函数将类别名称和对应的Intersection over Union值进行迭代。
            iou_value[class_name] = iou
        print(iou_value)
        self.metrics_train.reset()#重置训练指标，准备下一个epoch的计算。
        log_dict = {'train_mIoU': mIoU, 'train_F1': F1, 'train_OA': OA}
        self.log_dict(log_dict, prog_bar=True)
    def validation_step(self, batch, batch_idx):
        img, mask = batch['img'], batch['gt_semantic_seg']#从批次中获取图像和语义分割标签。
        prediction = self.forward(img)#使用模型的forward方法对图像进行预测，得到预测的语义分割结果。
        pre_mask = nn.Softmax(dim=1)(prediction)#对预测结果进行softmax归一化，dim=1表示按通道维度进行softmax操作。
        pre_mask = pre_mask.argmax(dim=1)#获取每个像素点预测的类别，即取softmax后概率最高的类别作为预测结果。
        for i in range(mask.shape[0]):#对于每个样本，在标签的第一个维度上进行迭代，即对批次中的每张图像进行处理。
            #将每张图像的真实标签和预测标签传递给验证指标对象，通常用于计算评估指标如IoU（Intersection over Union）等。
            self.metrics_val.add_batch(mask[i].cpu().numpy(), pre_mask[i].cpu().numpy())
        loss_val = self.loss(prediction, mask)#计算预测结果与真实标签之间的损失，通常使用交叉熵损失函数。
        return {"loss_val": loss_val}#返回一个字典，其中包含验证步骤的损失值，用于后续的验证过程中的汇总和记录
    def on_validation_epoch_end(self):#定义了一个在验证周期结束时执行的函数。
        if 'vaihingen' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])#计算平均IoU（Intersection over Union）。
            F1 = np.nanmean(self.metrics_val.F1()[:-1])#计算F1分数的平均值，
        elif 'potsdam' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        elif 'whubuilding' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        elif 'massbuilding' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        elif 'cropland' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        else:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union())
            F1 = np.nanmean(self.metrics_val.F1())
        OA = np.nanmean(self.metrics_val.OA())#计算整体精度（Overall Accuracy）的平均值。
        iou_per_class = self.metrics_val.Intersection_over_Union()#获取每个类别的IoU值。
        eval_value = {'mIoU': mIoU,'F1': F1,'OA': OA}#将计算得到的mIoU、F1和OA值存储在一个字典中。
        print('val:', eval_value)
        iou_value = {}
        for class_name, iou in zip(self.config.classes, iou_per_class):
            iou_value[class_name] = iou
        print(iou_value)
        self.metrics_val.reset()
        log_dict = {'val_mIoU': mIoU, 'val_F1': F1, 'val_OA': OA}
        self.log_dict(log_dict, prog_bar=True)
    def configure_optimizers(self):
        optimizer = self.config.optimizer
        lr_scheduler = self.config.lr_scheduler
        return [optimizer], [lr_scheduler]
    def train_dataloader(self):
        return self.config.train_loader
    def val_dataloader(self):
        return self.config.val_loader

# training
def main():
    args = get_args()
    config = py2cfg(args.config_path)
    seed_everything(42)
    checkpoint_callback = ModelCheckpoint(save_top_k=config.save_top_k, monitor=config.monitor,
                                          save_last=config.save_last, mode=config.monitor_mode,
                                          dirpath=config.weights_path,
                                          filename=config.weights_name)
    logger = CSVLogger('lightning_logs', name=config.log_name)
    model = Supervision_Train(config)
    if config.pretrained_ckpt_path:#如果有预训练模型的检查点路径，这行代码会加载预训练模型，并将其赋值给变量 model。
        model = Supervision_Train.load_from_checkpoint(config.pretrained_ckpt_path, config=config)
    trainer = pl.Trainer(devices=config.gpus,#指定训练时使用的设备
                         max_epochs=config.max_epoch,#指定训练的最大周期数
                         accelerator='auto',#自动选择加速器，可能是 GPU 或 TPU 等。
                         check_val_every_n_epoch=config.check_val_every_n_epoch,#指定每隔多少个周期进行一次验证。
                         callbacks=[checkpoint_callback],#指定回调函数，这里传入了一个 checkpoint_callback，用于在训练过程中保存模型的检查点。
                         strategy='auto',#训练策略选项，指定 Lightning 如何进行分布式训练。
                         logger=logger)
    trainer.fit(model=model, ckpt_path=config.resume_ckpt_path)
if __name__ == "__main__":
   main()
