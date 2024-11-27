from torch.utils.data import DataLoader
from geoseg.losses import *
from geoseg.datasets.aerial_dataset import *
from geoseg.models.A2FPN import A2FPN
from catalyst.contrib.nn import Lookahead
from catalyst import utils

# training hparam
max_epoch = 105
ignore_index = len(CLASSES)
train_batch_size = 16
val_batch_size = 8
lr = 6e-4
weight_decay = 0.01
backbone_lr = 6e-5
backbone_weight_decay = 0.01
num_classes = len(CLASSES)
classes = CLASSES

weights_name = "a2fpn-r18-1024-768crop-e105_V_fusetwo_L_fusethree"
weights_path = "model_weights/aerial/{}".format(weights_name)
test_weights_name = "a2fpn-r18-1024-768crop-e105_V_fusetwo_L_fusethree_16_8_6e-4_42"
log_name = 'aerial/{}'.format(weights_name)
monitor = 'val_F1'
monitor_mode = 'max'
save_top_k = 1
save_last = True
check_val_every_n_epoch = 1
pretrained_ckpt_path = None # the path for the pretrained model weight
gpus = 'auto'  # default or gpu ids:[0] or gpu nums: 2, more setting can refer to pytorch_lightning
resume_ckpt_path = None  # whether continue training with the checkpoint, default None

#  define the network
net = A2FPN(3,num_classes)
# define the loss
loss = UnetFormerLoss(ignore_index=ignore_index)
use_aux_loss = False
# define the dataloader
train_dataset = AerialDataset(data_root='data/aerial/train', mode='train',
                                 mosaic_ratio=0.25, transform=train_aug)
val_dataset = AerialDataset(transform=val_aug)
test_dataset = AerialDataset(data_root='data/aerial/test',
                                transform=val_aug)
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=4,
                          pin_memory=True,
                          shuffle=True,
                          drop_last=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=4,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False)
# define the optimizer
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = utils.process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)

