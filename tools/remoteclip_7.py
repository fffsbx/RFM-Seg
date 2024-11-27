import numpy as np
import torch
import open_clip
import torch.nn as nn
from torchvision.transforms import ToPILImage
from torch.nn import functional as F

model_name = 'RN50'  # 'RN50' or 'ViT-B-32' or 'ViT-L-14'
model, _, preprocess = open_clip.create_model_and_transforms(model_name)
model = model.cuda().eval()
path_to_your_checkpoints = 'checkpoints/models--chendelong--RemoteCLIP'
ckpt = torch.load(f"{path_to_your_checkpoints}/RemoteCLIP-{model_name}.pt", map_location="cuda")
model.load_state_dict(ckpt)
class remoteclip_image(nn.Module):
    def __init__(self):
        super(remoteclip_image, self).__init__()

    def forward(self, image):

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = model.visual.conv1(image.cuda())
            image_features = model.visual.bn1(image_features)
            image_features = model.visual.act1(image_features)
            image_features = model.visual.conv2(image_features)
            image_features = model.visual.bn2(image_features)
            image_features = model.visual.act2(image_features)
            image_features = model.visual.conv3(image_features)
            image_features = model.visual.bn3(image_features)
            image_features = model.visual.act3(image_features)
            image_features = model.visual.avgpool(image_features)
            image_features1 = model.visual.layer1(image_features)

            image_features2_0 = model.visual.layer2[0](image_features1)
            image_features2_1 = model.visual.layer2[1](image_features2_0)
            image_features2_2 = model.visual.layer2[2](image_features2_1)
            image_features2_3_conv1 = model.visual.layer2[3].conv1(image_features2_2)
            image_features2_3_bn1 = model.visual.layer2[3].bn1(image_features2_3_conv1)
            image_features2_3_act1 = model.visual.layer2[3].act1(image_features2_3_bn1)
            image_features2_3_conv2 = model.visual.layer2[3].conv2(image_features2_3_act1)
            image_features2_3_bn2 = model.visual.layer2[3].bn2(image_features2_3_conv2)
            image_features2_3_act2 = model.visual.layer2[3].act2(image_features2_3_bn2)
            image_features2_3_avgpool = model.visual.layer2[3].avgpool(image_features2_3_act2)
            image_features2_3_conv3 = model.visual.layer2[3].conv3(image_features2_3_avgpool)
            image_features2_3_bn3 = model.visual.layer2[3].bn3(image_features2_3_conv3)
            image_features2_3_act3 = model.visual.layer2[3].act3(image_features2_3_bn3)

            image_features3_0 = model.visual.layer3[0](image_features2_3_act3)
            image_features3_1 = model.visual.layer3[1](image_features3_0)
            image_features3_2 = model.visual.layer3[2](image_features3_1)
            image_features3_3 = model.visual.layer3[3](image_features3_2)
            image_features3_4 = model.visual.layer3[4](image_features3_3)
            image_features3_5_conv1 = model.visual.layer3[5].conv1(image_features3_4)
            image_features3_5_bn1 = model.visual.layer3[5].bn1(image_features3_5_conv1)
            image_features3_5_act1 = model.visual.layer3[5].act1(image_features3_5_bn1)
            image_features3_5_conv2 = model.visual.layer3[5].conv2(image_features3_5_act1)
            image_features3_5_bn2 = model.visual.layer3[5].bn2(image_features3_5_conv2)
            image_features3_5_act2 = model.visual.layer3[5].act2(image_features3_5_bn2)
            image_features3_5_avgpool = model.visual.layer3[5].avgpool(image_features3_5_act2)
            image_features3_5_conv3 = model.visual.layer3[5].conv3(image_features3_5_avgpool)
            image_features3_5_bn3 = model.visual.layer3[5].bn3(image_features3_5_conv3)
            image_features3_5_act3 = model.visual.layer3[5].act3(image_features3_5_bn3)

            image_features4_0 = model.visual.layer4[0](image_features3_5_act3)
            image_features4_1 = model.visual.layer4[1](image_features4_0)
            image_features4_2_conv1 = model.visual.layer4[2].conv1(image_features4_1)
            image_features4_2_bn1 = model.visual.layer4[2].bn1(image_features4_2_conv1)
            image_features4_2_act1 = model.visual.layer4[2].act1(image_features4_2_bn1)
            image_features4_2_conv2 = model.visual.layer4[2].conv2(image_features4_2_act1)
            image_features4_2_bn2 = model.visual.layer4[2].bn2(image_features4_2_conv2)
            image_features4_2_act2 = model.visual.layer4[2].act2(image_features4_2_bn2)
            image_features4_2_avgpool = model.visual.layer4[2].avgpool(image_features4_2_act2)

        return image_features,image_features2_3_avgpool,image_features3_5_avgpool,image_features4_2_avgpool


class remoteclip_text(nn.Module):
    def __init__(self):
        super(remoteclip_text, self).__init__()

    def forward(self, images,text_features):
        print('images',images.shape)
        for t in range(images.size(0)):
            with torch.no_grad(), torch.cuda.amp.autocast():
                single_image = images[t, :, :, :]
                single_image = single_image.unsqueeze(0)
                image_features = model.visual.conv1(single_image.cuda())
                image_features = model.visual.bn1(image_features)
                image_features = model.visual.act1(image_features)
                image_features = model.visual.conv2(image_features)
                image_features = model.visual.bn2(image_features)
                image_features = model.visual.act2(image_features)
                image_features = model.visual.conv3(image_features)
                image_features = model.visual.bn3(image_features)
                image_features = model.visual.act3(image_features)
                image_features = model.visual.avgpool(image_features)
                image_features = model.visual.layer1(image_features)
                image_features = model.visual.layer2(image_features)
                image_features = model.visual.layer3(image_features)
                image_features = model.visual.layer4(image_features)
                block_size = 7
                stride = 1
                result = torch.zeros(images.size(0), 512, image_features.shape[2],image_features.shape[3])
                result_features = torch.full((6,512, image_features.shape[2],image_features.shape[3]),0)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                for i in range(0, image_features.shape[2] - block_size + 1, stride):
                    for j in range(0, image_features.shape[3] - block_size + 1, stride):
                        # 提取子张量并添加到结果列表中
                        sub_tensor = image_features[:, :, i:i + 7, j:j + 7]
                        image_features_patch = model.visual.attnpool(sub_tensor)
                        image_features_patch = F.normalize(image_features_patch, dim=-1)
                        image_features_patch /= image_features_patch.norm(dim=-1, keepdim=True)
                        text_probs = (100.0 * image_features_patch.float() @ text_features.T.float()).softmax(dim=-1).detach().cpu().numpy()[0]
                        max_index = np.argmax(text_probs)
                        if max_index in (0, 1, 2, 3,4,5,6,7):
                            max_value = 0
                        elif max_index in (8,9,10,11,12,13,14):
                            max_value = 1
                        elif max_index in (15,16,17,18,19,20,21):
                            max_value = 2
                        elif max_index in (22,23,24,25,26,27,28):
                            max_value = 3
                        elif max_index in (29,30,31,32,33,34,35):
                            max_value = 4
                        else:
                            max_value = 5
                        # if max_index in (0, 1, 2, 3,4):
                        #     max_value = 0
                        # elif max_index in (5,6,7,8,9,10,11,12):
                        #     max_value = 1
                        # else:
                        #     max_value = 2
                        result_features[max_value, :, i:i + 7, j:j + 7] += 1
                max_values, max_indices = torch.max(result_features, dim=0)
                result[t,:,:,:] = max_indices
                result = F.normalize(result,p=2,dim=1)





        print('result:',result.shape)

        return result





