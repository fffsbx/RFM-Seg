import numpy as np
import torch
import open_clip
import torch.nn as nn
from torchvision.transforms import ToPILImage

model_name = 'RN50'  # 'RN50' or 'ViT-B-32' or 'ViT-L-14'
model, _, preprocess = open_clip.create_model_and_transforms(model_name)
model = model.cuda().eval()
path_to_your_checkpoints = '/media/ubuntu/762cdba6-78ae-4daa-9dfc-d3db8b704a24/fffsbx_projects/GeoSeg/checkpoints/models--chendelong--RemoteCLIP'
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
        max_values = []
        for i in range(images.size(0)):
            single_image = images[i,:,:,:]
            to_pil_image = ToPILImage()
            pil_image = to_pil_image(single_image)

            image = preprocess(pil_image).unsqueeze(0)

            with torch.no_grad(), torch.cuda.amp.autocast():
                image_features = model.encode_image(image.cuda())
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)

                text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1).cpu().numpy()[0]
                max_index = np.argmax(text_probs)
                if max_index in (0, 1, 2, 3):
                    max = 0
                elif max_index in (4, 5, 6, 7, 8):
                    max = 0.1
                elif max_index in (9, 10, 11, 12):
                    max = 0.2
                elif max_index in (13, 14, 15, 16):
                    max = 0.3
                elif max_index in (17, 18, 19, 20):
                    max = 0.4
                else:
                    max = 0.5
            max_values.append(max)
        max_values_tensor = torch.tensor(max_values).view(16, 1)




        return max_values_tensor





