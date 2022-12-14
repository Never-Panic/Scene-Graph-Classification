import random
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torchvision.models import resnet34,resnet50
from positional_encodings.torch_encodings import PositionalEncodingPermute2D

import clip

from cocoop import CustomCLIP


class BackBone(nn.Module):
    def __init__(self) -> None:
        super().__init__() 
        net = resnet50(pretrained=True)
        return_layers = {'layer4': '0'}
        self.get_feature_map = torchvision.models._utils.IntermediateLayerGetter(net,return_layers)
    
    def forward(self, x):
        return self.get_feature_map(x)['0']


class DETR(nn.Module):
    def __init__(self, backbone, input_dim=2048, hidden_dim=512, num_classes=56, num_queries=3) -> None:
        super().__init__()

        self.backbone = backbone
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        # self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.input_proj = nn.Conv2d(input_dim, hidden_dim, kernel_size=1)
        # self.transformer = torch.nn.Transformer(d_model=hidden_dim, 
        #                                         nhead=8, 
        #                                         num_encoder_layers=6,
        #                                         num_decoder_layers=6,
        #                                         dim_feedforward=2048,
        #                                         dropout=0.1
        #                                         )
        self.transformerEncoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=hidden_dim, 
                nhead=8,
            ),
            num_layers=6
        )
        self.pos_enc = PositionalEncodingPermute2D(input_dim)
    
    def forward(self, img):
        bs = img.size(0)
        feature = self.pos_enc(self.input_proj(self.backbone(img)))             # [N,C,H,W]
        feature = feature.flatten(2).permute(2, 0, 1)                           # [HW,N,C] 
        # query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)     # [num_queries,N,C]
        # hs = self.transformer(feature, query_embed)                             # [num_queries,N,C]
        transformed_feature = self.transformerEncoder(feature)                  # [HW,N,C]

        outputs_class = self.class_embed(transformed_feature)                   # [HW,N,num_classes]
        outputs_class = torch.mean(outputs_class, dim=0)                        # [N,num_classes]

        return outputs_class                                                    # [N,num_queries]


class CLIP_loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        clip_model, clip_preprocess = clip.load("ViT-B/16", device='cuda', jit=False)
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False
        
        self.clip_model = clip_model
        self.preprocess = T.Compose([
            T.Resize((224, 224)),
            T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def forward(self, imgs, texts):
        '''
        Args:
            imgs: [N, C, H, W]
            text: [N] string
        '''
        texts = clip.tokenize(texts).cuda()
        imgs = self.preprocess(imgs)

        text_z = self.clip_model.encode_text(texts)
        text_z = text_z / text_z.norm(dim=-1, keepdim=True)

        image_z = self.clip_model.encode_image(imgs)
        image_z = image_z / image_z.norm(dim=-1, keepdim=True)
        
        loss_clip = - (image_z * text_z).sum(-1).mean()

        return loss_clip


class CLIP_classifier(nn.Module):
    def __init__(self, hidden_dim=1024) -> None:
        super().__init__()
        clip_model, clip_preprocess = clip.load("ViT-L/14@336px", device='cuda', jit=False)

        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False
        
        self.clip_model = clip_model
        self.preprocess = T.Compose([
            T.Resize((336, 336)),
            T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        drop_out_p = 0.2
        self.ffn = torch.nn.Sequential(
            torch.nn.Dropout(p=drop_out_p),
            torch.nn.Linear(768, hidden_dim),
            nn.ReLU(),
            torch.nn.Dropout(p=drop_out_p),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            torch.nn.Dropout(p=drop_out_p),
            nn.Linear(hidden_dim, 56),
        )

        self.augmenter = T.AugMix()


    def augment(self, imgs):
        '''
        Args:
            imgs: tensor in [0, 1] float32

        Return:
            imgs: tensor in [-1, 1] float32
        '''
        imgs = (imgs * 255).type(torch.uint8)
        imgs = self.augmenter(imgs).type(torch.float32) / 255.
        imgs = T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(imgs)

        return imgs


    def forward(self, imgs):
        '''
        If training, imgs are [0, 1] float32
        Else, imgs are [-1, 1] float32
        '''
        if self.training:
            if random.random()>0.4:
                imgs = self.augment(imgs)
            else:
                imgs = T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(imgs)

        # need [-1,1] input
        imgs = self.preprocess(imgs)
        image_z = self.clip_model.encode_image(imgs).float()        # [N, 768]

        return self.ffn(image_z)                            # [N, 56]
        

def get_customCLIP(relations):
    clip_model, clip_preprocess = clip.load("ViT-L/14@336px", device='cpu', jit=False)
    return CustomCLIP(relations, clip_model)