import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet34,resnet50
from positional_encodings.torch_encodings import PositionalEncodingPermute2D

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
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.input_proj = nn.Conv2d(input_dim, hidden_dim, kernel_size=1)
        self.transformer = torch.nn.Transformer(d_model=hidden_dim, 
                                                nhead=8, 
                                                num_encoder_layers=6,
                                                num_decoder_layers=6,
                                                dim_feedforward=2048,
                                                dropout=0.1
                                                )
        self.pos_enc = PositionalEncodingPermute2D(input_dim)
    
    def forward(self, img):
        bs = img.size(0)
        feature = self.pos_enc(self.input_proj(self.backbone(img)))             # [N,C,H,W]
        feature = feature.flatten(2).permute(2, 0, 1)                           # [HW,N,C] 
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)     # [num_queries,N,C]
        hs = self.transformer(feature, query_embed)                             # [num_queries,N,C]

        outputs_class = self.class_embed(hs)                                    # [num_queries,N,num_classes]
        outputs_class = torch.sum(outputs_class, dim=0)                         # [N,num_classes]

        return outputs_class                                                    # [N,num_queries]





