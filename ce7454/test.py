import torch

from model import CLIP_classifier
from dataset import PSGClsDataset
from evaluator import Evaluator
from torch.utils.data import DataLoader


model = CLIP_classifier()
model.load_state_dict(torch.load('/data/Projects/OpenPSG/ce7454/checkpoints/clip_classifier_e60_lr0.001_bs8_m0.9_wd0.0005_best.ckpt'))

model.cuda()
evaluator = Evaluator(model, k=3)

dataset = PSGClsDataset(stage='val')
dataloader = DataLoader(dataset,
                        batch_size=32,
                        shuffle=False,
                        num_workers=2)

val_metrics = evaluator.eval_recall(dataloader) # [55]

print(100.0 * val_metrics['mean_recall'])
