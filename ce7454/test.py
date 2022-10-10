import torch

from model import CLIP_classifier, get_customCLIP
from dataset import PSGClsDataset
from evaluator import Evaluator
from torch.utils.data import DataLoader


test_dataset = PSGClsDataset(stage='test')
test_dataloader = DataLoader(test_dataset,
                             batch_size=32,
                             shuffle=False,
                             num_workers=1)


model = get_customCLIP(test_dataset.relations)
model.load_state_dict(torch.load('/data/Projects/OpenPSG/ce7454/checkpoints/coop_pos_weight_lr1e-2_bs32_epoch15_e15_lr0.01_bs32_m0.9_wd0.0005_best.ckpt'))

model.cuda()
evaluator = Evaluator(model, k=3)

val_metrics = evaluator.eval_recall(test_dataloader) # [55]

best_val_recall = 100.0 * val_metrics['mean_recall']
print(best_val_recall)

result = evaluator.submit(test_dataloader)

# save into the file
with open(f'results/test_{best_val_recall}.txt', 'w') as writer:
    for label_list in result:
        a = [str(x) for x in label_list]
        save_str = ' '.join(a)
        writer.writelines(save_str + '\n')
