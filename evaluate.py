import torch
from sklearn.metrics import precision_score, recall_score, jaccard_score, f1_score
from torch.utils.data import DataLoader
from Model import SiameseUNet
from dataset import ChangeDetectionDataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = ChangeDetectionDataset(
    'dataset/pre_eo',
    'dataset/pre_sar',
    'dataset/post_eo',
    'dataset/post_sar',
    'dataset/masks'
)
loader = DataLoader(dataset, batch_size=8)

model = SiameseUNet().to(DEVICE)
model.load_state_dict(torch.load('siamese_unet.pth'))
model.eval()

y_true, y_pred = [], []

with torch.no_grad():
    for pre, post, mask in loader:
        pre, post = pre.to(DEVICE), post.to(DEVICE)
        output = model(pre, post)
        pred = (output > 0.5).float().cpu().numpy().flatten()
        gt = mask.numpy().flatten()
        y_pred.extend(pred)
        y_true.extend(gt)

print('Precision:', precision_score(y_true, y_pred))
print('Recall:', recall_score(y_true, y_pred))
print('IoU:', jaccard_score(y_true, y_pred))
print('F1:', f1_score(y_true, y_pred))
