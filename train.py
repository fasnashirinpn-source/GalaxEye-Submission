import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import SiameseUNet
from dataset import ChangeDetectionDataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = ChangeDetectionDataset(
    'dataset/pre_eo',
    'dataset/pre_sar',
    'dataset/post_eo',
    'dataset/post_sar',
    'dataset/masks'
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

model = SiameseUNet().to(DEVICE)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(50):
    model.train()
    total_loss = 0

    for pre, post, mask in train_loader:
        pre, post, mask = pre.to(DEVICE), post.to(DEVICE), mask.to(DEVICE)

        optimizer.zero_grad()
        output = model(pre, post)
        loss = criterion(output, mask)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}')

torch.save(model.state_dict(), 'siamese_unet.pth')
print('Training complete')
