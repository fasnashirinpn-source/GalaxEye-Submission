import torch
from Model import SiameseUNet

model = SiameseUNet()
model.load_state_dict(torch.load('siamese_unet.pth'))
model.eval()
print('Model ready for inference')