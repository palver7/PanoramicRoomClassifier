import json
from PIL import Image

import torch
from torchvision import transforms

from efficientnet_pytorch import EfficientNet

model_name = 'efficientnet-b0'
# conv type must be between 'Equi' for EquiConvs and 'Std' for StdConvs
conv_type = 'Equi'
image_size = EfficientNet.get_image_size(model_name) # 224

# Open image
img = Image.open('examples/simple/img2.jpg')
#img

# Preprocess image
tfms = transforms.Compose([transforms.Resize(image_size), transforms.CenterCrop(image_size), 
                           transforms.ToTensor(),
                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
img = tfms(img).unsqueeze(0)

# Load class names
labels_map = json.load(open('examples/simple/labels_map.txt'))
labels_map = [labels_map[str(i)] for i in range(1000)]

#Classify with EfficientNet
model = EfficientNet.from_pretrained(model_name,conv_type)
model.eval()
with torch.no_grad():
    logits = model(img)
preds = torch.topk(logits, k=5).indices.squeeze(0).tolist()

print('-----')
for idx in preds:
    label = labels_map[idx]
    prob = torch.softmax(logits, dim=1)[0, idx].item()
    print('{:<75} ({:.2f}%)'.format(label, prob*100))

