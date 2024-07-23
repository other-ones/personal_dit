from transformers import BeitImageProcessor, BeitForMaskedImageModeling
import torch
from PIL import Image

image = Image.open('/home/twkim/project/dit/object_detection/publaynet_example.jpeg').convert('RGB')
processor = BeitImageProcessor.from_pretrained("microsoft/dit-large")
model = BeitForMaskedImageModeling.from_pretrained("microsoft/dit-large")

num_patches = (model.config.image_size // model.config.patch_size) ** 2
pixel_values = processor(images=image, return_tensors="pt").pixel_values
# create random boolean mask of shape (batch_size, num_patches)
bool_masked_pos = torch.randint(low=0, high=2, size=(1, num_patches)).bool()

outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)
loss, logits = outputs.loss, outputs.logits
print(outputs)
print(bool_masked_pos.shape,'bool_masked_pos')
print(num_patches,'num_patches')
print(image.size,'image')
print(pixel_values.shape,'pixel_values')
print(logits.shape,'logits')
print(loss,'loss')
print(model.config)

