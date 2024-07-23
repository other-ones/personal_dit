import os
import json
import numpy as np

gt_data=json.load(open('/data/twkim/doc_layout/raw/doclaynet/COCO/train_c6.json'))
print('loaded')
anns=gt_data['annotations']
target_id=np.random.choice(np.arange(200))
# 2. Ground Truth
per_img_anns={}
for item in anns[:]:
    img_id=item['image_id']
    if img_id not in per_img_anns:
        per_img_anns[img_id]=[item]
    else:
        per_img_anns[img_id].append(item)
max_num=0
counts=[]
for img_id in per_img_anns:
    img_anns=per_img_anns[img_id]
    num=len(img_anns)
    counts.append(num)
    if num>max_num:
        max_num=num
counts=sorted(counts)[::-1]
print(np.sum(np.array(counts)>=79),'count')
print(counts[:300],len(counts))
print(np.mean(counts))