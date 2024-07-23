import os
import json
import numpy as np




# /data/twkim/doc_layout/raw/DocLayNet/COCO
for mode in ['train']:
    old_data=json.load(open('/data/twkim/doc_layout/raw/DocLayNet/COCO/{}_c6.json'.format(mode)))
    dst_file=open('/data/twkim/doc_layout/raw/DocLayNet/COCO/{}_c6_sparse.json'.format(mode),'w')
    
    print('loaded')
    categories=old_data['categories']
    old_anns=old_data['annotations']
    old_imgs=old_data['images']
    new_data={'annotations':None,
              'images':None,
              'categories':categories}
    id2fname={}
    id2img={}
    root_doclaynet='/data/twkim/doc_layout/raw/DocLayNet/COCO/'
    absent_count=0
    for img in old_imgs:
        id=img['id']
        fname=img['file_name']
        fpath=os.path.join(root_doclaynet,'../PNG',fname)
        if not os.path.exists(fpath):
            absent_count+=1
            continue

        id2fname[id]=fname
        assert id not in id2img
        id2img[id]=img



    

    per_img_anns={}
    gt_items=[]
    new_imgs=[]
    new_anns=[]
    for ann in old_anns[:]:
        img_id=ann['image_id']
        if img_id not in per_img_anns:
            per_img_anns[img_id]=[ann]
        else:
            per_img_anns[img_id].append(ann)
    counts=[]
    removed_count=0
    for img_id in per_img_anns:
        img_anns=per_img_anns[img_id]
        img_data=id2img[img_id]
        num=len(img_anns)
        if num>=79:
            removed_count+=1
            continue
        new_imgs.append(img_data)
        new_anns+=img_anns  
    new_data['annotations']=new_anns
    new_data['images']=new_imgs
    print(mode,removed_count,'removed')
    print(len(old_imgs),'old_imgs',len(new_imgs),'len(new_imgs)',len(id2img),'len(id2img)',len(per_img_anns),'len(per_img_anns)')
    print(len(old_anns),'old_anns',len(new_anns),'len(new_anns)')
    print(absent_count,'absent_count')
    json.dump(new_data,dst_file,indent=1)          
    