import json
modes=['train','test','val']
per_cat_count={}
for mode in modes:
    fpath='../doclaynet_data/{}_c6.json'.format(mode)
    print('load {}'.format(mode))
    data=json.load(open(fpath))
    print('loaded')
    print(data['categories'])
    anns=data['annotations']
    for ann in anns:
        cat=ann['category_id']
        if cat in per_cat_count:
            per_cat_count[cat]+=1
        else:
            per_cat_count[cat]=1
    cats=sorted(list(per_cat_count.keys()))
    for cat in cats:
        print('{}\t{}'.format(cat,per_cat_count[cat]))
