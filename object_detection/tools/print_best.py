fpath='/home/twkim/project/dit/object_detection/outputs/maskrcnn_dit_base_c6_publaynet/log.txt'
lines=open(fpath).readlines()
best_ap=0
best_step=0
for idx,line in enumerate(lines):
    line=line.strip()
    if 'd2.evaluation.testing INFO: copypaste: Task: segm' in line:
        apline=lines[idx+2]
        apline_splits=apline.split('copypaste: ')
        ap_item=apline_splits[1]
        ap_splits=ap_item.split(',')
        ap=ap_splits[0]
        ap=float(ap)

        stepline=lines[idx+3]
        stepline_splits=stepline.split('iter: ')
        step_item=stepline_splits[-1]
        step=step_item.split()[0]
        step=int(step)


        if best_ap<ap:
            best_ap=ap
            best_step=step
print('best ap: {}\tat:{}'.format(best_ap,best_step))
print('last ap: {}\tat:{}'.format(ap,step))