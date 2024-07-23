python train_net.py \
--config-file configs/maskrcnn/maskrcnn_dit_large_c6.yaml \
--num-gpus 3 MODEL.WEIGHTS \
../weights/dit-large-224-p16-500k-d7a2fb.pth OUTPUT_DIR outputs/maskrcnn_dit_large_c6

python train_net.py \
--config-file configs/maskrcnn/maskrcnn_dit_large_c6_resume.yaml \
--num-gpus 3  \
--resume 

python train_net.py \
--config-file configs/maskrcnn/maskrcnn_dit_base_c6.yaml \
--num-gpus 3 MODEL.WEIGHTS \
../weights/publaynet_dit-b_mrcnn.pth OUTPUT_DIR outputs/maskrcnn_dit_base_c6_publaynet

python train_net.py \
--config-file configs/maskrcnn/maskrcnn_dit_base_c6_resume.yaml \
--num-gpus 3 \
--resume

# python ./object_detection/inference.py \
# --image_path 1000layout_samples_to_purdue/0020.jpg \
# --output_file_name output.jpg \
# --config object_detection/publaynet_configs/maskrcnn/maskrcnn_dit_base.yaml  \
# --opts MODEL.WEIGHTS weights/publaynet_dit-b_mrcnn.pth 
