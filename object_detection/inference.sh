export CUDA_VISIBLE_DEVICES=3;
python object_detection/inference.py \
--image_path object_detection/publaynet_example.jpeg \
--output_file_name output.jpg \
--config object_detection/configs/maskrcnn/maskrcnn_dit_large.yaml \
--opts MODEL.WEIGHTS weights/publaynet_dit-l_mrcnn.pth


python ./object_detection/inference.py \
--image_path 1000layout_samples_to_purdue/0020.jpg \
--output_file_name output.jpg \
--config object_detection/publaynet_configs/maskrcnn/maskrcnn_dit_large.yaml  \
--opts MODEL.WEIGHTS weights/publaynet_dit-l_mrcnn.pth 


python ./object_detection/inference.py \
--image_path 1000layout_samples_to_purdue/0020.jpg \
--output_file_name output.jpg \
--config object_detection/publaynet_configs/maskrcnn/maskrcnn_dit_base.yaml  \
--opts MODEL.WEIGHTS weights/publaynet_dit-b_mrcnn.pth 
