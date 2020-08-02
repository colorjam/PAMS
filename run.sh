VERSION=pams
DATA_DIR=/media/disk2/sr_data

edsr_x4() {
python main.py --scale 4 \
--k_bits $1 --model EDSR \
--pre_train ./pretrained/edsr_baseline_x4.pt --patch_size 192 \
--data_test Set14 \
--save "output/edsr_x4/$1bit" --dir_data $DATA_DIR --print_every 10
}

# edsr_x4 8


edsr_x4_eval() {
python main.py --scale 4 --model EDSR \
--k_bits $1 --save_results --test_only \
--data_test Set5+Set14+B100+Urban100  \
--save "output/edsr_x4/$1bit" --dir_data $DATA_DIR 
}

# edsr_x4 8

rdn_x4() {
python main.py --scale 4 --model RDN \
--k_bits $1 --model RDN \
--pre_train ./pretrained/rdn_baseline_x4.pt  --patch_size 96 \
--data_test Set14 \
--save "output/edsr_x4/$1bit" --dir_data $DATA_DIR
}

rdn_x4 8

rdn_x4_eval() {
python main.py --scale 4 --model RDN \
--k_bits $1  --save_results --test_only \
--data_test Set5+Set14+B100+Urban100 \
--save "output/edsr_x4/$1bit" --dir_data $DATA_DIR
}
 
