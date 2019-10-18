VERSION=pams
DATA_DIR=/media/disk2/sr_data

edsr_x4() {
CUDA_VISIBLE_DEVICES=0 \
python main.py --scale 4 \
--k_bits $1 --w_at $2 --model EDSR \
--pre_train ./pretrained/edsr_r16f64x4.pt --patch_size 192 --lr $3 \
--print_every 1 --data_test Set14 --epochs 30 --decay 10 \
--save ${VERSION}/edsr_x4/$1bit/init_10 --dir_data $4 --save_results --reset
}

edsr_x4 8 1e+3 1e-4 $DATA_DIR 

edsr_x2() {
CUDA_VISIBLE_DEVICES=0 \
python main.py --scale 2 \
--k_bits $1 --w_at $2 --model EDSR \
--pre_train ./pretrained/edsr_r16f64x2.pt --patch_size 96 \
--print_every 1000 --data_test Set14 --epochs 30 --decay 10 \
--save $VERSION'/edsr_x2/'$1'bit/'$2'*at-'$3 --dir_data $4 
}

edsr_x4_eval() {
CUDA_VISIBLE_DEVICES=2 \
python main.py --scale 4 --model EDSR \
--k_bits $1 --pre_train ../pretrained/rdn_x4_backbone.pt --save_results \
--print_every 500 --data_test Set5+Set14+B100+Urban100  \
--save $VERSION'/edsr_x4/'$1'bit/'$2'*at-'$3 --dir_data $4  --test_only 
}


edsr_x2_eval() {
CUDA_VISIBLE_DEVICES=1 \
python main.py --scale 2 --model EDSR \
--k_bits $1 --pre_train ../pretrained/edsr_r16f64x2.pt --save_results \
--print_every 200 --data_test Set5+Set14+B100+Urban100  \
--save $VERSION'/edsr_x2/'$1'bit/'$2'*at-'$3 --dir_data $4  --test_only 
}


rdn_x4() {
CUDA_VISIBLE_DEVICES=1 \
python main.py --scale 4 --model RDN \
--k_bits $1 --w_at $2 --model RDN --lr $3 \
--pre_train ./pretrained/rdn_x4_3.pt --patch_size 96 \
--print_every 1 --data_test Set14 --epochs 30 --decay 10 \
--save "$VERSION/rdn_x4/$1bit/$2*at-lr*$3" --dir_data $4 --reset --save_results
}

# rdn_x4 8 1e+3 1e-3 $DATA_DIR 

rdn_x4_eval() {
CUDA_VISIBLE_DEVICES=$1 \
python main.py --scale 4 \
--k_bits $5 --w_at $2 --model RDN  \
--pre_train ../pretrained/rdn_x4_backbone.pt \
--print_every 1000 --data_test Set5+Set14+B100+Urban100 --test_only --save_results  \
--save "$VERSION/rdn_x4/$5bit/$2*at-$3" --dir_data $4 
}
 

rdn_x2() {
CUDA_VISIBLE_DEVICES=$1 \
python main.py --scale 2 \
--k_bits $5 --w_at $2 --model RDN \
--pre_train ../pretrained/rdn_x4_backbone.pt --patch_size 48 \
--print_every 1000 --data_test Set14 --epochs 30 --decay 10 \
--save "$VERSION/rdn_x2/$5bit/$2*at-$3" --dir_data $4 --model rdn 
}


rdn_x2_eval() {
CUDA_VISIBLE_DEVICES=$1 \
python main.py --scale 2 \
--k_bits $5 --w_at $2 --model RDN \
--pre_train ../pretrained/rdn_x4_backbone.pt \
--print_every 1000 --data_test Set5+Set14+B100+Urban100 --test_only --save_results \
--save "$VERSION/rdn_x2/$5bit/$2*at-$3" --dir_data $4 --model rdn 
}

# rdn_x4 0 1e+3 1 $DATA_DIR 8