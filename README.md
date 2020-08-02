# PAMS: Quantized Super-Resolution via Parameterized Max Scale

This resposity is the official implementation of our paper. Our implementation is based on [EDSR(PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch).

### Dependencies
* Python3.6
* PyTorch >== 1.1.0

### Datasets
Please download DIV2K datasets from [here](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar) for training and [benchmark datasets](https://cv.snu.ac.kr/research/EDSR/benchmark.tar) for testing.

### Usage

* train

```
python main.py --scale 4 --k_bits 8 \
--pre_train ../pretrained/edsr_baseline_x4.pt \
--data_test Set14 --save edsr_x4/8bit/ \
--dir_data [DIR_DATA] --model EDSR               
```

* test

```
python main.py --scale 4 --k_bits 8 \ 
--pre_train ../pretrained/edsr_x4 --save_results \
--data_test Set5+Set14+B100+Urban100 \
--save edsr_x4/8bit/ --dir_data [DIR_DATA]  
--test_only --refine [REFINE] --model EDSR 
```

> set `--refine` to the saved model path for testing model.

More runnig scripts can be found in `run.sh`. 

* PSNR/SSIM

After saving the images, modify path in`metrics/calculate_PSNR_SSIM.m` to generate results.

```
matlab -nodesktop -nosplash -r "calculate_PSNR_SSIM('$dataset',$scale,$bit);quit"
```

refer to `metrics/run.sh` for more details.


### Experiment Results

| Model | Scale | bits        | PSNR(Set14)   |
| ----- | ----- | ----------- | ------------- |
| EDSR  | 4     | 32_backbone | 28.576/0.7813 |
|       |       | 8           | 28.585/0.7811 |
|       |       | 4           | 28.199/0.7725 |
| RDN   | 4     | 32_backbone | 28.669/0.7838 |
|       |       | 8           | 28.721/0.7848 |
|       |       | 4           | 27.536/0.7530 |

### Trained Models
The trained models of getting our paper's results can be download by [Google Drive](https://drive.google.com/open?id=14p3ZBs8VQdHkMWBa5kv_qN7b0w2qJq6c)

### Citations

If our paper helps your research, please cite it in your publications:




