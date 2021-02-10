# PAMS: Quantized Super-Resolution via Parameterized Max Scale

This resposity is the official implementation of our [paper](https://arxiv.org/pdf/2011.04212.pdf)

Our implementation is based on [EDSR(PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch).

### Dependencies
* Python3.6
* PyTorch == 1.1.0
* Pytorch == 
* coloredlogs >= 14.0
* scikit-image

### Datasets
Please download DIV2K datasets from [here](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar) for training and [benchmark datasets](https://cv.snu.ac.kr/research/EDSR/benchmark.tar) for testing.
Then, organise the dataset directory as follows:

```
datasets
  benchmark
  DIV2K
```


### Usage

* train

```
python main.py --scale 4 --k_bits 8 \
--pre_train ../pretrained/edsr_baseline_x4.pt \
--data_test Set14 --save edsr_x4/8bit/ \
--dir_data ./datasets --model EDSR               
```

* test

```
python main.py --scale 4 --k_bits 8 \ 
--pre_train ../pretrained/edsr_x4 --save_results \
--data_test Set5+Set14+B100+Urban100 \
--save edsr_x4/8bit/ --dir_data ./datasets  
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


### Trained Models
We also provide our baseline models below. Enjoy your training and testing!
[Google Drive](https://drive.google.com/open?id=14p3ZBs8VQdHkMWBa5kv_qN7b0w2qJq6c).


### Citations

If our paper helps your research, please cite it in your publications:
```
@article{li2020pams,
  title={PAMS: Quantized Super-Resolution via Parameterized Max Scale},
  author={Li, Huixia and Yan, Chenqian and Lin, Shaohui and Zheng, Xiawu and Li, Yuchao and Zhang, Baochang and Yang, Fan and Ji, Rongrong},
  journal={arXiv preprint arXiv:2011.04212},
  year={2020},
  publisher={Springer}
}
```