# PAMS

代码实现基于 [EDSR(PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch)

### 安装

* 依赖环境
  * python ==  3.6.8
  * PyTorch == 1.1.0
* 训练数据
  * DIV2K
* 测试数据
  * Set5 / Set14 / BSD100 / Urban100

### 用法

* 训练：

```
python main.py --scale 4 --k_bits 8 --w_at 1e+3 \
--pre_train ../pretrained/edsr_r16f64x4.pt --patch_size 192 --lr 1e-4 \
--print_every 1000 --data_test Set14 --epochs 30 --decay 10 \
--save edsr_x4/8bit/ --dir_data [DIR_DATA] --model EDSR               
```

* 测试：

```
python main.py --scale 4 --k_bits 8 \ 
--pre_train ../pretrained/edsr_x4 --save_results \
--data_test Set5+Set14+B100+Urban100 \
--save edsr_x4/8bit/ --dir_data [DIR_DATA]  --test_only 
--refine [REFINE] --model EDSR 
```

> 测试量化模型时讲`--refine`设置为为模型保存的路径



更多参数见`options.py`，运行代码见`run.sh`

* PSNR/SSIM

生成图像后修改`metrics/calculate_PSNR_SSIM.m`中的路径，生成PSNR/SSIM评估指标：

```
matlab -nodesktop -nosplash -r "calculate_PSNR_SSIM('$dataset',$scale,$bit);quit"
```

详见`metrics/run.sh`

 

### 实验结果

| Model | Scale | bits        | PSNR(Set14)   |
| ----- | ----- | ----------- | ------------- |
| EDSR  | 4     | 32_backbone | 28.576/0.7813 |
|       |       | 8           | 28.585/0.7811 |
|       |       | 4           | 28.199/0.7725 |
| RDN   | 4     | 32_backbone | 28.669/0.7838 |
|       |       | 8           | 28.721/0.7848 |
|       |       | 4           | 27.536/0.7530 |

模型地址 [Google Drive](https://drive.google.com/open?id=14p3ZBs8VQdHkMWBa5kv_qN7b0w2qJq6c)

