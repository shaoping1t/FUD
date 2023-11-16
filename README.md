# 中文文本识别算法 FUD

<a name="1"></a>
## 1. 简介

论文信息：

### FUD算法
本文提出单一视觉模型，从充分利用图像特征局部信息出发，利用注意力计算实现对特征充分挖掘，保证推理效率同时获得极佳的准确率，并且架构简单利于实施。针对注意力漂移导致的位置信息衰减，本方法提出位置编码矫正模块PECM实现对特征内位置信息的强化、矫正，显著促进了后续注意力计算有效性。针对中文复杂结构，提出局部-全局交融的注意力计算方式，并在注意力计算内部融入卷积以及引入本文提出的CLA通道来获取归纳偏置优势，配合位置信息的矫正强化，实现了对图像文本局部细节的充分发掘，模型仅需处理图像输入就可以兼顾特征细节与全局信息。
<a name="model"></a>

<a name="2"></a>
## 2. 环境配置
我们使用paddle环境
```
python -m pip install paddlepaddle-gpu==2.5.2.post102 -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html

pip install -r requirements.txt
```

<a name="3"></a>
## 3. 模型训练、评估、预测

<a name="3-1"></a>
### 3.1 模型训练

#### 数据集准备
* 数据集来自于[Chinese Benckmark](https://arxiv.org/abs/2112.15093) ，FUD的训练评估使用该数据集。

* [中文数据集下载](https://github.com/fudanvi/benchmarking-chinese-text-recognition#download)

#### 训练

* 我们提供模型在三种数据集场景下的预训练权重：https://pan.baidu.com/s/1b_YoPNE92pgBxD3_U-EFbQ?pwd=4466 
提取码：4466 

训练FUD模型时需要**修改目录./configs/rec/下的配置文件**。
如FUD-tiny对应的为rec_fud_tiny_ch.yml文件。

完成数据准备后，可以进行训练了，操作如下：
```shell
#在.yml中如下位置修改数据集的地址
Train:
  dataset:
    name: LMDBDataSet
    data_dir: ./scene_train/ #训练集地址
    
Eval:
  dataset:
    name: LMDBDataSet
    data_dir: ./scene_test/ #测试集地址
    
# Global中需修改的参数如下
Global:
  save_model_dir: ./output/tiny/ #模型保存地址
  pretrained_model: ./tiny_scene.pdparams #加载预训练模型
  character_dict_path: ./fud/utils/ppocr_keys_v1.txt #字典
  save_res_path: ./output/rec/predicts_t.txt #识别结果保存地址

#完成参数配置可进行训练，下命令为单卡训练
python tools/train.py -c ./configs/rec/rec_fud_tiny_ch.yml

# 多卡训练
python -m paddle.distributed.launch tools/train.py -c configs/rec/rec_fud_tiny_ch.yml
```

<a name="3-2"></a>
### 3.2 评估


```shell
# 单卡评估命令
python tools/eval.py -c ./configs/rec/rec_fud_tiny_ch.yml
```

<a name="3-3"></a>
### 3.3 预测

使用如下命令进行对指定文件夹内的图片预测：
```shell
# 注意将Global.infer_img的路径设置为指定路径的文件夹。
python tools/infer_rec.py -c configs/rec/rec_fud_tiny_ch.yml  -o  Global.infer_img=./infer/scene/  Global.use_gpu=True
```

<a name="4"></a>



## 引用

```bibtex
@article{
}
```
