# FUD
# 中文文本识别算法 FUD

<a name="1"></a>
## 1. 简介

论文信息：

### FUD算法
本文提出单一视觉模型，从充分利用图像特征局部信息出发，利用注意力计算实现对特征充分挖掘，保证推理效率同时获得极佳的准确率，并且架构简单利于实施。针对注意力漂移导致的位置信息衰减，本方法提出位置编码矫正模块PECM实现对特征内位置信息的强化、矫正，显著促进了后续注意力计算有效性。针对中文复杂结构，提出局部-全局交融的注意力计算方式，并在注意力计算内部融入卷积以及引入本文提出的CLA通道来获取归纳偏置优势，配合位置信息的矫正强化，实现了对图像文本局部细节的充分发掘，模型仅需处理图像输入就可以兼顾特征细节与全局信息。
<a name="model"></a>

![1.PNG](..%2F..%2FDesktop%2F%CB%D8%B2%C4%2F%CD%BC%2F1.PNG)
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

*我们提供模型的预训练权重：https://pan.baidu.com/s/1b_YoPNE92pgBxD3_U-EFbQ?pwd=4466 
提取码：4466 

训练FUD模型时需要**更换配置文件**为`SVTR`的[配置文件](../../configs/rec/rec_svtrnet.yml)。

具体地，在完成数据准备后，便可以启动训练，训练命令如下：
```shell
#单卡训练（训练周期长，不建议）
python3 tools/train.py -c configs/rec/rec_svtrnet.yml

#多卡训练，通过--gpus参数指定卡号
python3 -m paddle.distributed.launch --gpus '0,1,2,3'  tools/train.py -c configs/rec/rec_svtrnet.yml
```

<a name="3-2"></a>
### 3.2 评估

可下载`SVTR`提供的模型文件和配置文件：[下载地址](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/rec_svtr_tiny_none_ctc_en_train.tar) ，以`SVTR-T`为例，使用如下命令进行评估：

```shell
# 下载包含SVTR-T的模型文件和配置文件的tar压缩包并解压
wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/rec_svtr_tiny_none_ctc_en_train.tar && tar xf rec_svtr_tiny_none_ctc_en_train.tar
# 注意将pretrained_model的路径设置为本地路径。
python3 -m paddle.distributed.launch --gpus '0' tools/eval.py -c ./rec_svtr_tiny_none_ctc_en_train/rec_svtr_tiny_6local_6global_stn_en.yml -o Global.pretrained_model=./rec_svtr_tiny_none_ctc_en_train/best_accuracy
```

<a name="3-3"></a>
### 3.3 预测

使用如下命令进行单张图片预测：
```shell
# 注意将pretrained_model的路径设置为本地路径。
python3 tools/infer_rec.py -c ./rec_svtr_tiny_none_ctc_en_train/rec_svtr_tiny_6local_6global_stn_en.yml -o Global.infer_img='./doc/imgs_words_en/word_10.png' Global.pretrained_model=./rec_svtr_tiny_none_ctc_en_train/best_accuracy
# 预测文件夹下所有图像时，可修改infer_img为文件夹，如 Global.infer_img='./doc/imgs_words_en/'。
```


<a name="4"></a>

**注意**：

- 如果您调整了训练时的输入分辨率，需要通过参数`rec_image_shape`设置为您需要的识别图像形状。
- 在推理时需要设置参数`rec_char_dict_path`指定字典，如果您修改了字典，请修改该参数为您的字典文件。
- 如果您修改了预处理方法，需修改`tools/infer/predict_rec.py`中SVTR的预处理为您的预处理方法。

<a name="4-2"></a>


## 引用

```bibtex
@article{
}
```

