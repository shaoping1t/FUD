# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import os
import sys
import json

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'


import cv2
import paddle
import paddle.vision.transforms as T

class Display:
    def __init__(self, model, layer):
        self.model = model
        self.feature_map_hook = []
        self.hooks = []
        if type(layer) == str:
            self.layer_name = layer
            self.target_layer = self.select_layer_by_name(layer)
        else:
            self.layer_name = None
            self.target_layer = layer
        self.hook_feature(self.target_layer)

    @staticmethod
    def show_network(model):
        for name, layer in model.named_sublayers():
            print(name)

    @staticmethod
    def transform(img, norm=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                  img_size=(224, 224)):
        transform_func = T.Compose([
            T.Resize(img_size),
            T.ToTensor(),
            T.Normalize(norm[0], norm[1])
        ])
        return transform_func(img)

    def select_layer_by_name(self, layer_name):
        for name, layer in self.model.named_sublayers():
            if name == layer_name:
                return layer
        raise NameError('No layer named: %s' % layer_name)

    def hook_feature(self, target_layer):
        # 需要注册的hook函数，拿到layer的输出
        def hook(layer, fea_in, fea_out):
            self.feature_map_hook.append(fea_out)
        hook_item = target_layer.register_forward_post_hook(hook=hook)
        self.hooks.append(hook_item)

    def clear_hooks(self):
        for hoo_item in self.hooks:
            hoo_item.remove()
        self.hooks.clear()

    def forward(self, image_tensor):
        """output:(目标特征图，网络输出结果)"""
        # 单张图片需转为batch_data:
        image_tensor = image_tensor.unsqueeze(0)
        out = image_tensor.clone()
        self.model.eval()  # 采用eval模式，保证BN层和dropout不出错
        final_out = self.model(out)
        feature_map = self.feature_map_hook.pop(0)
        return feature_map, final_out

    def get_gradcam(self, feature_map, net_out, target_class=None):
        """
        Args:
            feature_map: 最后的特征图
            net_out: 网络预测结果(before softmax)
            target_class: 关心的某个分类, 若为默认值None, 则会采用网络实际预测的分类
        Returns:特征图大小的注意力map
        """
        predict = net_out.squeeze() # [40, 6625]
        num_class = predict.size
        # 计算传入标签对应的grad_cam图
        # 没有target_class的话，默认为网络预测的分类
        if target_class is not None:
            target_class_tensor = paddle.to_tensor(target_class)
            one_hot_label = paddle.nn.functional.one_hot(target_class_tensor, num_class)
            predict = one_hot_label * predict
        loss = paddle.max(predict)
        loss.backward()
        # grad_map = paddle.grad(feature_map, feature_map)
        # print(grad_map)
        # grad_map = feature_map.grad
        # grad = paddle.mean(grad_map, (1), keepdim=True)
        grad = paddle.mean(feature_map, (2, 3), keepdim=True)  # 全局平均池化
        gradcam = paddle.sum(grad * feature_map, axis=1)
        gradcam = paddle.maximum(gradcam, paddle.to_tensor(0.))  # Relu
        gradcam = gradcam / (1e-7 + paddle.max(gradcam))  # 归一化至[0, 1]，1e-7避免分母为0
        feature_map.clear_grad()
        return gradcam

    def generate_image(self, img, gradcam):
        """融合gradcam和img"""
        grad_img = (gradcam * 255.)
        grad_img = paddle.transpose(grad_img, [1, 2, 0])  # 归一化至[0,255]区间，形状：[h,w,c]
        # 调整热图尺寸与图片一致、归一化
        heatmap = cv2.resize(grad_img.numpy(), (img.shape[1], img.shape[0])).astype('uint8')
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将热图转化为“伪彩热图”显示模式
        img = img[:, :, ::-1]  # 将RGB图片转为BGR
        superimposed_img = cv2.addWeighted(heatmap, .9, img, .3, 0.)  # 将特图叠加到原图片上
        return superimposed_img

    def save(self, image, target_class=None, file='gradcam.jpg'):
        img_tensor = self.transform(image)
        featurs, output = self.forward(img_tensor)
        grad_cam = self.get_gradcam(featurs, output, target_class)
        out_img = self.generate_image(image, grad_cam)
        cv2.imwrite(f'{file}', out_img)


from fud.data import create_operators, transform
from fud.modeling.architectures import build_model
from fud.postprocess import build_post_process
from fud.utils.save_load import load_model
from fud.utils.utility import get_image_file_list
import tools.program as program


def main():
    global_config = config['Global']

    # build post process
    post_process_class = build_post_process(config['PostProcess'],
                                            global_config)

    # build model
    if hasattr(post_process_class, 'character'):
        char_num = len(getattr(post_process_class, 'character'))
        if config['Architecture']["algorithm"] in ["Distillation",
                                                   ]:  # distillation model
            for key in config['Architecture']["Models"]:
                if config['Architecture']['Models'][key]['Head'][
                        'name'] == 'MultiHead':  # for multi head
                    out_channels_list = {}
                    if config['PostProcess'][
                            'name'] == 'DistillationSARLabelDecode':
                        char_num = char_num - 2
                    out_channels_list['CTCLabelDecode'] = char_num
                    out_channels_list['SARLabelDecode'] = char_num + 2
                    config['Architecture']['Models'][key]['Head'][
                        'out_channels_list'] = out_channels_list
                else:
                    config['Architecture']["Models"][key]["Head"][
                        'out_channels'] = char_num
        elif config['Architecture']['Head'][
                'name'] == 'MultiHead':  # for multi head loss
            out_channels_list = {}
            if config['PostProcess']['name'] == 'SARLabelDecode':
                char_num = char_num - 2
            out_channels_list['CTCLabelDecode'] = char_num
            out_channels_list['SARLabelDecode'] = char_num + 2
            config['Architecture']['Head'][
                'out_channels_list'] = out_channels_list
        else:  # base rec model
            config['Architecture']["Head"]['out_channels'] = char_num

    model = build_model(config['Architecture'])
    load_model(config, model)

    # layer = model.backbone
    # display = Display(model, layer)
    # img_folder = './infer/scene/'
    # img_list = os.listdir(img_folder)
    # for img_file in img_list:
    #     img_path = os.path.join(img_folder, img_file)
    #     img = cv2.imread(img_path)
    #     img = img[:, :, ::-1]
    #     save_path = os.path.join('./infer/res/', f'result_{img_file}')
    #     # 3. 保存模型注意力图
    #     display.save(img, file=save_path)

    # create data ops
    transforms = []
    for op in config['Eval']['dataset']['transforms']:
        op_name = list(op)[0]
        if 'Label' in op_name:
            continue
        elif op_name in ['RecResizeImg']:
            op[op_name]['infer_mode'] = True
        elif op_name == 'KeepKeys':
            if config['Architecture']['algorithm'] == "SRN":
                op[op_name]['keep_keys'] = [
                    'image', 'encoder_word_pos', 'gsrm_word_pos',
                    'gsrm_slf_attn_bias1', 'gsrm_slf_attn_bias2'
                ]
            elif config['Architecture']['algorithm'] == "SAR":
                op[op_name]['keep_keys'] = ['image', 'valid_ratio']
            elif config['Architecture']['algorithm'] == "RobustScanner":
                op[op_name][
                    'keep_keys'] = ['image', 'valid_ratio', 'word_positons']
            else:
                op[op_name]['keep_keys'] = ['image']
        transforms.append(op)
    global_config['infer_mode'] = True
    ops = create_operators(transforms, global_config)

    save_res_path = config['Global'].get('save_res_path',
                                         "./output/rec/predicts_rec.txt")
    if not os.path.exists(os.path.dirname(save_res_path)):
        os.makedirs(os.path.dirname(save_res_path))

    model.eval()

    with open(save_res_path, "w") as fout:
        for file in get_image_file_list(config['Global']['infer_img']):
            logger.info("infer_img: {}".format(file))
            with open(file, 'rb') as f:
                img = f.read()
                data = {'image': img}
            batch = transform(data, ops)
            if config['Architecture']['algorithm'] == "SRN":
                encoder_word_pos_list = np.expand_dims(batch[1], axis=0)
                gsrm_word_pos_list = np.expand_dims(batch[2], axis=0)
                gsrm_slf_attn_bias1_list = np.expand_dims(batch[3], axis=0)
                gsrm_slf_attn_bias2_list = np.expand_dims(batch[4], axis=0)

                others = [
                    paddle.to_tensor(encoder_word_pos_list),
                    paddle.to_tensor(gsrm_word_pos_list),
                    paddle.to_tensor(gsrm_slf_attn_bias1_list),
                    paddle.to_tensor(gsrm_slf_attn_bias2_list)
                ]
            if config['Architecture']['algorithm'] == "SAR":
                valid_ratio = np.expand_dims(batch[-1], axis=0)
                img_metas = [paddle.to_tensor(valid_ratio)]
            if config['Architecture']['algorithm'] == "RobustScanner":
                valid_ratio = np.expand_dims(batch[1], axis=0)
                word_positons = np.expand_dims(batch[2], axis=0)
                img_metas = [
                    paddle.to_tensor(valid_ratio),
                    paddle.to_tensor(word_positons),
                ]
            if config['Architecture']['algorithm'] == "CAN":
                image_mask = paddle.ones(
                    (np.expand_dims(
                        batch[0], axis=0).shape), dtype='float32')
                label = paddle.ones((1, 36), dtype='int64')
            images = np.expand_dims(batch[0], axis=0)
            images = paddle.to_tensor(images)
            if config['Architecture']['algorithm'] == "SRN":
                preds = model(images, others)
            elif config['Architecture']['algorithm'] == "SAR":
                preds = model(images, img_metas)
            elif config['Architecture']['algorithm'] == "RobustScanner":
                preds = model(images, img_metas)
            elif config['Architecture']['algorithm'] == "CAN":
                preds = model([images, image_mask, label])
            else:
                preds = model(images)

            post_result = post_process_class(preds)
            info = None
            if isinstance(post_result, dict):
                rec_info = dict()
                for key in post_result:
                    if len(post_result[key][0]) >= 2:
                        rec_info[key] = {
                            "label": post_result[key][0][0],
                            "score": float(post_result[key][0][1]),
                        }
                info = json.dumps(rec_info, ensure_ascii=False)
            elif isinstance(post_result, list) and isinstance(post_result[0],
                                                              int):
                # for RFLearning CNT branch 
                info = str(post_result[0])
            else:
                if len(post_result[0]) >= 2:
                    info = post_result[0][0] + "\t" + str(post_result[0][1])

            if info is not None:
                logger.info("\t result: {}".format(info))
                fout.write(file + "\t" + info + "\n")
    logger.info("success!")


if __name__ == '__main__':
    config, device, logger, vdl_writer = program.preprocess()
    main()
