# Copyright 2020-2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""ResNet."""
import math
import numpy as np
from scipy.stats import truncnorm
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from src.model_utils.config import config

# 定义卷积层权重初始化函数，使用方差缩放初始化
def conv_variance_scaling_initializer(in_channel, out_channel, kernel_size):
    # 计算输入通道数的fan_in（每个输出神经元与输入神经元的连接数）
    fan_in = in_channel * kernel_size * kernel_size
    # 初始化缩放因子为1.0
    scale = 1.0
    # 缩放因子除以最大值（1和fan_in之间的最大值）
    scale /= max(1., fan_in)
    # 计算标准差
    stddev = (scale ** 0.5) / .87962566103423978
    # 如果网络名称为resnet152，更新标准差值
    if config.net_name == "resnet152":
        stddev = (scale ** 0.5)
    # 设置截断正态分布的均值和标准差
    mu, sigma = 0, stddev
    # 从截断正态分布中采样权重
    weight = truncnorm(-2, 2, loc=mu, scale=sigma).rvs(out_channel * in_channel * kernel_size * kernel_size)
    # 将权重数组重塑为适当的形状（输出通道数，输入通道数，卷积核大小，卷积核大小）
    weight = np.reshape(weight, (out_channel, in_channel, kernel_size, kernel_size))
    return Tensor(weight, dtype=mstype.float32)


def _weight_variable(shape, factor=0.01):
    # 初始化权重矩阵，使用正态分布生成初始值，并乘以缩放因子
    init_value = np.random.randn(*shape).astype(np.float32) * factor
    return Tensor(init_value)


def calculate_gain(nonlinearity, param=None):
    """calculate_gain"""
    """计算激活函数的增益（gain）"""
    # 线性函数列表
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
    res = 0
    # 对于线性函数或者 Sigmoid 激活函数，增益值为 1
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        res = 1
    # 对于 Tanh 激活函数，增益值为 5/3
    elif nonlinearity == 'tanh':
        res = 5.0 / 3
    # 对于 ReLU 激活函数，增益值为 sqrt(2)
    elif nonlinearity == 'relu':
        res = math.sqrt(2.0)
        # 对于 Leaky ReLU 激活函数，增益值需要根据负斜率（neg_slope）计算
    elif nonlinearity == 'leaky_relu':
        # 如果未提供负斜率参数，则默认为 0.01
        if param is None:
            neg_slope = 0.01
        # 如果提供了合适的负斜率参数，使用给定值
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            neg_slope = param
        # 如果提供了无效的负斜率参数，抛出异常
        else:
            raise ValueError("neg_slope {} not a valid number".format(param))
        # 计算 Leaky ReLU 的增益值
        res = math.sqrt(2.0 / (1 + neg_slope ** 2))
    # 对于不支持的激活函数，抛出异常
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))
    # 返回计算得到的增益值
    return res


def _calculate_fan_in_and_fan_out(tensor):
    """_calculate_fan_in_and_fan_out"""
    """计算张量的 fan_in 和 fan_out"""

    dimensions = len(tensor)
    if dimensions < 2:# 张量维度数小于2，无法计算fan_in和fan_out
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
    if dimensions == 2:  # Linear若为线性层
        fan_in = tensor[1] # 输入的特征数
        fan_out = tensor[0] # 输出的特征数
    else:
        num_input_fmaps = tensor[1]
        num_output_fmaps = tensor[0]
        receptive_field_size = 1
        if dimensions > 2:# 如果维度数大于2，则计算receptive field大小
            receptive_field_size = tensor[2] * tensor[3]
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size
    return fan_in, fan_out


def _calculate_correct_fan(tensor, mode):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:# 如果mode不在支持的列表中，则抛出异常
        raise ValueError("Unsupported mode {}, please use one of {}".format(mode, valid_modes))
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)# 计算fan_in和fan_out
    return fan_in if mode == 'fan_in' else fan_out# 根据mode返回fan_in或fan_out

# 使用Kaiming正态分布初始化权重
def kaiming_normal(inputs_shape, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    # 计算fan-in或fan-out
    fan = _calculate_correct_fan(inputs_shape, mode)
    # 计算增益
    gain = calculate_gain(nonlinearity, a)
    # 计算标准差
    std = gain / math.sqrt(fan)
    # 从正态分布中采样权重
    return np.random.normal(0, std, size=inputs_shape).astype(np.float32)

# 使用Kaiming均匀分布初始化权重
def kaiming_uniform(inputs_shape, a=0., mode='fan_in', nonlinearity='leaky_relu'):
    fan = _calculate_correct_fan(inputs_shape, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    # 计算均匀分布的上下界
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    # 从均匀分布中采样权重
    return np.random.uniform(-bound, bound, size=inputs_shape).astype(np.float32)


def _conv3x3(in_channel, out_channel, stride=1, use_se=False, res_base=False):
    if use_se:
        weight = conv_variance_scaling_initializer(in_channel, out_channel, kernel_size=3)
    else:
        weight_shape = (out_channel, in_channel, 3, 3)
        weight = Tensor(kaiming_normal(weight_shape, mode="fan_out", nonlinearity='relu'))
        if config.net_name == "resnet152":
            weight = _weight_variable(weight_shape)
    if res_base:
        return nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride,
                         padding=1, pad_mode='pad', weight_init=weight)
    return nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride,
                     padding=0, pad_mode='same', weight_init=weight)


def _conv1x1(in_channel, out_channel, stride=1, use_se=False, res_base=False):
    #
    if use_se:
        weight = conv_variance_scaling_initializer(in_channel, out_channel, kernel_size=1)
    else:
        weight_shape = (out_channel, in_channel, 1, 1)
        weight = Tensor(kaiming_normal(weight_shape, mode="fan_out", nonlinearity='relu'))
        if config.net_name == "resnet152":
            weight = _weight_variable(weight_shape)
    if res_base:
        return nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride,
                         padding=0, pad_mode='pad', weight_init=weight)
    return nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride,
                     padding=0, pad_mode='same', weight_init=weight)


def _conv7x7(in_channel, out_channel, stride=1, use_se=False, res_base=False):
    if use_se:
        weight = conv_variance_scaling_initializer(in_channel, out_channel, kernel_size=7)
    else:
        weight_shape = (out_channel, in_channel, 7, 7)
        weight = Tensor(kaiming_normal(weight_shape, mode="fan_out", nonlinearity='relu'))
        if config.net_name == "resnet152":
            weight = _weight_variable(weight_shape)
    if res_base:
        return nn.Conv2d(in_channel, out_channel,
                         kernel_size=7, stride=stride, padding=3, pad_mode='pad', weight_init=weight)
    return nn.Conv2d(in_channel, out_channel,
                     kernel_size=7, stride=stride, padding=0, pad_mode='same', weight_init=weight)


def _bn(channel, res_base=False):
    if res_base:
        return nn.BatchNorm2d(channel, eps=1e-5, momentum=0.1,
                              gamma_init=1, beta_init=0, moving_mean_init=0, moving_var_init=1)
    return nn.BatchNorm2d(channel, eps=1e-4, momentum=0.9,
                          gamma_init=1, beta_init=0, moving_mean_init=0, moving_var_init=1)


def _bn_last(channel):
    return nn.BatchNorm2d(channel, eps=1e-4, momentum=0.9,
                          gamma_init=0, beta_init=0, moving_mean_init=0, moving_var_init=1)


def _fc(in_channel, out_channel, use_se=False):
    if use_se:
        weight = np.random.normal(loc=0, scale=0.01, size=out_channel * in_channel)
        weight = Tensor(np.reshape(weight, (out_channel, in_channel)), dtype=mstype.float32)
    else:
        weight_shape = (out_channel, in_channel)
        weight = Tensor(kaiming_uniform(weight_shape, a=math.sqrt(5)))
        if config.net_name == "resnet152":
            weight = _weight_variable(weight_shape)
    return nn.Dense(in_channel, out_channel, has_bias=True, weight_init=weight, bias_init=0)


class ResidualBlock(nn.Cell):
    """
    ResNet V1 残差块定义

    参数:
        in_channel (int): 输入通道数。
        out_channel (int):输出通道数。
        stride (int): 第一个卷积层的步幅大小。默认值为 1。
        use_se (bool): 是否启用 SE-ResNet50 网络。默认值为 False。
        se_block(bool): 是否在 SE-ResNet50 网络中使用 SE 块。默认值为 False。

    返回值:
        Tensor, output tensor.

    示例:
        >>> ResidualBlock(3, 256, stride=2)
    """
    expansion = 4

    def __init__(self,
                 in_channel,
                 out_channel,
                 stride=1,
                 use_se=False, se_block=False):
        super(ResidualBlock, self).__init__()
        self.stride = stride
        self.use_se = use_se
        self.se_block = se_block
        channel = out_channel // self.expansion# 计算通道数
        # 第1个卷积层
        self.conv1 = _conv1x1(in_channel, channel, stride=1, use_se=self.use_se)
        self.bn1 = _bn(channel)
        # 第2个卷积层
        if self.use_se and self.stride != 1:
            # 如果使用SE模块且步长不为1，使用池化操作
            self.e2 = nn.SequentialCell([_conv3x3(channel, channel, stride=1, use_se=True), _bn(channel),
                                         nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='same')])
        else:
            # 否则使用卷积操作
            self.conv2 = _conv3x3(channel, channel, stride=stride, use_se=self.use_se)
            self.bn2 = _bn(channel)
        # 第3个卷积层
        self.conv3 = _conv1x1(channel, out_channel, stride=1, use_se=self.use_se)
        self.bn3 = _bn(out_channel)
        # 如果使用Thor优化器或者网络模型为resnet152，使用特殊的BN层
        if config.optimizer == "Thor" or config.net_name == "resnet152":
            self.bn3 = _bn_last(out_channel)
        # 如果使用SE块，添加SE模块
        if self.se_block:
            self.se_global_pool = ops.ReduceMean(keep_dims=False)
            self.se_dense_0 = _fc(out_channel, int(out_channel / 4), use_se=self.use_se)
            self.se_dense_1 = _fc(int(out_channel / 4), out_channel, use_se=self.use_se)
            self.se_sigmoid = nn.Sigmoid()
            self.se_mul = ops.Mul()
        self.relu = nn.ReLU()

        self.down_sample = False

        # 如果步长不为1或输入通道数不等于输出通道数，需要进行下采样操作
        if stride != 1 or in_channel != out_channel:
            self.down_sample = True
        self.down_sample_layer = None

        if self.down_sample:
            # 如果使用SE模块，根据步长选择不同的下采样方式
            if self.use_se:
                if stride == 1:
                    self.down_sample_layer = nn.SequentialCell([_conv1x1(in_channel, out_channel,
                                                                         stride, use_se=self.use_se), _bn(out_channel)])
                else:
                    self.down_sample_layer = nn.SequentialCell([nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='same'),
                                                                _conv1x1(in_channel, out_channel, 1,
                                                                         use_se=self.use_se), _bn(out_channel)])
            else:
                # 否则直接使用卷积操作进行下采样
                self.down_sample_layer = nn.SequentialCell([_conv1x1(in_channel, out_channel, stride,
                                                                     use_se=self.use_se), _bn(out_channel)])


# 定义了 ResidualBlock 类的 construct 方法，该方法接收一个输入张量 x，用于计算 Residual Block 的输出。
    def construct(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 如果使用了 SE-ResNet50 网络且步幅不为1，则将 x 通过 SE 模块中的 e2 层进行处理，
        # 否则将 x 通过卷积层 conv2 进行卷积操作，再通过批归一化层 bn2 进行归一化，
        # 最后通过ReLU激活函数 relu 进行激活。
        if self.use_se and self.stride != 1:
            out = self.e2(out)
        else:
            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

        # 将输出结果再通过卷积层 conv3 进行卷积操作，再通过批归一化层 bn3 进行归一化。
        out = self.conv3(out)
        out = self.bn3(out)

        # 如果使用了 SE-ResNet50 网络且开启了 SE 块，则将输出结果 out 保存在变量 out_se 中，
        # 并对 out 进行全局平均池化操作，然后通过两个全连接层、ReLU激活函数和 Sigmoid 激活函数计算得到一个权重向量，
        # 再通过乘法操作与 out_se 相乘，得到最终的输出结果 out。
        if self.se_block:
            out_se = out
            out = self.se_global_pool(out, (2, 3))
            out = self.se_dense_0(out)
            out = self.relu(out)
            out = self.se_dense_1(out)
            out = self.se_sigmoid(out)
            out = ops.reshape(out, ops.shape(out) + (1, 1))
            out = self.se_mul(out, out_se)

        # 如果 down_sample 属性为 True，则通过下采样层 down_sample_layer 对 identity 进行下采样操作，以便与 out 进行加法操作。
        if self.down_sample:
            identity = self.down_sample_layer(identity)

        # 最后将 out 与 identity 相加，再通过ReLU激活函数 relu 进行激活，得到 Residual Block 的输出。
        out = out + identity
        out = self.relu(out)

        return out


class ResidualBlockBase(nn.Cell):
    """
    ResNet V1 基础的残差块定义

    参数:
        in_channel (int): 输入通道数。
        out_channel (int): 输出通道数。
        stride (int): 第一个卷积层的步幅大小。默认值为 1。
        use_se (bool): 是否启用 SE-ResNet50 网络。默认值为 False。
        se_block(bool): 是否在 SE-ResNet50 网络中使用 SE 块。默认值为 False。
        res_base (bool): 是否启用 ResNet18 参数设置。默认值为 True。

    返回值:
        Tensor, output tensor.

    示例:
        >>> ResidualBlockBase(3, 256, stride=2)
    """

    def __init__(self,
                 in_channel,
                 out_channel,
                 stride=1,
                 use_se=False,
                 se_block=False,
                 res_base=True):
        super(ResidualBlockBase, self).__init__()
        self.res_base = res_base

        # 创建两个卷积层 conv1 和 conv2，和两个批归一化层 bn1d 和 bn2d。这里的 _conv3x3 和 _bn 函数是对 nn.Conv2d 和 nn.BatchNorm2d 的封装，
        # 用于创建卷积层和批归一化层。其中，conv1 的输入通道数为 in_channel，输出通道数为 out_channel，
        # 步幅大小为 stride；conv2 的输入通道数和输出通道数都为 out_channel，步幅大小为 1。
        # 两个卷积层后都跟着一个批归一化层和 ReLU 激活函数。
        self.conv1 = _conv3x3(in_channel, out_channel, stride=stride, res_base=self.res_base)
        self.bn1d = _bn(out_channel)
        self.conv2 = _conv3x3(out_channel, out_channel, stride=1, res_base=self.res_base)
        self.bn2d = _bn(out_channel)
        self.relu = nn.ReLU()

        # 根据 stride 和 in_channel 是否等于 out_channel 来判断是否需要进行下采样操作，
        # 若需要，则设置 down_sample 为 True。
        self.down_sample = False
        if stride != 1 or in_channel != out_channel:
            self.down_sample = True

        # 最后，将 down_sample_layer 设置为 None，如果需要进行下采样操作，则创建一个下采样层 down_sample_layer，
        # 它包含一个 1x1 的卷积层和一个批归一化层，
        # 用于将输入张量进行下采样操作，使其与残差块的输出张量具有相同的通道数和形状。
        self.down_sample_layer = None
        if self.down_sample:
            self.down_sample_layer = nn.SequentialCell([_conv1x1(in_channel, out_channel, stride,
                                                                 use_se=use_se, res_base=self.res_base),
                                                        _bn(out_channel, res_base)])

    #  ResidualBlockBase 类的 construct 方法，用于计算残差块的输出。该方法接收一个输入张量 x，并返回一个输出张量 out。
    def construct(self, x):
        identity = x

        # 对输入张量进行卷积、批归一化和 ReLU 激活函数操作，得到 out 张量。
        # 首先，out 经过 conv1 卷积层，
        # 接着进行批归一化，最后进行 ReLU 激活函数操作。
        # 然后，out 再经过 conv2 卷积层，接着进行批归一化。 
        out = self.conv1(x)
        out = self.bn1d(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2d(out)

        # 如果需要进行下采样操作，则将输入张量 identity 带入下采样层 down_sample_layer 中，得到下采样后的张量。
        # 这个下采样层包含一个 1x1 的卷积层和一个批归一化层。
        if self.down_sample:
            identity = self.down_sample_layer(identity)
        
        # 将 out 张量和 identity 张量进行残差相加操作，得到残差块的输出张量。
        # 最后，将输出张量 out 经过 ReLU 激活函数，得到最终的输出。
        # 注意，这里的加法操作是张量级别的加法，而不是元素级别的加法。
        out = out + identity
        out = self.relu(out)

        return out


class ResNet(nn.Cell):
    """
    ResNet 网络架构

    参数:
        block (Cell):网络中使用的块类型。
        layer_nums (list): 不同阶段中块的数量。
        in_channels (list):不同阶段中块的数量。
        out_channels (list): 每个阶段中输出张量的通道数。
        strides (list): 每个阶段中卷积层的步幅大小。
        num_classes (int): 训练图像所属类别的数量。
        use_se (bool): 是否启用 SE-ResNet50 网络。默认为 False。
        se_block(bool):是否在 SE-ResNet50 网络的第三个和第四个阶段中使用 SE 块。默认为 False。
        res_base (bool): 是否启用 ResNet18 的参数设置。默认为 False。

    返回值:
        Tensor, output tensor.

    示例:
        >>> ResNet(ResidualBlock,
        >>>        [3, 4, 6, 3],
        >>>        [64, 256, 512, 1024],
        >>>        [256, 512, 1024, 2048],
        >>>        [1, 2, 2, 2],
        >>>        10)
    """

    def __init__(self,
                 block,
                 layer_nums,
                 in_channels,
                 out_channels,
                 strides,
                 num_classes,
                 use_se=False,
                 res_base=False):
        super(ResNet, self).__init__()

        # 检查输入的列表长度是否为4
        if not len(layer_nums) == len(in_channels) == len(out_channels) == 4:
            raise ValueError("the length of layer_num, in_channels, out_channels list must be 4!")
        self.use_se = use_se
        self.res_base = res_base
        self.se_block = False
        # 判断是否使用SE模块,如果启用 SE 模块，则构建 ResNet 的第一个卷积层，包括三个卷积层和两个批归一化层，用于提取特征。
        if self.use_se:
            self.se_block = True

        # 构建ResNet的第一个卷积层
        if self.use_se:
            self.conv1_0 = _conv3x3(3, 32, stride=2, use_se=self.use_se)
            self.bn1_0 = _bn(32)
            self.conv1_1 = _conv3x3(32, 32, stride=1, use_se=self.use_se)
            self.bn1_1 = _bn(32)
            self.conv1_2 = _conv3x3(32, 64, stride=1, use_se=self.use_se)
        else:
            self.conv1 = _conv7x7(3, 64, stride=2, res_base=self.res_base)
        self.bn1 = _bn(64, self.res_base)
        self.relu = ops.ReLU()

        # 构建ResNet的最大池化层
        if self.res_base:
            self.pad = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)))
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="valid")
        else:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")

        # 构建ResNet的四个stage，其中包括多个残差块，每个残差块都包含了多个卷积层和批归一化层。
        self.layer1 = self._make_layer(block,
                                       layer_nums[0],
                                       in_channel=in_channels[0],
                                       out_channel=out_channels[0],
                                       stride=strides[0],
                                       use_se=self.use_se)
        self.layer2 = self._make_layer(block,
                                       layer_nums[1],
                                       in_channel=in_channels[1],
                                       out_channel=out_channels[1],
                                       stride=strides[1],
                                       use_se=self.use_se)
        self.layer3 = self._make_layer(block,
                                       layer_nums[2],
                                       in_channel=in_channels[2],
                                       out_channel=out_channels[2],
                                       stride=strides[2],
                                       use_se=self.use_se,
                                       se_block=self.se_block)
        self.layer4 = self._make_layer(block,
                                       layer_nums[3],
                                       in_channel=in_channels[3],
                                       out_channel=out_channels[3],
                                       stride=strides[3],
                                       use_se=self.use_se,
                                       se_block=self.se_block)

        # 构建ResNet的全局平均池化层和全连接层
        self.mean = ops.ReduceMean(keep_dims=True)
        self.flatten = nn.Flatten()
        self.end_point = _fc(out_channels[3], num_classes, use_se=self.use_se)

    def _make_layer(self, block, layer_num, in_channel, out_channel, stride, use_se=False, se_block=False):
        """
        Make stage network of ResNet.

        Args:
            block (Cell): Resnet block.
            layer_num (int): Layer number.
            in_channel (int): Input channel.
            out_channel (int): Output channel.
            stride (int): Stride size for the first convolutional layer.
            se_block(bool): Use se block in SE-ResNet50 net. Default: False.
        Returns:
            SequentialCell, the output layer.

        Examples:
            >>> _make_layer(ResidualBlock, 3, 128, 256, 2)
        """
        # 构建ResNet的一个stage，包含多个残差块
        layers = []
        
        # 构建第1个残差块
        resnet_block = block(in_channel, out_channel, stride=stride, use_se=use_se)
        layers.append(resnet_block)
        # 如果使用SE模块，构建其它残差块
        if se_block:
            # 构建除最后一个残差块之外的其它残差块
            for _ in range(1, layer_num - 1):
                resnet_block = block(out_channel, out_channel, stride=1, use_se=use_se)
                layers.append(resnet_block)
            # 构建最后一个残差块，使用SE模块
            resnet_block = block(out_channel, out_channel, stride=1, use_se=use_se, se_block=se_block)
            layers.append(resnet_block)
        # 如果不使用SE模块，构建所有残差块
        else:
            for _ in range(1, layer_num):
                resnet_block = block(out_channel, out_channel, stride=1, use_se=use_se)
                layers.append(resnet_block)
        return nn.SequentialCell(layers)

    def construct(self, x):
        if self.use_se:
            x = self.conv1_0(x)
            x = self.bn1_0(x)
            x = self.relu(x)
            x = self.conv1_1(x)
            x = self.bn1_1(x)
            x = self.relu(x)
            x = self.conv1_2(x)
        else:
            x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.res_base:
            x = self.pad(x)
        c1 = self.maxpool(x)

        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        out = self.mean(c5, (2, 3))
        out = self.flatten(out)
        out = self.end_point(out)

        return out


def resnet18(class_num=10):
    """
    Get ResNet18 neural network.

    Args:
        class_num (int): Class number.

    Returns:
        Cell, cell instance of ResNet18 neural network.

    Examples:
        >>> net = resnet18(10)
    """
    return ResNet(ResidualBlockBase,
                  [2, 2, 2, 2],
                  [64, 64, 128, 256],
                  [64, 128, 256, 512],
                  [1, 2, 2, 2],
                  class_num,
                  res_base=True)

def resnet34(class_num=10):
    """
    Get ResNet34 neural network.

    Args:
        class_num (int): Class number.

    Returns:
        Cell, cell instance of ResNet34 neural network.

    Examples:
        >>> net = resnet18(10)
    """
    return ResNet(ResidualBlockBase,
                  [3, 4, 6, 3],
                  [64, 64, 128, 256],
                  [64, 128, 256, 512],
                  [1, 2, 2, 2],
                  class_num,
                  res_base=True)

def resnet50(class_num=10):
    """
    Get ResNet50 neural network.

    Args:
        class_num (int): Class number.

    Returns:
        Cell, cell instance of ResNet50 neural network.

    Examples:
        >>> net = resnet50(10)
    """
    return ResNet(ResidualBlock,
                  [3, 4, 6, 3],
                  [64, 256, 512, 1024],
                  [256, 512, 1024, 2048],
                  [1, 2, 2, 2],
                  class_num)


def se_resnet50(class_num=1001):
    """
    Get SE-ResNet50 neural network.

    Args:
        class_num (int): Class number.

    Returns:
        Cell, cell instance of SE-ResNet50 neural network.

    Examples:
        >>> net = se-resnet50(1001)
    """
    return ResNet(ResidualBlock,
                  [3, 4, 6, 3],
                  [64, 256, 512, 1024],
                  [256, 512, 1024, 2048],
                  [1, 2, 2, 2],
                  class_num,
                  use_se=True)


def resnet101(class_num=1001):
    """
    Get ResNet101 neural network.

    Args:
        class_num (int): Class number.

    Returns:
        Cell, cell instance of ResNet101 neural network.

    Examples:
        >>> net = resnet101(1001)
    """
    return ResNet(ResidualBlock,
                  [3, 4, 23, 3],
                  [64, 256, 512, 1024],
                  [256, 512, 1024, 2048],
                  [1, 2, 2, 2],
                  class_num)


def resnet152(class_num=1001):
    """
    Get ResNet152 neural network.

    Args:
        class_num (int): Class number.

    Returns:
        Cell, cell instance of ResNet152 neural network.

    Examples:
        # >>> net = resnet152(1001)
    """
    return ResNet(ResidualBlock,
                  [3, 8, 36, 3],
                  [64, 256, 512, 1024],
                  [256, 512, 1024, 2048],
                  [1, 2, 2, 2],
                  class_num)
