# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
"""Bert model."""

import math
import copy
import numpy as np
import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.ops.functional as F
from mindspore.common.initializer import TruncatedNormal, initializer
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter


class BertConfig:
    """
    Configuration for `BertModel`.

    Args:
        seq_length (int): Length of input sequence. Default: 128.
        vocab_size (int): The shape of each embedding vector. Default: 32000.
        hidden_size (int): Size of the bert encoder layers. Default: 768.
        num_hidden_layers (int): Number of hidden layers in the BertTransformer encoder
                           cell. Default: 12.
        num_attention_heads (int): Number of attention heads in the BertTransformer
                             encoder cell. Default: 12.
        intermediate_size (int): Size of intermediate layer in the BertTransformer
                           encoder cell. Default: 3072.
        hidden_act (str): Activation function used in the BertTransformer encoder
                    cell. Default: "gelu".
        hidden_dropout_prob (float): The dropout probability for BertOutput. Default: 0.1.
        attention_probs_dropout_prob (float): The dropout probability for
                                      BertAttention. Default: 0.1.
        max_position_embeddings (int): Maximum length of sequences used in this
                                 model. Default: 512.
        type_vocab_size (int): Size of token type vocab. Default: 16.
        initializer_range (float): Initialization value of TruncatedNormal. Default: 0.02.
        use_relative_positions (bool): Specifies whether to use relative positions. Default: False.
        dtype (:class:`mindspore.dtype`): Data type of the input. Default: mstype.float32.
        compute_type (:class:`mindspore.dtype`): Compute type in BertTransformer. Default: mstype.float32.
    """
    def __init__(self,
                 seq_length=128,
                 vocab_size=32000,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 initializer_range=0.02,
                 use_relative_positions=False,
                 dtype=mstype.float32,
                 compute_type=mstype.float32):
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.use_relative_positions = use_relative_positions
        self.dtype = dtype
        self.compute_type = compute_type


class EmbeddingLookup(nn.Cell):
    """
    一个具有固定字典和大小的嵌入查找表。

    参数：
        vocab_size (int)：嵌入字典的大小。
        embedding_size (int)：每个嵌入向量的大小。
        embedding_shape (list)：[batch_size，seq_length，embedding_size]，每个嵌入向量的形状。
        use_one_hot_embeddings (bool)：指定是否使用独热编码形式。默认值：False。
        initializer_range (float)：TruncatedNormal的初始化值。默认值：0.02。
    """

    def __init__(self,
                 vocab_size,
                 embedding_size,
                 embedding_shape,
                 use_one_hot_embeddings=False,
                 initializer_range=0.02):
        super(EmbeddingLookup, self).__init__()
        self.vocab_size = vocab_size
        self.use_one_hot_embeddings = use_one_hot_embeddings
        # 创建一个Parameter类型的embedding_table，用来存储嵌入向量
        self.embedding_table = Parameter(initializer
                                         (TruncatedNormal(initializer_range),
                                          [vocab_size, embedding_size]))
        # 定义ExpandDims操作，用于扩展input_ids的维度
        self.expand = P.ExpandDims()
        self.shape_flat = (-1,)
        # 定义Gather操作，用于从embedding_table中查找嵌入向量
        self.gather = P.Gather()
        # 定义OneHot操作，用于将flat_ids转换为独热编码形式
        self.one_hot = P.OneHot()
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        # 定义MatMul操作，用于计算独热编码与embedding_table的乘积
        self.array_mul = P.MatMul()
        # 定义Reshape操作，用于调整张量形状
        self.reshape = P.Reshape()
        self.shape = tuple(embedding_shape)

    def construct(self, input_ids):
        """Get output and embeddings lookup table"""
        # 使用ExpandDims操作扩展input_ids的维度
        extended_ids = self.expand(input_ids, -1)
        # 使用Reshape操作将extended_ids展平
        flat_ids = self.reshape(extended_ids, self.shape_flat)
        if self.use_one_hot_embeddings:
            # 使用OneHot操作将flat_ids转换为独热编码形式
            one_hot_ids = self.one_hot(flat_ids, self.vocab_size, self.on_value, self.off_value)
            # 使用MatMul操作计算one_hot_ids与embedding_table的乘积
            output_for_reshape = self.array_mul(
                one_hot_ids, self.embedding_table)
        else:
            # 使用Gather操作从embedding_table中查找flat_ids对应的嵌入向量
            output_for_reshape = self.gather(self.embedding_table, flat_ids, 0)
        # 使用Reshape操作将output_for_reshape调整为指定形状
        output = self.reshape(output_for_reshape, self.shape)
        return output, self.embedding_table.value()


class EmbeddingPostprocessor(nn.Cell):
    """
    嵌入后处理器将位置嵌入和令牌类型嵌入应用于词嵌入。

    参数：
        embedding_size (int)：每个嵌入向量的大小。
        embedding_shape (list)：[batch_size，seq_length，embedding_size]，每个嵌入向量的形状。
        use_token_type (bool)：指定是否使用令牌类型嵌入。默认值：False。
        token_type_vocab_size (int)：令牌类型词汇表大小。默认值：16。
        use_one_hot_embeddings (bool)：指定是否使用独热编码形式。默认值：False。
        initializer_range (float)：TruncatedNormal的初始化值。默认值：0.02。
        max_position_embeddings (int)：此模型中使用的序列的最大长度。默认值：512。
        dropout_prob (float)：丢弃概率。默认值：0.1。
    """
    def __init__(self,
                 embedding_size,
                 embedding_shape,
                 use_relative_positions=False,
                 use_token_type=False,
                 token_type_vocab_size=16,
                 use_one_hot_embeddings=False,
                 initializer_range=0.02,
                 max_position_embeddings=512,
                 dropout_prob=0.1):
        super(EmbeddingPostprocessor, self).__init__()
        self.use_token_type = use_token_type
        self.token_type_vocab_size = token_type_vocab_size
        self.use_one_hot_embeddings = use_one_hot_embeddings
        self.max_position_embeddings = max_position_embeddings
        # 创建一个nn.Embedding类型的token_type_embedding，用来存储令牌类型嵌入向量
        self.token_type_embedding = nn.Embedding(
            vocab_size=token_type_vocab_size,
            embedding_size=embedding_size,
            use_one_hot=use_one_hot_embeddings)
        self.shape_flat = (-1,)
        # 定义OneHot操作，用于将flat_ids转换为独热编码形式
        self.one_hot = P.OneHot()
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.1, mstype.float32)
        # 定义MatMul操作，用于计算独热编码与embedding_table的乘积
        self.array_mul = P.MatMul()
        # 定义Reshape操作，用于调整张量形状
        self.reshape = P.Reshape()
        self.shape = tuple(embedding_shape)
        # 定义Dropout操作，用于丢弃处理
        self.dropout = nn.Dropout(p=dropout_prob)
        # 定义Gather操作，用于从embedding_table中查找嵌入向量
        self.gather = P.Gather()
        self.use_relative_positions = use_relative_positions
        # 定义StridedSlice操作，用于从position_ids中获取位置id
        self.slice = P.StridedSlice()
        _, seq, _ = self.shape
        # 创建一个nn.Embedding类型的full_position_embedding，用来存储位置嵌入向量
        self.full_position_embedding = nn.Embedding(
            vocab_size=max_position_embeddings,
            embedding_size=embedding_size,
            use_one_hot=False)
        # 定义LayerNorm操作，用于层归一化处理
        self.layernorm = nn.LayerNorm((embedding_size,))
        # 定义position_ids，用于存储位置id
        self.position_ids = Tensor(np.arange(seq).reshape(-1, seq).astype(np.int32))
        self.add = P.Add()

    def construct(self, token_type_ids, word_embeddings):
        """嵌入后处理器将位置嵌入和令牌类型嵌入应用于词嵌入。"""
        # 将word_embeddings赋值给output
        output = word_embeddings
        if self.use_token_type:
            # 使用token_type_embedding查找token_type_ids对应的令牌类型嵌入向量
            token_type_embeddings = self.token_type_embedding(token_type_ids)
            # 将token_type_embeddings与output相加
            output = self.add(output, token_type_embeddings)
        if not self.use_relative_positions:
            # 获取output的形状
            shape = F.shape(output)
            # 从position_ids中获取位置id
            position_ids = self.position_ids[:, :shape[1]]
            # 使用full_position_embedding查找position_ids对应的位置嵌入向量
            position_embeddings = self.full_position_embedding(position_ids)
            # 将position_embeddings与output相加
            output = self.add(output, position_embeddings)
        # 对output进行层归一化处理
        output = self.layernorm(output)
        # 对output进行丢弃处理
        output = self.dropout(output)
        return output


class BertOutput(nn.Cell):
    """
    对隐藏状态应用线性计算并对输入应用残差计算。

    参数：
        in_channels (int)：输入通道数。
        out_channels (int)：输出通道数。
        initializer_range (float)：TruncatedNormal的初始化值。默认值：0.02。
        dropout_prob (float)：丢弃概率。默认值：0.1。
        compute_type (:class:`mindspore.dtype`)：BertTransformer中的计算类型。默认值：mstype.float32。
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 initializer_range=0.02,
                 dropout_prob=0.1,
                 compute_type=mstype.float32):
        super(BertOutput, self).__init__()
        # 创建一个nn.Dense类型的dense，用来进行线性计算
        self.dense = nn.Dense(in_channels, out_channels,
                              weight_init=TruncatedNormal(initializer_range)).to_float(compute_type)
        # 定义Dropout操作，用于丢弃处理
        self.dropout = nn.Dropout(p=dropout_prob)
        self.dropout_prob = dropout_prob
        # 定义Add操作，用于将input_tensor与output相加
        self.add = P.Add()
        # 定义LayerNorm操作，用于层归一化处理
        self.layernorm = nn.LayerNorm((out_channels,)).to_float(compute_type)
        # 定义Cast操作，用于转换数据类型
        self.cast = P.Cast()

    def construct(self, hidden_status, input_tensor):
        """对隐藏状态应用线性计算并对输入应用残差计算。"""
        # 使用dense对hidden_status进行线性计算
        output = self.dense(hidden_status)
        # 对output进行丢弃处理
        output = self.dropout(output)
        # 将input_tensor与output相加
        output = self.add(input_tensor, output)
        # 对output进行层归一化处理
        output = self.layernorm(output)
        return output


class RelaPosMatrixGenerator(nn.Cell):
    """
    生成输入之间的相对位置矩阵。

    参数：
        length (int)：要生成的矩阵的一维长度。
        max_relative_position (int)：相对位置的最大值。
    """
    def __init__(self, max_relative_position):
        super(RelaPosMatrixGenerator, self).__init__()
        self._max_relative_position = max_relative_position
        self._min_relative_position = -max_relative_position

        # 定义Tile操作，用于平铺range_vec_row_out和range_vec_col_out
        self.tile = P.Tile()
        # 定义Reshape操作，用于转换数据形状
        self.range_mat = P.Reshape()
        # 定义Sub操作，用于计算range_mat_out和transpose_out之差
        self.sub = P.Sub()
        # 定义ExpandDims操作，用于扩展input_ids的维度
        self.expanddims = P.ExpandDims()
        # 定义Cast操作，用于转换数据类型
        self.cast = P.Cast()

    def construct(self, length):
        """生成输入之间的相对位置矩阵。"""
        # 使用F.make_range和F.tuple_to_array生成一个范围向量range_vec_row_out，并将其转换为int32类型
        range_vec_row_out = self.cast(F.tuple_to_array(F.make_range(length)), mstype.int32)
        # 使用Reshape将range_vec_row_out转换为range_vec_col_out
        range_vec_col_out = self.range_mat(range_vec_row_out, (length, -1))
        # 使用Tile分别对range_vec_row_out和range_vec_col_out进行平铺
        tile_row_out = self.tile(range_vec_row_out, (length,))
        tile_col_out = self.tile(range_vec_col_out, (1, length))
        # 使用Reshape将tile_row_out和tile_col_out分别转换为range_mat_out和transpose_out
        range_mat_out = self.range_mat(tile_row_out, (length, length))
        transpose_out = self.range_mat(tile_col_out, (length, length))
        # 使用Sub计算range_mat_out和transpose_out之差得到distance_mat
        distance_mat = self.sub(range_mat_out, transpose_out)

        # 使用C.clip_by_value将distance_mat的值限制在_min_relative_position和_max_relative_position之间
        distance_mat_clipped = C.clip_by_value(distance_mat,
                                               self._min_relative_position,
                                               self._max_relative_position)

        # 将distance_mat_clipped的值平移_max_relative_position个单位
        final_mat = distance_mat_clipped + self._max_relative_position
        return final_mat


class RelaPosEmbeddingsGenerator(nn.Cell):
    """
    生成大小为[length, length, depth]的张量。

    参数：
        length (int)：要生成的矩阵的一维长度。
        depth (int)：每个注意力头的大小。
        max_relative_position (int)：相对位置的最大值。
        initializer_range (float)：TruncatedNormal的初始化值。
        use_one_hot_embeddings (bool)：指定是否使用独热编码形式。默认值：False。
    """
    def __init__(self,
                 depth,
                 max_relative_position,
                 initializer_range,
                 use_one_hot_embeddings=False):
        super(RelaPosEmbeddingsGenerator, self).__init__()
        self.depth = depth
        self.vocab_size = max_relative_position * 2 + 1
        self.use_one_hot_embeddings = use_one_hot_embeddings

        # 创建一个Parameter类型的embeddings_table，用来存储嵌入向量
        self.embeddings_table = Parameter(
            initializer(TruncatedNormal(initializer_range),
                        [self.vocab_size, self.depth]))

        # 定义RelaPosMatrixGenerator操作，用于生成相对位置矩阵
        self.relative_positions_matrix = RelaPosMatrixGenerator(max_relative_position=max_relative_position)
        # 定义Reshape操作，用于调整张量形状
        self.reshape = P.Reshape()
        # 定义Shape操作，用于获取张量形状
        self.shape = P.Shape()
        # 定义Gather操作，用于从embeddings_table中查找嵌入向量
        self.gather = P.Gather()  # index_select
        # 定义BatchMatMul操作，用于计算独热编码与embeddings_table的乘积
        self.matmul = P.BatchMatMul()

    def construct(self, length):
        """为每个相对位置生成维度为depth的嵌入。"""
        # 使用relative_positions_matrix生成相对位置矩阵relative_positions_matrix_out
        relative_positions_matrix_out = self.relative_positions_matrix(length)

        if self.use_one_hot_embeddings:
            # 使用Reshape将relative_positions_matrix_out展平
            flat_relative_positions_matrix = self.reshape(relative_positions_matrix_out, (-1,))
            # 使用ops.one_hot将flat_relative_positions_matrix转换为独热编码形式
            one_hot_relative_positions_matrix = ops.one_hot(flat_relative_positions_matrix,
                                                            self.vocab_size, 1.0, 0.0)
            # 使用BatchMatMul计算one_hot_relative_positions_matrix与embeddings_table的乘积
            embeddings = self.matmul(one_hot_relative_positions_matrix, self.embeddings_table)
            # 获取relative_positions_matrix_out的形状，并在其末尾添加depth维度
            my_shape = self.shape(relative_positions_matrix_out) + (self.depth,)
            # 使用Reshape将embeddings调整为指定形状
            embeddings = self.reshape(embeddings, my_shape)
        else:
            # 使用Gather从embeddings_table中查找relative_positions_matrix_out对应的嵌入向量
            embeddings = self.gather(self.embeddings_table,
                                     relative_positions_matrix_out, 0)
        return embeddings


class SaturateCast(nn.Cell):
    """
    执行安全的饱和转换。此操作在转换之前应用适当的限制，以防止值溢出或下溢。

    参数：
        src_type (:class:`mindspore.dtype`)：输入张量元素的类型。默认值：mstype.float32。
        dst_type (:class:`mindspore.dtype`)：输出张量元素的类型。默认值：mstype.float32。
    """
    def __init__(self, src_type=mstype.float32, dst_type=mstype.float32):
        super(SaturateCast, self).__init__()
        # 根据dst_type获取np_type，并根据np_type获取tensor_min_type和tensor_max_type
        np_type = mstype.dtype_to_nptype(dst_type)

        self.tensor_min_type = float(np.finfo(np_type).min)
        self.tensor_max_type = float(np.finfo(np_type).max)

        # 定义Minimum操作，用于将out与tensor_max_type取最小值
        self.min_op = P.Minimum()
        # 定义Maximum操作，用于将x与tensor_min_type取最大值
        self.max_op = P.Maximum()
        # 定义Cast操作，用于转换数据类型
        self.cast = P.Cast()

    def construct(self, x):
        """执行安全的饱和转换。"""
        # 使用Maximum将x与tensor_min_type取最大值，并将结果赋值给out
        out = self.max_op(x, self.tensor_min_type)
        # 使用Minimum将out与tensor_max_type取最小值，并将结果赋值给out
        out = self.min_op(out, self.tensor_max_type)
        # 将out转换为dst_type类型并返回
        return self.cast(out, self.dst_type)


class BertAttention(nn.Cell):
    """
    Apply multi-headed attention from "from_tensor" to "to_tensor".

    Args:
        from_tensor_width (int): Size of last dim of from_tensor.
        to_tensor_width (int): Size of last dim of to_tensor.
        num_attention_heads (int): Number of attention heads. Default: 1.
        size_per_head (int): Size of each attention head. Default: 512.
        query_act (str): Activation function for the query transform. Default: None.
        key_act (str): Activation function for the key transform. Default: None.
        value_act (str): Activation function for the value transform. Default: None.
        has_attention_mask (bool): Specifies whether to use attention mask. Default: False.
        attention_probs_dropout_prob (float): The dropout probability for
                                      BertAttention. Default: 0.0.
        use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form. Default: False.
        initializer_range (float): Initialization value of TruncatedNormal. Default: 0.02.
        use_relative_positions (bool): Specifies whether to use relative positions. Default: False.
        compute_type (:class:`mindspore.dtype`): Compute type in BertAttention. Default: mstype.float32.
    """
    def __init__(self,
                 from_tensor_width,
                 to_tensor_width,
                 num_attention_heads=1,
                 size_per_head=512,
                 query_act=None,
                 key_act=None,
                 value_act=None,
                 has_attention_mask=False,
                 attention_probs_dropout_prob=0.0,
                 use_one_hot_embeddings=False,
                 initializer_range=0.02,
                 use_relative_positions=False,
                 compute_type=mstype.float32):

        super(BertAttention, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.size_per_head = size_per_head
        self.has_attention_mask = has_attention_mask
        self.use_relative_positions = use_relative_positions

        # 计算scores_mul的值
        self.scores_mul = 1.0 / math.sqrt(float(self.size_per_head))
        # 定义Reshape操作
        self.reshape = P.Reshape()
        # 定义shape_from_2d和shape_to_2d变量
        self.shape_from_2d = (-1, from_tensor_width)
        self.shape_to_2d = (-1, to_tensor_width)
        weight = TruncatedNormal(initializer_range)
        units = num_attention_heads * size_per_head
        # 创建三个nn.Dense类型的层：query_layer，key_layer和value_layer
        self.query_layer = nn.Dense(from_tensor_width,
                                    units,
                                    activation=query_act,
                                    weight_init=weight).to_float(compute_type)
        self.key_layer = nn.Dense(to_tensor_width,
                                  units,
                                  activation=key_act,
                                  weight_init=weight).to_float(compute_type)
        self.value_layer = nn.Dense(to_tensor_width,
                                    units,
                                    activation=value_act,
                                    weight_init=weight).to_float(compute_type)

        # 定义BatchMatMul操作
        self.matmul_trans_b = P.BatchMatMul(transpose_b=True)
        # 定义Mul操作
        self.multiply = P.Mul()
        # 定义Transpose操作
        self.transpose = P.Transpose()
        self.trans_shape = (0, 2, 1, 3)
        self.trans_shape_relative = (2, 0, 1, 3)
        self.trans_shape_position = (1, 2, 0, 3)
        # 计算multiply_data的值
        self.multiply_data = -10000.0
        # 定义BatchMatMul操作
        self.matmul = P.BatchMatMul()

        # 定义Softmax操作
        self.softmax = nn.Softmax()
        # 定义Dropout操作
        self.dropout = nn.Dropout(p=attention_probs_dropout_prob)

        if self.has_attention_mask:
            # 定义ExpandDims，Sub，Add，Cast和DType操作
            self.expand_dims = P.ExpandDims()
            self.sub = P.Sub()
            self.add = P.Add()
            self.cast = P.Cast()
            self.get_dtype = P.DType()

        # 定义shape_return变量
        self.shape_return = (-1, num_attention_heads * size_per_head)

        # 创建一个SaturateCast类型的cast_compute_type
        self.cast_compute_type = SaturateCast(dst_type=compute_type)
        if self.use_relative_positions:
            # 创建一个RelaPosEmbeddingsGenerator类型的_generate_relative_positions_embeddings
            self._generate_relative_positions_embeddings = \
                RelaPosEmbeddingsGenerator(depth=size_per_head,
                                           max_relative_position=16,
                                           initializer_range=initializer_range,
                                           use_one_hot_embeddings=use_one_hot_embeddings)

    def construct(self, from_tensor, to_tensor, attention_mask):
        """reshape 2d/3d input tensors to 2d"""
        # 使用F.shape获取attention_mask的第三维度shape_from
        shape_from = F.shape(attention_mask)[2]
        # 使用F.depend将from_tensor与shape_from关联起来
        from_tensor = F.depend(from_tensor, shape_from)
        # 使用Reshape将from_tensor和to_tensor分别转换为from_tensor_2d和to_tensor_2d
        from_tensor_2d = self.reshape(from_tensor, self.shape_from_2d)
        to_tensor_2d = self.reshape(to_tensor, self.shape_to_2d)
        # 使用query_layer，key_layer和value_layer分别计算查询，键和值
        query_out = self.query_layer(from_tensor_2d)
        key_out = self.key_layer(to_tensor_2d)
        value_out = self.value_layer(to_tensor_2d)

        # 使用Reshape和Transpose分别对query_out和key_out进行调整
        query_layer = self.reshape(query_out, (-1, shape_from, self.num_attention_heads, self.size_per_head))
        query_layer = self.transpose(query_layer, self.trans_shape)
        key_layer = self.reshape(key_out, (-1, shape_from, self.num_attention_heads, self.size_per_head))
        key_layer = self.transpose(key_layer, self.trans_shape)

        # 使用BatchMatMul计算query_layer和key_layer的乘积
        attention_scores = self.matmul_trans_b(query_layer, key_layer)

        # use_relative_position，补充逻辑
        if self.use_relative_positions:
            # relations_keys是[F|T，F|T，H]
            relations_keys = self._generate_relative_positions_embeddings(shape_from)
            relations_keys = self.cast_compute_type(relations_keys)
            # query_layer_t是[F，B，N，H]
            query_layer_t = self.transpose(query_layer, self.trans_shape_relative)
            # query_layer_r是[F，B * N，H]
            query_layer_r = self.reshape(query_layer_t,
                                         (shape_from,
                                          -1,
                                          self.size_per_head))
            # key_position_scores是[F，B * N，F|T]
            key_position_scores = self.matmul_trans_b(query_layer_r,
                                                      relations_keys)
            # key_position_scores_r是[F，B，N，F|T]
            key_position_scores_r = self.reshape(key_position_scores,
                                                 (shape_from,
                                                  -1,
                                                  self.num_attention_heads,
                                                  shape_from))
            # key_position_scores_r_t是[B，N，F，F|T]
            key_position_scores_r_t = self.transpose(key_position_scores_r,
                                                     self.trans_shape_position)
            attention_scores = attention_scores + key_position_scores_r_t

            # 使用Mul将scores_mul与attention_scores相乘
        attention_scores = self.multiply(self.scores_mul, attention_scores)

        if self.has_attention_mask:
            # 使用ExpandDims，Sub，Cast，Add等操作对attention_mask进行处理
            attention_mask = self.expand_dims(attention_mask, 1)
            multiply_out = self.sub(self.cast(F.tuple_to_array((1.0,)), self.get_dtype(attention_scores)),
                                    self.cast(attention_mask, self.get_dtype(attention_scores)))

            adder = self.multiply(multiply_out, self.multiply_data)
            attention_scores = self.add(adder, attention_scores)

        # 使用Softmax对attention_scores进行处理
        attention_probs = self.softmax(attention_scores)
        # 使用Dropout对attention_probs进行丢弃处理
        attention_probs = self.dropout(attention_probs)

        # 使用Reshape和Transpose分别对value_out进行调整
        value_layer = self.reshape(value_out, (-1, shape_from, self.num_attention_heads, self.size_per_head))
        value_layer = self.transpose(value_layer, self.trans_shape)
        # 使用BatchMatMul计算attention_probs和value_layer的乘积
        context_layer = self.matmul(attention_probs, value_layer)

        # use_relative_position，补充逻辑
        # 如果use_relative_positions为True，则继续执行一些操作
        if self.use_relative_positions:
            # 使用_generate_relative_positions_embeddings生成相对位置嵌入relations_values
            relations_values = self._generate_relative_positions_embeddings(shape_from)
            # 使用cast_compute_type将relations_values转换为计算类型
            relations_values = self.cast_compute_type(relations_values)
            # 使用Transpose和Reshape分别对attention_probs进行调整
            attention_probs_t = self.transpose(attention_probs, self.trans_shape_relative)
            attention_probs_r = self.reshape(
                attention_probs_t,
                (shape_from,
                 -1,
                 shape_from))
            # 使用BatchMatMul计算attention_probs_r和relations_values的乘积
            value_position_scores = self.matmul(attention_probs_r,
                                                relations_values)
            # 使用Reshape和Transpose分别对value_position_scores进行调整
            value_position_scores_r = self.reshape(value_position_scores,
                                                   (shape_from,
                                                    -1,
                                                    self.num_attention_heads,
                                                    self.size_per_head))
            value_position_scores_r_t = self.transpose(value_position_scores_r,
                                                       self.trans_shape_position)
            # 将value_position_scores_r_t与context_layer相加并更新context_layer
            context_layer = context_layer + value_position_scores_r_t

        # 使用Transpose和Reshape分别对context_layer进行调整
        context_layer = self.transpose(context_layer, self.trans_shape)
        context_layer = self.reshape(context_layer, self.shape_return)

        # 返回context_layer
        return context_layer


class BertSelfAttention(nn.Cell):
    """
    应用自注意力。

    参数：
        hidden_size (int)：Bert编码器层的大小。
        num_attention_heads (int)：注意力头数。默认值：12。
        attention_probs_dropout_prob (float)：BertAttention的丢弃概率。默认值：0.1。
        use_one_hot_embeddings (bool)：指定是否使用独热编码形式。默认值：False。
        initializer_range (float)：TruncatedNormal的初始化值。默认值：0.02。
        hidden_dropout_prob (float)：BertOutput的丢弃概率。默认值：0.1。
        use_relative_positions (bool)：指定是否使用相对位置。默认值：False。
        compute_type (:class:`mindspore.dtype`)：BertSelfAttention中的计算类型。默认值：mstype.float32。
    """
    def __init__(self,
                 hidden_size,
                 num_attention_heads=12,
                 attention_probs_dropout_prob=0.1,
                 use_one_hot_embeddings=False,
                 initializer_range=0.02,
                 hidden_dropout_prob=0.1,
                 use_relative_positions=False,
                 compute_type=mstype.float32):
        super(BertSelfAttention, self).__init__()
        # 检查hidden_size是否是num_attention_heads的倍数
        if hidden_size % num_attention_heads != 0:
            raise ValueError("The hidden size (%d) is not a multiple of the number "
                             "of attention heads (%d)" % (hidden_size, num_attention_heads))

        # 计算size_per_head的值
        self.size_per_head = int(hidden_size / num_attention_heads)

        # 创建一个BertAttention类型的attention
        self.attention = BertAttention(
            from_tensor_width=hidden_size,
            to_tensor_width=hidden_size,
            num_attention_heads=num_attention_heads,
            size_per_head=self.size_per_head,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            use_one_hot_embeddings=use_one_hot_embeddings,
            initializer_range=initializer_range,
            use_relative_positions=use_relative_positions,
            has_attention_mask=True,
            compute_type=compute_type)

        # 创建一个BertOutput类型的output
        self.output = BertOutput(in_channels=hidden_size,
                                 out_channels=hidden_size,
                                 initializer_range=initializer_range,
                                 dropout_prob=hidden_dropout_prob,
                                 compute_type=compute_type)
        # 定义Reshape操作
        self.reshape = P.Reshape()
        # 定义shape变量
        self.shape = (-1, hidden_size)

    def construct(self, input_tensor, attention_mask):
        # 使用attention计算注意力
        attention_output = self.attention(input_tensor, input_tensor, attention_mask)
        # 使用output计算输出
        output = self.output(attention_output, input_tensor)
        return output


class BertEncoderCell(nn.Cell):
    """
    用于BertTransformer的编码器单元。

    参数：
        hidden_size (int)：Bert编码器层的大小。默认值：768。
        num_attention_heads (int)：注意力头数。默认值：12。
        intermediate_size (int)：中间层的大小。默认值：3072。
        attention_probs_dropout_prob (float)：BertAttention的丢弃概率。默认值：0.02。
        use_one_hot_embeddings (bool)：指定是否使用独热编码形式。默认值：False。
        initializer_range (float)：TruncatedNormal的初始化值。默认值：0.02。
        hidden_dropout_prob (float)：BertOutput的丢弃概率。默认值：0.1。
        use_relative_positions (bool)：指定是否使用相对位置。默认值：False。
        hidden_act (str)：激活函数。默认值："gelu"。
        compute_type (:class:`mindspore.dtype`)：注意力中的计算类型。默认值：mstype.float32。
    """
    def __init__(self,
                 hidden_size=768,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 attention_probs_dropout_prob=0.02,
                 use_one_hot_embeddings=False,
                 initializer_range=0.02,
                 hidden_dropout_prob=0.1,
                 use_relative_positions=False,
                 hidden_act="gelu",
                 compute_type=mstype.float32):
        super(BertEncoderCell, self).__init__()
        # 创建一个BertSelfAttention类型的attention
        self.attention = BertSelfAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            use_one_hot_embeddings=use_one_hot_embeddings,
            initializer_range=initializer_range,
            hidden_dropout_prob=hidden_dropout_prob,
            use_relative_positions=use_relative_positions,
            compute_type=compute_type)
        # 创建一个nn.Dense类型的intermediate
        self.intermediate = nn.Dense(in_channels=hidden_size,
                                     out_channels=intermediate_size,
                                     activation=hidden_act,
                                     weight_init=TruncatedNormal(initializer_range)).to_float(compute_type)
        # 创建一个BertOutput类型的output
        self.output = BertOutput(in_channels=intermediate_size,
                                 out_channels=hidden_size,
                                 initializer_range=initializer_range,
                                 dropout_prob=hidden_dropout_prob,
                                 compute_type=compute_type)

    def construct(self, hidden_states, attention_mask):
        # 自注意力
        attention_output = self.attention(hidden_states, attention_mask)
        # 喂入construct
        intermediate_output = self.intermediate(attention_output)
        # 添加并规范化
        output = self.output(intermediate_output, attention_output)
        return output


class BertTransformer(nn.Cell):
    """
    多层bert变换器。

    参数：
        hidden_size (int)：编码器层的大小。
        num_hidden_layers (int)：编码器单元中隐藏层的数量。
        num_attention_heads (int)：编码器单元中注意力头数。默认值：12。
        intermediate_size (int)：编码器单元中中间层的大小。默认值：3072。
        attention_probs_dropout_prob (float)：BertAttention的丢弃概率。默认值：0.1。
        use_one_hot_embeddings (bool)：指定是否使用独热编码形式。默认值：False。
        initializer_range (float)：TruncatedNormal的初始化值。默认值：0.02。
        hidden_dropout_prob (float)：BertOutput的丢弃概率。默认值：0.1。
        use_relative_positions (bool)：指定是否使用相对位置。默认值：False。
        hidden_act (str)：编码器单元中使用的激活函数。默认值："gelu"。
        compute_type (:class:`mindspore.dtype`)：BertTransformer中的计算类型。默认值：mstype.float32。
        return_all_encoders (bool)：指定是否返回所有编码器。默认值：False。
    """
    def __init__(self,
                 hidden_size,
                 num_hidden_layers,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 attention_probs_dropout_prob=0.1,
                 use_one_hot_embeddings=False,
                 initializer_range=0.02,
                 hidden_dropout_prob=0.1,
                 use_relative_positions=False,
                 hidden_act="gelu",
                 compute_type=mstype.float32,
                 return_all_encoders=False):
        super(BertTransformer, self).__init__()
        # 初始化return_all_encoders变量
        self.return_all_encoders = return_all_encoders

        # 创建一个空列表layers
        layers = []
        for _ in range(num_hidden_layers):
            # 在layers中添加num_hidden_layers个BertEncoderCell类型的layer
            layer = BertEncoderCell(hidden_size=hidden_size,
                                    num_attention_heads=num_attention_heads,
                                    intermediate_size=intermediate_size,
                                    attention_probs_dropout_prob=attention_probs_dropout_prob,
                                    use_one_hot_embeddings=use_one_hot_embeddings,
                                    initializer_range=initializer_range,
                                    hidden_dropout_prob=hidden_dropout_prob,
                                    use_relative_positions=use_relative_positions,
                                    hidden_act=hidden_act,
                                    compute_type=compute_type)
            layers.append(layer)

        # 将layers转换为nn.CellList类型并赋值给self.layers
        self.layers = nn.CellList(layers)

        # 定义Reshape操作
        self.reshape = P.Reshape()
        # 定义shape变量
        self.shape = (-1, hidden_size)

    def construct(self, input_tensor, attention_mask):
        """多层bert变换器。"""
        # 使用Reshape将input_tensor调整为指定形状
        prev_output = self.reshape(input_tensor, self.shape)

        all_encoder_layers = ()
        for layer_module in self.layers:
            # 使用layer_module计算层输出
            layer_output = layer_module(prev_output, attention_mask)
            prev_output = layer_output

            if self.return_all_encoders:
                # 获取input_tensor的形状
                shape = F.shape(input_tensor)
                # 使用Reshape将layer_output调整为指定形状
                layer_output = self.reshape(layer_output, shape)
                all_encoder_layers = all_encoder_layers + (layer_output,)

        if not self.return_all_encoders:
            # 获取input_tensor的形状
            shape = F.shape(input_tensor)
            # 使用Reshape将prev_output调整为指定形状
            prev_output = self.reshape(prev_output, shape)
            all_encoder_layers = all_encoder_layers + (prev_output,)
        return all_encoder_layers


class CreateAttentionMaskFromInputMask(nn.Cell):
    """
    Create attention mask according to input mask.

    Args:
        config (Class): Configuration for BertModel.
    """
    def __init__(self, config):
        super(CreateAttentionMaskFromInputMask, self).__init__()
        self.input_mask = None

        self.cast = P.Cast()
        self.reshape = P.Reshape()

    def construct(self, input_mask):
        seq_length = F.shape(input_mask)[1]
        attention_mask = self.cast(self.reshape(input_mask, (-1, 1, seq_length)), mstype.float32)
        return attention_mask


class BertModel(nn.Cell):
    """
    双向编码器表示的变换器。

    参数：
        config (Class)：BertModel的配置。
        is_training (bool)：True表示训练模式。False表示评估模式。
        use_one_hot_embeddings (bool)：指定是否使用独热编码形式。默认值：False。
    """
    def __init__(self,
                 config,
                 is_training,
                 use_one_hot_embeddings=False):
        super(BertModel, self).__init__()
        # 复制config
        config = copy.deepcopy(config)
        # 如果is_training为False，则将config.hidden_dropout_prob和config.attention_probs_dropout_prob设置为0.0
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0

        # 初始化一些变量
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        self.embedding_size = config.hidden_size
        self.token_type_ids = None

        # 定义last_idx变量
        self.last_idx = self.num_hidden_layers - 1
        # 计算output_embedding_shape的值
        output_embedding_shape = [-1, config.seq_length, self.embedding_size]

        # 创建一个nn.Embedding类型的bert_embedding_lookup
        self.bert_embedding_lookup = nn.Embedding(
            vocab_size=config.vocab_size,
            embedding_size=self.embedding_size,
            use_one_hot=use_one_hot_embeddings,
            embedding_table=TruncatedNormal(config.initializer_range))

        # 创建一个EmbeddingPostprocessor类型的bert_embedding_postprocessor
        self.bert_embedding_postprocessor = EmbeddingPostprocessor(
            embedding_size=self.embedding_size,
            embedding_shape=output_embedding_shape,
            use_relative_positions=config.use_relative_positions,
            use_token_type=True,
            token_type_vocab_size=config.type_vocab_size,
            use_one_hot_embeddings=use_one_hot_embeddings,
            initializer_range=0.02,
            max_position_embeddings=config.max_position_embeddings,
            dropout_prob=config.hidden_dropout_prob)

        # 创建一个BertTransformer类型的bert_encoder
        self.bert_encoder = BertTransformer(
            hidden_size=self.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_hidden_layers=self.num_hidden_layers,
            intermediate_size=config.intermediate_size,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            use_one_hot_embeddings=use_one_hot_embeddings,
            initializer_range=config.initializer_range,
            hidden_dropout_prob=config.hidden_dropout_prob,
            use_relative_positions=config.use_relative_positions,
            hidden_act=config.hidden_act,
            compute_type=config.compute_type,
            return_all_encoders=True)

        # 定义Cast操作
        self.cast = P.Cast()
        # 初始化dtype变量
        self.dtype = config.dtype
        # 定义SaturateCast操作
        self.cast_compute_type = SaturateCast(dst_type=config.compute_type)
        # 定义StridedSlice操作
        self.slice = P.StridedSlice()

        # 定义Squeeze操作
        self.squeeze_1 = P.Squeeze(axis=1)
        # 创建一个nn.Dense类型的dense
        self.dense = nn.Dense(self.hidden_size, self.hidden_size,
                              activation="tanh",
                              weight_init=TruncatedNormal(config.initializer_range)).to_float(config.compute_type)
        # 创建一个CreateAttentionMaskFromInputMask类型的_create_attention_mask_from_input_mask
        self._create_attention_mask_from_input_mask = CreateAttentionMaskFromInputMask(config)

    def construct(self, input_ids, token_type_ids, input_mask):
        """Bidirectional Encoder Representations from Transformers."""
        # embedding
        embedding_tables = self.bert_embedding_lookup.embedding_table
        word_embeddings = self.bert_embedding_lookup(input_ids)
        embedding_output = self.bert_embedding_postprocessor(token_type_ids,
                                                             word_embeddings)

        # attention mask [batch_size, seq_length, seq_length]
        attention_mask = self._create_attention_mask_from_input_mask(input_mask)

        # bert encoder
        encoder_output = self.bert_encoder(self.cast_compute_type(embedding_output),
                                           attention_mask)

        sequence_output = self.cast(encoder_output[self.last_idx], self.dtype)

        # pooler
        batch_size = P.Shape()(input_ids)[0]
        sequence_slice = self.slice(sequence_output,
                                    (0, 0, 0),
                                    (batch_size, 1, self.hidden_size),
                                    (1, 1, 1))
        first_token = self.squeeze_1(sequence_slice)
        pooled_output = self.dense(first_token)
        pooled_output = self.cast(pooled_output, self.dtype)

        return sequence_output, pooled_output, embedding_tables
