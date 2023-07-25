import hashlib
import os
import urllib
import warnings
from typing import Any, Union, List
from pkg_resources import packaging

import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm

from .model import build_model
from .simple_tokenizer import SimpleTokenizer as _Tokenizer

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


if packaging.version.parse(torch.__version__) < packaging.version.parse("1.7.1"):
    warnings.warn("PyTorch version 1.7.1 or higher is recommended")

# 定义模块级变量 __all__，这个变量决定了使用 "from module import *" 语法时哪些名称会被导入
__all__ = ["available_models", "load", "tokenize"]

# 创建一个模块级对象 _tokenizer，这是 _Tokenizer 类的一个实例
_tokenizer = _Tokenizer()

# 定义了一个Python字典变量 _MODELS ，其中包含了一些预训练模型的名称和对应的下载链接。
_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
}


# 用于从给定的URL下载文件并计算SHA256哈希值以验证下载的文件是否完整和正确。
def _download(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError("Model has been downloaded but the SHA256 checksum does not not match")

    return download_target


# 该函数接受一个PIL Image对象作为输入，并将其转换为RGB格式。
def _convert_image_to_rgb(image):
    return image.convert("RGB")


# 函数接受一个整数 n_px 作为输入，并返回一个 torchvision.transforms.Compose 对象，该对象包含一系列图像变换操作。
def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC), # 将输入图像的大小调整为 (n_px, n_px)，并使用 BICUBIC 插值进行插值。
        CenterCrop(n_px), # 将输入图像从中心位置裁剪为 (n_px, n_px) 的大小。
        _convert_image_to_rgb, # 将输入图像转换为RGB格式。
        ToTensor(), # 将输入图像转换为RGB格式。

        # 对张量进行标准化，将图像的像素值减去均值 [0.48145466, 0.4578275, 0.40821073]，然后除以标准差 [0.26862954, 0.26130258, 0.27577711]。
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def available_models() -> List[str]:
    """返回可用的CLIP模型的名称"""
    return list(_MODELS.keys())


# 该函数用于加载一个CLIP模型，并返回一个模型对象和一个预处理函数，该函数将PIL Image对象转换为模型输入的张量。
def load(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", jit: bool = False, download_root: str = None):
    """加载一个CLIP模型

    参数
    ----------
    name : str
        available_models()列出的模型名称,或者包含state_dict的模型checkpoint的路径

    device : Union[str, torch.device]
        加载模型的设备

    jit : bool
        是否加载优化后的JIT模型或者更可改动的非JIT模型(默认)

    download_root: str
        下载模型文件的路径; 默认使用 "~/.cache/clip"

    Returns
    -------
    model : torch.nn.Module
        CLIP模型

    preprocess : Callable[[PIL.Image], torch.Tensor]
        一个torchvision transform,将PIL图像转换成模型输入的tensor
    """

    # 如果指定的模型名称在 _MODELS 字典中，则该函数将下载该模型的预训练权重文件。
    # 如果指定的模型名称不在 _MODELS 字典中且指定的路径不是一个文件，则该函数会引发 RuntimeError 异常。
    if name in _MODELS:
        model_path = _download(_MODELS[name], download_root or os.path.expanduser("~/.cache/clip"))
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    with open(model_path, 'rb') as opened_file:
        try:
            # loading JIT archive
            model = torch.jit.load(opened_file, map_location=device if jit else "cpu").eval()
            state_dict = None
        except RuntimeError:
            # loading saved state dict
            if jit:
                warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
                jit = False
            state_dict = torch.load(opened_file, map_location="cpu")

    if not jit:
        model = build_model(state_dict or model.state_dict()).to(device)
        if str(device) == "cpu":
            model.float()
        return model, _transform(model.visual.input_resolution)

    # 对设备名称进行修补
    device_holder = torch.jit.trace(lambda: torch.ones([]).to(torch.device(device)), example_inputs=[])
    device_node = [n for n in device_holder.graph.findAllNodes("prim::Constant") if "Device" in repr(n)][-1]

# 这个方法的目的是以多态的方式获取节点属性的值。
# 使用 node 对象的 kindOf 方法来确定属性的类型，然后调用相应的 getter 方法来检索属性的值。
    def _node_get(node: torch._C.Node, key: str):
        """获取节点的多态属性,属性类型可以不同。
        
        来源 https://github.com/pytorch/pytorch/pull/82628
        """
        sel = node.kindOf(key)
        return getattr(node, sel)(key)

# 接受一个参数 module，并尝试在该模块中的所有图中查找 prim::Constant 节点，
# 该函数的目的是将模块中的所有常量张量设备属性设置为 device_node，以确保它们在正确的设备上运行。
    def patch_device(module):
        try:
            graphs = [module.graph] if hasattr(module, "graph") else []
        except RuntimeError:
            graphs = []

        if hasattr(module, "forward1"):
            graphs.append(module.forward1.graph)

        for graph in graphs:
            for node in graph.findAllNodes("prim::Constant"):
                if "value" in node.attributeNames() and str(_node_get(node, "value")).startswith("cuda"):
                    node.copyAttributes(device_node)

# 将 patch_device 函数应用于 model 所有的子模块。
    model.apply(patch_device)

# 将 patch_device 应用于模型中的 encode_image 和 encode_text 方法。
    patch_device(model.encode_image)
    patch_device(model.encode_text)

    # 在CPU上将dtype修补为float32
    if str(device) == "cpu":

        # 使用 torch.jit.trace 函数创建一个只返回 torch.ones([]).float() 的 float_holder
        float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])

        # 从其中获取 aten::to 节点的第二个输入。
        float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]

        # 使用 _node_get 方法获取该节点的值为 5 的输入，并将该输入的节点的设备属性和数据类型设置为 float_node。
        float_node = float_input.node()

# 确保所有输入为值为 5 的常量整数张量的节点都被正确地设置为 CPU 设备和 float 类型。
        def patch_float(module):
            try:
                graphs = [module.graph] if hasattr(module, "graph") else []
            except RuntimeError:
                graphs = []

            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)

            for graph in graphs:
                for node in graph.findAllNodes("aten::to"):
                    inputs = list(node.inputs())
                    for i in [1, 2]:  # dtype可以是aten::to()的第二个或第三个参数
                        if _node_get(inputs[i].node(), "value") == 5:
                            inputs[i].node().copyAttributes(float_node)

# 将 patch_float 函数应用于 model 所有的子模块。
        model.apply(patch_float)

# 将 patch_float 应用于模型中的 encode_image 和 encode_text 方法。
        patch_float(model.encode_image)
        patch_float(model.encode_text)

# 将模型中所有参数和缓存的数据类型转换为 float 类型。
        model.float()

    return model, _transform(model.input_resolution.item())


def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Union[torch.IntTensor, torch.LongTensor]:
    """
    返回给定输入字符串的标记化表示

    参数
    ----------
    texts : Union[str, List[str]]
        输入字符串或输入字符串列表进行标记化

    context_length : int
        使用的上下文长度;所有CLIP模型使用77作为上下文长度

    truncate: bool
        是否在文本编码长度超过上下文长度时对其进行截断

    Returns
    -------
    二维tensor,包含结果标记,形状 = [输入字符串数量, 上下文长度]。
    当torch版本<1.8.0时,我们返回LongTensor,因为旧的index_select需要索引是long类型。
    """

    # 检查 texts 是否是字符串类型。
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"] # 将变量 sot_token 赋值为 _tokenizer 对象的 encoder 属性中 <|startoftext|> 标记的索引值。
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

# 遍历 all_tokens 列表中的所有文本序列，并将它们转换为 PyTorch 张量 result。
    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result
