import gzip
import html
import os
from functools import lru_cache

import ftfy
import regex as re


# 这是一个装饰器函数 @lru_cache()，它将函数 default_bpe() 装饰为带有缓存的函数。这意味着函数的输出结果会被缓存，避免重复计算，提高代码的运行效率。
@lru_cache()

# 返回一个默认的 BPE（Byte Pair Encoding）词汇表文件的路径，这个词汇表文件通常用于自然语言处理中的文本编码和解码。
def default_bpe():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe_simple_vocab_16e6.txt.gz")


@lru_cache()
def bytes_to_unicode():
    """
    返回utf-8字节列表和相应的unicode字符串列表。
    可逆的bpe码作用于unicode字符串上。
    这意味着,如果你想避免UNK,你的词汇表中需要大量的unicode字符。
    当你处理大约100亿个标记的数据集时,你需要大约5K个字符来获得足够的覆盖范围。
    这占你正常的3.2万个bpe词汇表的较大比例。
    为了避免这一点,我们需要utf-8字节到unicode字符串的查找表。
    并避免映射到bpe码无法处理的空白/控制字符。
    """

    # 定义了一个包含可打印 ASCII 字符和部分 Latin-1 字符的 UTF-8 字节列表 bs
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0

    # 将不在 bs 中的字节添加到 bs 和 cs 列表中，并将其映射到从 256 开始的整数，以便与 ASCII 字符区分开来。
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1

    # 将 cs 列表中的整数转换为相应的 Unicode 字符，构建 UTF-8 字节到 Unicode 字符的映射表，并将其作为字典返回。
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """返回在一个词中出现的符号对集合。
    词被表示为一个符号元组(符号是可变长度的字符串)。
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


# 用于对文本进行基本的清洗操作，包括去除空白、HTML 实体转义、Unicode 修复等。
def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()

# 用于去除文本中的多余空白字符，并将所有空白字符替换为单个空格。
def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


# 是 OpenAI 的 CLIP 模型中用于文本编码的 BPE（Byte Pair Encoding）分词器的简单实现。
class SimpleTokenizer(object):

    # 将其中的 BPE 合并操作解析为元组，并构建词汇表，包含所有的 Unicode 字符以及对应的 BPE 码表中的编码。
    def __init__(self, bpe_path: str = default_bpe()):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
        merges = merges[1:49152-256-2+1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v+'</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}
        self.pat = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)

    # 将单词拆分为 BPE 编码，并返回 BPE 编码序列。
    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + ( token[-1] + '</w>',)
        pairs = get_pairs(word)

        if not pairs:
            return token+'</w>'

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    # 该函数用于将文本转换为 BPE 编码序列。
    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    # 该函数用于将 BPE 编码序列转换为文本。 
    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
        return text
