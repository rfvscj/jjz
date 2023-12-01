import re
from textrank4zh import TextRank4Keyword, TextRank4Sentence
import json
from tqdm import tqdm
import torch

# 中文分句
def cut_sent(para):
    para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    return para.split("\n")



def get_summary(text, k=10, max_len=512):
    tr4s = TextRank4Sentence()
    tr4s.analyze(text, lower=True, source='all_filters')
    
    summary = [item for item in tr4s.get_key_sentences(num=k)]
    summary = sorted(summary, key=lambda x: x.index)
    summary = ' '.join([item.sentence for item in summary])[:max_len]
    return summary


def num_clamp(x, low, high):
    x = min(x, high)
    x = max(x, low)
    return x


def batch_acc(logits: torch.Tensor, label: torch.Tensor):
    # batch, 39   batch
    preds = torch.argmax(logits, dim=1)
    # 计算得分，暂未精确考虑特殊情况
    score_0 = 1 - torch.mean(torch.abs(torch.log(1 + preds) - torch.log(1 + label))).item()
    
    acc_num = torch.sum(preds == label).item()
    exact_acc = acc_num / logits.shape[0]
    
    return exact_acc, score_0

def batch_div(predictions: torch.Tensor, targets: torch.Tensor):
    return torch.sqrt(torch.mean((predictions - targets)**2))
    
    