# coding=utf-8
import jieba
import re
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
import io
import json
import pickle
from utils import cut_sent
sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')

def get_stop_words():
    # 加载停用词表
    with open('data/hit_stopwords.txt', 'r', encoding='utf-8') as f:
        stop_words = [line.strip() for line in f]
    return stop_words


def fit(corpus):
    formatted_corpus = []
    # jieba分词
    for document in corpus:
        words = jieba.cut(document, cut_all=False, HMM=False)
        words = [word for word in words if word not in stop_words and re.match(r'^[\s\u4e00-\u9fa5]*$', word) and word!=' ']
        str1 = ' '.join(words)
        formatted_corpus.append(str1)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(formatted_corpus)
    print(X.shape)
    weights = X.toarray()
    with open("tf_idf_model.pkl", 'wb') as fw:
        pickle.dump(vectorizer, fw)
    return X, vectorizer
    



def get_keywords(text):
    # 使用 jieba 对文本进行分词

    words = jieba.cut(text, cut_all=False, HMM=False)
    # 过滤掉标点符号和停用词
    words = [word for word in words if word not in stop_words and re.match(r'^[\s\u4e00-\u9fa5]*$', word) and word!=' ']
    # print(words)
    # 使用 TfidfVectorizer 计算每个词的权重\
    str1 = ''
    for i in words: str1 = str1 + i + ' '
    word1 = [str1]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(word1)
    # 得到每个词的权重
    weights = X.toarray()
    # 将每个词和对应的权重放入字典中
    keywords = {vectorizer.get_feature_names_out()[i]: weights[0][i] for i in range(len(weights[0]))}
    # 按照权重从大到小排序
    keywords = sorted(keywords.items(), key=lambda x: x[1], reverse=True)
    return keywords

def get_summary(text, num_sentences):

    # 对文本进行句子切分
    sentences = re.split(r'[。！？，]', text)
    # 获取文本中的关键词
    keywords = get_keywords(text)
    # 对每个句子计算其中包含的重要词的权重总和
    sentence_scores = []
    for sentence in sentences:
        score = 0
        for keyword, weight in keywords:
            if keyword in sentence:
                score += weight
        sentence_scores.append(score)

  # 将权重最大的几个句子连接起来，得到文本的摘要
#   首先调用函数sentence_scores.index(max(sentence_scores))获取权重最大的句子的索引
#   然后使用这个索引从句子列表sentences中取出这个句子并将它加入到结果列表summary中
#   最后将权重最大的句子的权重设为-1,以避免在之后的迭代中再次选中。
    summary = []
    for i in range(num_sentences):
        summary.append(sentences[sentence_scores.index(max(sentence_scores))])
        sentence_scores[sentence_scores.index(max(sentence_scores))] = -1
    return '。'.join(summary)

if __name__ == '__main__':
    # 加载停用词表
    stop_words = get_stop_words()
    # 读取文本
    corpus = []
    with open('data/train_2.json', 'r', encoding='utf8') as f:
        train_list = json.load(f)
        for item in tqdm(train_list):
            fact1 = item['fact1']
            fact2 = item['fact2']
            corpus += [fact1, fact2]
    X, vectorizer = fit(corpus)

    # 生成文本摘要
    # summary = get_summary(corpus, 3)
    
    # print(summary)
