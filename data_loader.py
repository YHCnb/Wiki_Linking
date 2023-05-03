import torch
from transformers import BertTokenizer
import json
from tqdm import tqdm
from configs.bi_encoder_config import Bi_encoder_config
from my_bi_encoder import BiEncoderModule
from torch.utils.data import TensorDataset

bi_encoder_config = Bi_encoder_config()
biEncoder = BiEncoderModule(bi_encoder_config)
tokenizer = BertTokenizer.from_pretrained(bi_encoder_config['bert_path'])


# 获得表示entity的向量
def get_entity_vecs(title, text):
    max_length = 32
    title_tokens = tokenizer.tokenize(title)
    text_tokens = tokenizer.tokenize(text)
    entity_tokens = title_tokens + ["[unused02]"] + text_tokens
    entity_tokens = entity_tokens[0: max_length - 2]
    entity_tokens = ["[CLS]"] + entity_tokens + ["[SEP]"]  # 加头尾标识符
    input_ids = tokenizer.convert_tokens_to_ids(entity_tokens)
    padding = [0] * (max_length - len(input_ids))
    input_ids += padding
    assert len(input_ids) == max_length
    return input_ids


# wiki库转dictionary
def wiki_preprocess():
    dict = {}
    with open('zh-wiki_final', encoding='utf-8') as f:
        cnt = 0
        while True:
            text = f.readline()
            # 取wikidata_id
            start = text.find("\"wikidata_id\": \"")
            length = len("\"wikidata_id\": \"")
            end = text.find("\"", start+length)
            wikidata_id = text[start+length:end]
            # 取title
            start = text.find("\"title\": \"")
            length = len("\"title\": \"")
            end = text.find("\"", start + length)
            title = text[start + length:end]
            # 取text
            start = text.find("\"text\": \"")
            length = len("\"text\": \"")
            text = text[start + length:len(text)-3]
            entity_vectors = get_entity_vecs(title, text)
            dict[wikidata_id] = entity_vectors
            cnt = cnt+1
            if(cnt==5):
                # print(dict)
                break
    return dict


# context预处理
def context_preprocess(datapath):
    dict = wiki_preprocess()
    # print(dict)
    processed_context = []
    labels = []
    label_vectors = []
    with open(datapath, encoding='utf-8') as (f):
        flag = 0
        for sample in tqdm(f):
            x = json.loads(sample)
            mention_tokens = tokenizer.tokenize(x['mention'])
            context_tokens = tokenizer.tokenize(x['text'])
            mention_tokens = ["[unused0]"] + mention_tokens + ["[unused1]"]  # 加mention头尾标识符
            max_length = 32  # context 最大长度
            if( len(mention_tokens) > 32):
                mention_tokens = mention_tokens[0:32]
            max_left = (max_length - len(mention_tokens))//2 -1   # 左侧最大长度
            max_right = max_length - 2 - max_left - len(mention_tokens)   # 右侧最大长度
            left_length = x['start']  # 文本左侧实际长度
            right_length = len(context_tokens) - x['end']-1  # 文本右侧实际长度
            left_quote = min(left_length, max_left)  # 实际引用的文本左侧长度
            right_quote = min(right_length, max_right)  # 实际引用的文本右侧长度

            context_tokens = context_tokens[x['start']-left_quote:x['start']] + mention_tokens + \
                             context_tokens[x['end']+1:x['end']+right_quote+1]
            context_tokens = ["[CLS]"] + context_tokens + ["[SEP]"]  # 加头尾标识符
            input_ids = tokenizer.convert_tokens_to_ids(context_tokens)
            padding = [0] * (max_length - len(input_ids))  # 补0
            input_ids += padding
            assert len(input_ids) == max_length  # 确认input_ids 长度为max_length
            label_vecs = dict[x['gold_id']]
            processed_context.append(input_ids)
            labels.append(x['gold_id'])
            label_vectors.append(label_vecs)

            if(flag==0):
                print(context_tokens)
                print(input_ids)
                flag = 1

    processed_context = torch.Tensor(processed_context)
    print(processed_context.shape)
    labels = torch.Tensor(labels)
    print(labels.shape)
    label_vectors = torch.Tensor(label_vectors)
    print(label_vectors.shape)

    return processed_context, labels, label_vectors


def process(datapath):
    context_vecs, label_idx, label_vecs = context_preprocess(datapath)
    dataset = TensorDataset(context_vecs, label_vecs, label_idx)
    return dataset

process('hansel-val.jsonl')