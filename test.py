import os

import torch
import pickle
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from tqdm import tqdm
from configs.bi_encoder_config import Bi_encoder_config
from my_bi_encoder import BiEncoderModule
import data_loader as data

bi_encoder_config = Bi_encoder_config()
biEncoder = BiEncoderModule(bi_encoder_config)
if os.path.exists('biEncoder.pt'):
    biEncoder.load_state_dict(torch.load('biEncoder.pt'))
    print("load from existed model")
if torch.cuda.is_available():
    biEncoder.cuda()
    device = torch.device("cuda:0")
else: device = torch.device("cpu")
print(device)

dict = data.wiki_preprocess()


# 一个向量转换为3个数据，使其符合BERT输入
def to_bert_input(token_idx, null_idx):
    token_idx = token_idx.long()
    segment_idx = token_idx * 0
    mask = token_idx * 1
    return token_idx, segment_idx, mask


def encode_context(context):
    token_idx_context, segment_idx_context, mask_context = to_bert_input(context, 0)
    embedding_context, _ = biEncoder(
        token_idx_context, segment_idx_context, mask_context, None, None, None
    )
    return embedding_context


def encode_candidate(cands):
    token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(cands, 0)
    _, embedding_cands = biEncoder(
        None, None, None, token_idx_cands, segment_idx_cands, mask_cands
    )
    return embedding_cands


def load_candidate():
    entity_embedding_list = []  # BERT输出的entity_embedding
    entity_id_list = []  # 去Q的entity_wiki_id
    list = []
    with torch.no_grad():
        for key in dict:
            cand_vecs = dict[key]
            list.append(cand_vecs)
            entity_id_list.append(int(key[1:]))
            if len(list) == batch_size:
                embedding_cands = encode_candidate(torch.tensor(list).to(device))
                torch.cuda.empty_cache()
                list = []
                for i in range(batch_size):
                    entity_embedding_list.append(embedding_cands[i])
    print('finish load_candidate')
    return entity_embedding_list, entity_id_list


def score_candidate(
        text_vecs,
        cand_vecs,
        cand_encs=None,  # 已编码的candidate train和vali为none test有
):
    embedding_context = encode_context(text_vecs)

    # 测试时，cand_encs已知，无需转换，直接返回scores
    if cand_encs is not None:
        return embedding_context.mm(cand_encs.t())
    embedding_cands = encode_candidate(cand_vecs)
    scores = embedding_context.mm(embedding_cands.t())
    return scores


def test(batch_size):
    test_total = 0
    test_correct = 0
    test_dataset = data.process('data/test.jsonl', dict)
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size
    )
    test_iter = tqdm(test_dataloader, desc="Batch")
    with torch.no_grad():
        print("load_candidate")
        entity_embedding_list, entity_id_list = load_candidate()
        for step, batch in enumerate(test_iter):
            context_vectors, entity_vectors, label_id = batch
            scores = score_candidate(context_vectors, None, cand_encs=entity_embedding_list)
            values, indices = scores.topk(1)  # indices 为score最大的cand_encs的索引
            for i in range(context_vectors.size(0)):
                index = indices[i]
                prediction_id = entity_id_list[index]
                if prediction_id == label_id[i]:
                    test_correct += 1
                test_total += 1
    test_accuracy = test_correct / test_total
    print('test accuracy:  {:.5f}'.format(test_accuracy))
    return


if __name__ == "__main__":
    batch_size = 32
    test(batch_size)
