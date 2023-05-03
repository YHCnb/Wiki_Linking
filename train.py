import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from tqdm import tqdm
from configs.bi_encoder_config import Bi_encoder_config
from my_bi_encoder import BiEncoderModule
import data_loader as data
import torch.nn.functional
from transformers.optimization import AdamW
import matplotlib.pyplot as plt

bi_encoder_config = Bi_encoder_config()
biEncoder = BiEncoderModule(bi_encoder_config)
tokenizer = BertTokenizer.from_pretrained(bi_encoder_config['bert_path'])

# 一个向量转换为3个数据，使其符合BERT输入
def to_bert_input(token_idx, null_idx):
    segment_idx = token_idx * 0
    mask = token_idx != null_idx
    token_idx = token_idx * mask.long()
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

        # 训练和验证时，比较一个batch中所有的candidate（以其它sample的label作为负样本）
        embedding_context = embedding_context.unsqueeze(1)  # batchsize x 1 x embed_size
        embedding_cands = embedding_cands.unsqueeze(2)  # batchsize x embed_size x 1
        scores = torch.bmm(embedding_context, embedding_cands)  # batchsize x 1 x 1
        scores = torch.squeeze(scores)
        return scores


def loss_function(scores):
    length = scores.size(0)
    target = torch.LongTensor(torch.arange(length))
    loss_ = torch.nn.functional.cross_entropy(scores, target, reduction="mean")
    #  mean：返回loss的平均值
    return loss_


# 绘图
def plot_loss(ep, t_loss, v_loss):
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(ep, t_loss, color='blue', label='Training loss')
    plt.plot(ep, v_loss, color='red', label='vali loss')
    plt.title("Training and vali loss")
    plt.legend()
    plt.show()

def train(num_epochs, learning_rate, batch_size):
    # 存储绘图数据
    t_loss = []
    v_loss = []
    ep = []  # epoch

    train_dataset = data.process('hansel-train.jsonl')
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size
    )
    train_iter = tqdm(train_dataloader, desc="Batch")

    vali_dataset = data.process('hansel-val.jsonl')
    vali_dataloader = DataLoader(
        vali_dataset, batch_size=batch_size
    )
    vali_iter = tqdm(vali_dataloader, desc="Batch")

    net = biEncoder
    optimizer = AdamW(net.parameters(), lr=learning_rate)  # 选用优化器AdamW

    for epoch in range(num_epochs):
        train_loss = 0.0
        # step为从start参数开始枚举的数（从0开始） batch为train_iter参数中的值
        for step, batch in enumerate(train_iter):
            context_vectors, entity_vectors, gold_id = batch
            optimizer.zero_grad()
            scores = score_candidate(context_vectors, entity_vectors)
            loss = loss_function(scores)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        with torch.no_grad():  # 不计算误差损失梯度
            vali_loss = 0.0
            for batch in vali_iter:
                context_vectors, entity_vectors, gold_id = batch
                scores = score_candidate(context_vectors, entity_vectors)
                loss = loss_function(scores)
                vali_loss += loss.item()

        train_loss = train_loss / len(train_iter)
        vali_loss = vali_loss / len(vali_iter)
        t_loss.append(train_loss)
        v_loss.append(vali_loss)
        ep.append(epoch)

    plot_loss(ep, t_loss, v_loss)
    return

num_epochs = 50
learning_rate = 0.01
batch_size = 32
train(num_epochs, learning_rate, batch_size)
