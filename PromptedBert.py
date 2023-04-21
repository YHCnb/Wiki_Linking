import math
import torch
from torch import nn, mul
from torch.nn import Dropout
from transformers import BertModel, BertConfig, BertTokenizer
from configs.prompt_config import Prompt_config


class PromptedBert(nn.Module):
    def __init__(self, prompt_config, bert_path, num_classes):
        super(PromptedBert, self).__init__()
        config = BertConfig.from_pretrained(bert_path)
        num_tokens = prompt_config.NUM_TOKENS
        self.bert_config = config
        self.num_tokens = num_tokens
        self.prompt_config = prompt_config
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.prompt_dropout = Dropout(prompt_config.DROPOUT)  # dropout防止过拟合

        # 用bert-base初始化
        model = BertModel.from_pretrained(bert_path, config=config)
        self.encoder = model.encoder
        self.embeddings = model.get_input_embeddings()
        self.head = nn.Linear(config.hidden_size, num_classes)

        # 初始化 prompt线性层
        prompt_dim = prompt_config.PROJECT
        self.prompt_proj = nn.Linear(
            prompt_dim, config.hidden_size)
        nn.init.kaiming_normal_(self.prompt_proj.weight, a=0, mode='fan_out')

        # 用Xavier方法初始化 prompt_embeddings,
        val = math.sqrt(6. / (num_tokens + prompt_dim))
        self.prompt_embeddings = nn.Parameter(torch.zeros(
            1, num_tokens, prompt_dim))
        nn.init.uniform_(self.prompt_embeddings.data, -val, val)
        # 如果采用deep策略,额外创造多层prompt_embeddings
        if prompt_config.DEEP:
            num_layers = config.num_hidden_layers - 1
            self.deep_prompt_embeddings = nn.Parameter(torch.zeros(
                num_layers, num_tokens, prompt_dim))
            nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)

    def incorporate_prompt(self, input_ids):
        # 将input_embeddings和prompt_embeddings组合，prompt_embeddings放在[CLS]后面
        batch_size = input_ids.shape[0]
        x = self.embeddings(input_ids)
        x = torch.cat((
            x[:, :1, :],
            self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(batch_size, -1, -1)),
            x[:, 1:, :]
        ), dim=1)
        return x

    def train(self, mode=True):
        # 设置模型状态
        if mode:
            # training:保持bert主体参数不变
            self.encoder.eval()
            self.embeddings.eval()
            self.head.train()
            self.prompt_dropout.train()
            self.prompt_proj.train()
            self.prompt_dropout.train()
        else:
            # eval:所有参数保持不变
            for module in self.children():
                module.train(mode)

    def forward_deep_prompt(self, embedding_output):
        # 获得deep模式下的输出
        hidden_states = None
        batch_size = embedding_output.shape[0]
        num_layers = self.bert_config.num_hidden_layers

        for i in range(num_layers):
            if i == 0:
                hidden_states = self.encoder.layer[i](embedding_output)[0]
            else:
                if i <= self.deep_prompt_embeddings.shape[0]:
                    deep_prompt_emb = self.prompt_dropout(self.prompt_proj(
                        self.deep_prompt_embeddings[i - 1]).expand(batch_size, -1, -1))

                    hidden_states = torch.cat((
                        hidden_states[:, :1, :],
                        deep_prompt_emb,
                        hidden_states[:, (1 + self.num_tokens):, :]
                    ), dim=1)

                hidden_states = self.encoder.layer[i](hidden_states)[0]

        encoded = self.layer_norm(hidden_states)
        return encoded

    def forward(self, inputs):
        embedding_output = self.incorporate_prompt(inputs['input_ids'])
        # attention_mask = torch.cat([torch.full((1, self.num_tokens), 1), inputs['attention_mask']], 1)
        # token_type_ids = torch.cat((
        #     inputs['token_type_ids'][:, :1],
        #     torch.full((1, self.num_tokens), 0),
        #     inputs['token_type_ids'][:, 1:]
        # ), dim=1)

        if self.prompt_config.DEEP:
            encoded = self.forward_deep_prompt(
                embedding_output)
        else:
            encoded = self.encoder(embedding_output)
        # 选择第一个结点的输出，通过线性头得到结果（由于目标是提取特征，则不进行归一化）
        encoded = encoded[:, 0]
        logits = self.head(encoded)
        return logits


# 下载bert-base-ch
# snapshot_download(repo_id="bert-base-chinese", cache_dir="D:\Pycharm\myCodes\EntityLinking",
#                   ignore_patterns=["*.h5", "*.ot", "*.msgpack"])
bert_path = "bert-base-chinese"

tokenizer = BertTokenizer.from_pretrained(bert_path)
prompt_config = Prompt_config()
promptedBert = PromptedBert(prompt_config, bert_path, num_classes=10000)

inputs1 = tokenizer("今天天气真好，适合机器学习", return_tensors="pt")
inputs2 = tokenizer("我真爱机器学习", return_tensors="pt")
encoded1 = promptedBert(inputs1)
print(encoded1[0])
print(encoded1[0].shape)
encoded2 = promptedBert(inputs2)
print(encoded2[0])
print(encoded2[0].shape)
