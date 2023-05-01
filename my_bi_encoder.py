import torch
from transformers import BertTokenizer

from PromptedBert import PromptedBert
from configs.bi_encoder_config import Bi_encoder_config
from configs.prompt_config import Prompt_config


class BiEncoderModule(torch.nn.Module):
    def __init__(self, config):
        super(BiEncoderModule, self).__init__()
        prompt_config = Prompt_config()
        self.context_encoder = PromptedBert(
            prompt_config,
            config['bert_path'],
            out_dim=config['out_dim']
        )
        self.cand_encoder = PromptedBert(
            prompt_config,
            config['bert_path'],
            out_dim=config['out_dim']
        )
        self.config = prompt_config  # 保存prompt_config

    def forward(
            self,
            token_idx_ctxt,
            segment_idx_ctxt,
            mask_ctxt,
            token_idx_cands,
            segment_idx_cands,
            mask_cands,
    ):
        embedding_ctxt = None
        if token_idx_ctxt is not None:
            embedding_ctxt = self.context_encoder(
                token_idx_ctxt, segment_idx_ctxt, mask_ctxt
            )
        embedding_cands = None
        if token_idx_cands is not None:
            embedding_cands = self.cand_encoder(
                token_idx_cands, segment_idx_cands, mask_cands
            )
        return embedding_ctxt, embedding_cands


if __name__ == "__main__":
    bi_enconder_config = Bi_encoder_config()
    biEncoder = BiEncoderModule(bi_enconder_config)
    tokenizer = BertTokenizer.from_pretrained(bi_enconder_config['bert_path'])

    inputs1 = tokenizer("今天天气真好，适合机器学习", return_tensors="pt")
    inputs2 = tokenizer("我真爱机器学习", return_tensors="pt")

    embedding_context, _ = biEncoder(
                inputs1['input_ids'], inputs1['token_type_ids'], inputs1['attention_mask'], None, None, None
            )
    print(embedding_context[0])
    print(embedding_context[0].shape)
    _, embedding_cands = biEncoder(
                None, None, None, inputs2['input_ids'], inputs2['token_type_ids'], inputs2['attention_mask']
            )
    print(embedding_cands[0])
    print(embedding_cands[0].shape)
