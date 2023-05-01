class Bi_encoder_config:
    def __init__(self):
        self.config = {
            'bert_path': "bert-base-chinese",  # bert-base地址
            'out_dim': 10000,  # CLS线性头输出的维度
            "ENT_START_TAG": "[unused0]",  # 3个特殊token的标记
            "ENT_END_TAG": "[unused1]",
            "ENT_TITLE_TAG": "[unused2]",
        }

    def __getitem__(self, key):
        return self.config[key]


config = Bi_encoder_config()
print(config['bert_path'])
