class Prompt_config:
    def __init__(self):
        self.NUM_TOKENS = 10  # 软提示数量
        self.DROPOUT = 0.2  # 初始化dropout概率
        self.PROJECT = 768  # prompt_dim
        self.DEEP = True  # 是否采用DEEP策略
