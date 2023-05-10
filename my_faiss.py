import numpy as np
import faiss
import torch

d = 1                                           # 向量维度
nb = 2                                      # index向量库的数据量
nq = 1                                      # 待检索query的数目
np.random.seed(123)
xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 1000.                # index向量库的向量
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000.
print(xb)
print(xq)

index = faiss.IndexFlatL2(d)
print(index.is_trained)         # 输出为True，代表该类index不需要训练，只需要add向量进去即可
index.add(xb)                   # 将向量库中的向量加入到index中
print(index.ntotal)

k = 4                     # topK的K值
D, I = index.search(xq, k)# xq为待检索向量，返回的I为每个待检索query最相似TopK的索引list，D为其对应的距离
print(I)
print(D)