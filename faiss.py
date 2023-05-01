import faiss

# 创建一个IndexFlatL2索引，d是向量维度
index = faiss.IndexFlatL2(d)

# 将特征向量添加到索引中
index.add(features)

# 使用search方法来查找最相似的向量
# 使用索引中的向量来搜索与查询向量最相似的向量。其中，D是距离向量，I是索引向量，k是要返回的最相似向量的数量。
D, I = index.search(query, k)