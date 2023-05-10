import json
import random

# 用于从训练集分离出10000条记录作为测试集
filename = 'data/hansel-train.jsonl'
n = 10000  # 取10000条作为测试集

with open(filename, encoding='utf-8') as f:
    lines = f.readlines()

samples = random.sample(lines, n)

test_set = set()
goldid_set = set()
with open('data/test.jsonl', 'w', encoding='utf-8') as test_file:
    for sample in samples:
        data = json.loads(sample)
        # if data['gold_id'] not in goldid_set:
        test_file.write(sample)
        test_set.add(data['id'])
        goldid_set.add(data['gold_id'])

with open('data/train.jsonl', 'w', encoding='utf-8') as train_file:
    count = 0
    for i, line in enumerate(lines):
        data = json.loads(line)
        if (data['id'] not in test_set):  # and (data['gold_id'] in goldid_set):  # id不在分离出的记录，那就重新写进训练集
            train_file.write(line)
            # goldid_set.remove(data['gold_id'])
            # count += 1
            # if count == n:
            #     break
