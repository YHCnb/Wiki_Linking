import re

import opencc
from wikimapper import WikiMapper

# 获取wikidata_id
mapper = WikiMapper("D:\Pycharm\myCodes\EntityLinking\zhwiki\data\index_zhwiki-20230320.db")
converter = opencc.OpenCC('t2s.json')  # 从繁体到简体的转换器

with open('D:\Pycharm\myCodes\EntityLinking\zhwiki\zh-wiki_new', encoding='utf-8') as f:
    content = f.read()
    content = content.replace('\n', '')  # 去除所有换行符
    content = re.sub('{"id"', '\n{"id"', content)  # 在每个json对象之前加上换行符

with open('D:\Pycharm\myCodes\EntityLinking\zhwiki\zh-wiki_final', 'w', encoding='utf-8') as f:
    for line in content.split('\n'):  # 按换行符分割每个json对象
        if line:  # 跳过空行
            line = line.strip()
            start = line.find('"title":') + 9
            end = line.find('"', start+1)
            title = line[start+1:end]
            wikidata_id = mapper.title_to_id(title)
            # print(title)
            line = line[0] + '"wikidata_id": "'+str(wikidata_id)+'", ' + line[1:]
            line = converter.convert(line)
            f.write(line + '\n')
            # print(line)



# # 如果行中包含"text":
# if '"text":' in line:
#     # 去掉行首和行尾的空白字符
#     line = line.strip()
#     # 找到"text":后面的文本的开始和结束位置
#     start = line.find('"text":') + 8
#     end = line.rfind('"')
#     # 提取出"text":后面的文本
#     text = line[start + 1:end]
#     print(text)
#     print(json.dumps(text))
#     # 把文本中的双引号转义，即在前面加上反斜杠
#     # text = text.replace('"', '\\"')
#     # 把转义后的文本替换原来的文本
#     line = line[:start] + json.dumps(text) + line[end + 1:]
#     # 在行尾加上换行符
#     line = line
# d = json.loads(line)  # 把字符串转为json对象
# text = d["text"]
# if len(text) > 400:
#     text = text[:400]
# d["text"] = text
# f.write(json.dumps(d) + '\n')  # 把json对象转回字符串并写入文件，每个对象占一行
