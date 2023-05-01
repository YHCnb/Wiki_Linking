import re
import opencc


# 用于处理非法的Unicode字符
def remove_surrogates(text):
    # 代理对的范围是\uD800-\uDFFF
    pattern = re.compile(r"[\uD800-\uDFFF]")
    # 用空字符串替换掉匹配到的代理对
    return pattern.sub("", text)

# 将文件中的unicode编码的中文转为utf-8中文
with open("zh-wiki", "r", encoding="utf-8") as f:
    lines = f.readlines()
# 创建一个新的文件，写入转换后的文.本
with open("zh-wiki_new", "w", encoding="utf-8") as f:
    for line in lines:
        # 把每一行的Unicode编码转换成中文
        line = line.encode("utf-8").decode("unicode-escape")

        line = remove_surrogates(line)
        # 把每一行的文本转换成简体中文
        # line = converter.convert(line)
        # 把转换后的文本写入新文件
        f.write(line)
