import re

# 示例数据
prompts = [
    "<s>[INST]<Img><ImgHere></Img>[vqa] the question [/INST]",
    "[INST]<Img><ImgHere></Img>[ref] another question [/INST]",
    "[INST]<Img><ImgHere></Img>[caption] yet another question [/INST]"
]

# 正则表达式
pattern = r'</Img>\[(.*?)\] (.*?)(?=\[/INST\])'

# 提取问题
questions = []
for prompt in prompts:
    match = re.search(pattern, prompt)
    if match:
        questions.append(match.group(2).strip())

# 输出提取结果
print(questions)
