file_path = "/home/zhuyangyang/project/test/src/language/python/llm/input.txt"

with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

print("length of dataset in characters: ", len(text))
print(text[:1000])

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)
