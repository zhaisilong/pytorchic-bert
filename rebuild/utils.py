import random

def truncate_tokens_pair(tokens_a, tokens_b, max_len):
    """句子 a 和 b 的总长不超过 max_len，哪个长截掉哪个
    """
    while True:
        if len(tokens_a) + len(tokens_b) <= max_len:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def get_random_word(vocab_words):
    """ABCDEFG: 随机 -> B
    """
    i = random.randint(0, len(vocab_words)-1)  # 包括最后一个
    return vocab_words[i]