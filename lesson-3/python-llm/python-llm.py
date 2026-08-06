from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import tiktoken


def learn_nltk():
    

    # 参考句（分词后的列表）
    reference = [['今天', '阳光', '不错']]
    candidate = ['今天', "阳光",'很好']

    smooth = SmoothingFunction().method4
    bleu = sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
    print(f"BLEU-2: {bleu:.4f}")

def learn_bpe():
    enc = tiktoken.get_encoding("cl100k_base")  # GPT-4 使用的编码
    tokens = enc.encode("unhappy")
    print(f"Token IDs: {tokens}")
    print(f"Tokens: {[enc.decode([t]) for t in tokens]}")
    # 输出示例：['un', 'happy']
def main():
    learn_nltk()
    learn_bpe()
if __name__ == "__main__":
    main()
