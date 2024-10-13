from app.utils.metric import cosine_similarity
from app.utils.submit import get_sentence_embedding


def test():
    sentence1 = "Вы забыли поставить префикс f перед строкой, переданной функции print()."
    sentence2 = "Вы забыли поставить префикс f перед строкой."

    embedding1 = get_sentence_embedding(sentence1)
    embedding2 = get_sentence_embedding(sentence2)

    cos_sim = cosine_similarity(embedding1.unsqueeze(0), embedding2.unsqueeze(0))
    print(f"Cosine Similarity: {cos_sim.item():.4f}")


if __name__ == "__main__":
    test()
