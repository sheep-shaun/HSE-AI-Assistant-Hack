from typing import Callable

import pandas as pd
import torch
from transformers import BertModel, BertTokenizer

print("Loading models...", end="")
model_name = "DeepPavlov/rubert-base-cased-sentence"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
print("OK")


def get_sentence_embedding(sentence: str) -> torch.Tensor:
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].squeeze()
    return embedding


def string2embedding(string: str) -> torch.Tensor:
    return torch.Tensor([float(i) for i in string.split()])


def embedding2string(embedding: torch.Tensor) -> str:
    return " ".join([str(i) for i in embedding.tolist()])


def generate_submit(test_solutions_path: str, predict_func: Callable, save_path: str, use_tqdm: bool = True) -> None:
    test_solutions = pd.read_excel(test_solutions_path)
    bar = range(len(test_solutions))
    if use_tqdm:
        import tqdm

        bar = tqdm.tqdm(bar, desc="Predicting")

    submit_df = pd.DataFrame(columns=["solution_id", "author_comment", "author_comment_embedding"])
    for i in bar:
        idx = test_solutions.index[i]
        solution_row = test_solutions.iloc[i]

        text = predict_func(solution_row)  # here you can do absolute whatever you want

        embedding = embedding2string(get_sentence_embedding(text))
        submit_df.loc[i] = [idx, text, embedding]
    submit_df.to_csv(save_path, index=False)
