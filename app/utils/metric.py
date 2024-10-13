import pandas as pd
from torch.nn.functional import cosine_similarity

from app.utils.submit import string2embedding


def _get_cosine_similarity(pred_df: pd.DataFrame, true_df: pd.DataFrame) -> float:
    predictions = pred_df["author_comment_embedding"]
    true_values = true_df["author_comment_embedding"]
    total_cos_sim = 0

    for idx in range(len(true_values)):
        pred_value = string2embedding(predictions.iloc[idx])
        gt_value = string2embedding(true_values.iloc[idx])

        if len(pred_value) != len(gt_value):
            raise ValueError(f"Embeddings have different sizes: {len(pred_value)} != {len(gt_value)}")

        cos_sim_value = cosine_similarity(pred_value.unsqueeze(0), gt_value.unsqueeze(0))
        total_cos_sim += cos_sim_value
    return float(total_cos_sim / len(true_df))


def calculate_score(submit_path: str, gt_path: str) -> float:
    submit_df = pd.read_csv(submit_path)
    true_df = pd.read_excel(gt_path)
    submit_df = submit_df[submit_df["solution_id"].isin(true_df["id"])]
    return (_get_cosine_similarity(submit_df, true_df) - 0.6) / 0.4


def calculate_score_and_save(submit_path: str, gt_path: str, save_path: str) -> float:
    score = calculate_score(submit_path, gt_path)
    with open(save_path, "w") as f:
        f.write(f"{score}")
    return score
