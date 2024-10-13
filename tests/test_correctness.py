import pandas as pd

from app.utils.submit import string2embedding

TEST_SIZE = 347
EMBEDDING_SIZE = 768


def _check_ids_correctness(submit_df: pd.DataFrame, submit_example_df: pd.DataFrame) -> bool:
    not_presented = set(submit_example_df["solution_id"]) - set(submit_df["solution_id"])
    not_needed = set(submit_df["solution_id"]) - set(submit_example_df["solution_id"])

    not_presented = list(not_presented)
    not_presented.sort()
    not_needed = list(not_needed)
    not_needed.sort()

    error_message = "Submit is incorrect."
    if len(not_presented) + len(not_needed) > 0:
        if len(not_presented) > 0:
            error_message += f" Not presented solution_id: {not_presented}."
        if len(not_needed) > 0:
            error_message += f" Not needed solution_id: {not_needed}."
        raise ValueError(error_message)
    return True


def _check_rows_size_correctness(submit_df: pd.DataFrame) -> bool:
    incorrect_rows = []
    for idx in range(TEST_SIZE):
        if len(string2embedding(submit_df["author_comment_embedding"].iloc[idx])) != EMBEDDING_SIZE:
            incorrect_rows.append(idx)
    if len(incorrect_rows) > 0:
        raise ValueError(f"Submit has incorrect rows: {incorrect_rows}. (incorrect size of embedding)")
    return True


def check_submit_correctness(submit_path: str, submit_example_path: str) -> bool:
    if not submit_path.endswith(".csv"):
        raise ValueError(f"{submit_path} is not a .csv file.")

    submit_df = pd.read_csv(submit_path)
    submit_example_df = pd.read_csv(submit_example_path)

    _check_ids_correctness(submit_df, submit_example_df)
    _check_rows_size_correctness(submit_df)

    return True


if __name__ == "__main__":
    check_submit_correctness(submit_path="data/complete/submit.csv", submit_example_path="data/raw/submit_example.csv")
