import pandas as pd


def merge_datasets(solutions: pd.DataFrame, tasks: pd.DataFrame, tests: pd.DataFrame) -> pd.DataFrame:
    result = solutions.copy()

    result = result.merge(tasks, left_on="task_id", right_on="id", suffixes=('', '_need_to_drop'))

    tests_grouped = tests.groupby(by="task_id").agg(list)
    tests_grouped = tests_grouped.rename(
        columns={
            "id": "test_id",
            "type": "test_type",
            "input": "test_input",
            "output": "test_output",
        },
    )
    result = result.merge(tests_grouped, left_on="task_id", right_on="task_id", suffixes=('', '_need_to_drop'))

    need_to_drop = [col for col in result.columns if col.endswith("_need_to_drop")]
    result = result.drop(columns=need_to_drop)

    assert result.shape[0] == solutions.shape[0]
    assert result.shape[1] == 13

    return result
