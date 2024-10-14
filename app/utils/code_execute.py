import sys
from io import StringIO

import copy
import pandas as pd


def execute_code(code: str, code_input: str):
    # redirect stdout
    original_stdout = sys.stdout
    captured_output = StringIO()
    sys.stdout = captured_output

    # imitating stdin
    original_stdin = sys.stdin
    sys.stdin = StringIO(code_input)

    try:
        exec(code)
    except Exception as e:
        # restore stdout/stdin
        sys.stdout = original_stdout
        sys.stdin = original_stdin
        return repr(e)

    code_output = captured_output.getvalue()[:-1]

    # restore stdout/stdin
    sys.stdout = original_stdout
    sys.stdin = original_stdin

    return code_output


def execute_dataset(data: pd.DataFrame) -> pd.DataFrame:
    data_rows = list()

    for i, row in enumerate(data.to_dict(orient="records")):
        result = None
        for test_i, test_case in enumerate(row["test_input"]):
            if pd.isna(test_case):
                continue
            test_result = execute_code(row["student_solution"], test_case)
            if "FileNotFoundError" in test_result:
                test_result = None
            if test_result != str(row["test_output"][test_i]):
                result = test_result
                break
        row["test_result"] = result
        row["failed_test"] = None if result is None else str(int(test_i))
        data_rows.append(copy.deepcopy(row))

    return pd.DataFrame(data_rows)
    

if __name__ == "__main__":
    from dataset import merge_datasets
    train_solutions = pd.read_excel("./data/train/solutions.xlsx")
    train_tasks = pd.read_excel("./data/train/tasks.xlsx")
    train_tests = pd.read_excel("./data/train/tests.xlsx")
    train = merge_datasets(train_solutions, train_tasks, train_tests)
    train = execute_dataset(train)
    train.to_excel("./data/train/merged_executed.xlsx")

    test_solutions = pd.read_excel("./data/test/solutions.xlsx")
    test_tasks = pd.read_excel("./data/test/tasks.xlsx")
    test_tests = pd.read_excel("./data/test/tests.xlsx")
    test = merge_datasets(test_solutions, test_tasks, test_tests)
    test = execute_dataset(test)
    test.to_excel("./data/test/merged_executed.xlsx")
