import ast
import difflib
import pandas as pd


class DatasetRow:
    def __init__(self, row: pd.Series) -> None:
        self.id = row["id"]
        self.task_id = row["task_id"]
        self.student_solution = row["student_solution"]
        self.author_comment = row["author_comment"]
        self.author_comment_embedding = row["author_comment_embedding"]
        self.level = row["level"]
        self.description = row["description"]
        self.author_solution = row["author_solution"]
        self.test_id = row["test_id"]
        self.number = row["number"]
        self.test_type = row["test_type"]
        self.test_input = row["test_input"]
        self.test_output = row["test_output"]
        self.test_result = row["test_result"]
        self.failed_test = row["failed_test"]
        
        self.code_diff = '\n'.join(difflib.ndiff(self.student_solution.splitlines(), self.author_solution.splitlines()))
        
        self.test_input = ast.literal_eval(self.test_input.replace("nan", "None"))
        self.test_output = ast.literal_eval(self.test_output.replace("nan", "None"))
        self.test_type = ast.literal_eval(self.test_type.replace("nan", "None"))

        self.executed = not pd.isna(self.test_result)

        if self.executed:
            self.failed_test_input = self.test_input[int(self.failed_test)]
            self.failed_test_output = self.test_output[int(self.failed_test)]  # correct answer to the test
            self.failed_test_type = "Открытый" if self.test_type[int(self.failed_test)] == "open" else "Закрытый"
