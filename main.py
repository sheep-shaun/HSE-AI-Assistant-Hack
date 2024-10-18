import os
import time

from typing import Optional

import pandas as pd
from dotenv import load_dotenv

from app.models.yandexgpt import YandexGPT
from app.models.catboost_jailbreak_inference import jailbreak_inference
from app.models.ranker import Ranker
from app.utils.submit import generate_submit
from app.utils.postcheck_regex import is_python_code
from app.utils.prompts import zaglushka
from app.utils.entities import DatasetRow
from app.utils.dataset import merge_datasets
from app.utils.code_execute import execute_dataset


if __name__ == "__main__":
    load_dotenv()

    system_prompt = """Ты - профессиональный программист и ментор. Давай очень короткие подсказки о синтаксических ошибках в коде."""

    yandex_gpt = YandexGPT(
        token=os.environ["YANDEX_GPT_IAM_TOKEN"],
        folder_id=os.environ["YANDEX_GPT_FOLDER_ID"],
        model_name="tune_v1.3",
        system_prompt=system_prompt,
        temperature=0.3,
        max_tokens=200,
    )

    ranker_api = Ranker(model_path="./ranker_weighted.bin", device="cpu")

    def predict(row: pd.Series) -> Optional[str]:
        row = DatasetRow(row)
        
        if jailbreak_inference([row.student_solution]) == 1: # jailbreak class
            print(f"filtred: {row.student_solution}")
            return zaglushka()

        prefix = f'''Описание задачи, которую решает студент:
{row.description}\n
diff решения студента с авторским (правильным) решением:
```diff
{row.code_diff}
```'''

        if row.executed and "error" in row.test_result.lower():
            prefix = f'''{prefix}\n
Тип теста: {row.failed_test_type}
Тест: {row.failed_test_input}
Ошибка решения студента: {row.test_result}'''

        elif row.executed:

            prefix = f'''{prefix}\n
Тип теста: {row.failed_test_type}
Тест: {row.failed_test_input}
Ответ решения студента: {row.test_result}
Правильный ответ: {row.failed_test_output}'''

        prefix += f'''\n
Ошибка в коде гарантированно есть.
Дай подсказку студенту.'''

        #
        generated = list()
        for _ in range(3):
            time.sleep(1)
            generated.append(yandex_gpt.ask(prefix).strip())
        max_score = -1
        final_predict = generated[0]

        for generation in generated:
            score = ranker_api.predict_proba(generation)
            if score > max_score:
                max_score = score
                final_predict = generation
        #
        
        if is_python_code(final_predict):
            print(final_predict)
            return zaglushka()

        return final_predict
    
    test_solutions = pd.read_excel("./data/test/solutions.xlsx")
    test_tasks = pd.read_excel("./data/test/tasks.xlsx")
    test_tests = pd.read_excel("./data/test/tests.xlsx")
    test = merge_datasets(test_solutions, test_tasks, test_tests)
    test = execute_dataset(test)
    final_dataset_path = "./data/test/merged_executed_test.xlsx"
    test.to_excel(final_dataset_path)

    generate_submit(
        test_solutions_path=final_dataset_path,
        predict_func=predict,
        save_path="./data/processed/yandexgpt_tune_v1_3_weighted_ranker_test_2.csv",
        use_tqdm=True,
    )
