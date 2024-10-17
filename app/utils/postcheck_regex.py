import re

def is_python_code(text):
    ignore_phrases = [
        r"(обратите внимание|ошибка|неправильн|некорректн|проблема)",  
        r"(переменная|цикл|условие|функция|двоеточие|отступ|синтаксис)", 
        r"(использован|выполняет|выведите|проверьте|попробуйте|подправьте|скобки|метод|оператор)"  
    ]
    
    code_pattern = re.compile(r"\b(if|for|while|def|return|class|try|except|with|import)\b.*?:")
    
    for phrase in ignore_phrases:
        if re.search(phrase, text, flags=re.IGNORECASE):
            return False

    return bool(code_pattern.search(text))


if __name__ == "__main__":
    test = """Вы забыли поставить двоеточие после условия цикла for ."""
    print(is_python_code(test))