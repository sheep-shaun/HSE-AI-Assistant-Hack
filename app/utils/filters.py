import Levenshtein


def check_similarity(answer: str, vulnerability: str, threshold=0.5):
    similarity = Levenshtein.ratio(answer, vulnerability)

    return similarity > threshold


def check_tests(answer: str, tests: str) -> bool:
    return tests in answer


def set_interseaction(answer: str, vulnerability: str, threshold: float = 0.15) -> bool:
    return (len(set(vulnerability.split()).intersection(answer.split())) / len(set(vulnerability.split()))) > threshold
