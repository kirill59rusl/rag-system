import re

# "PostgreSQL 18", "PG 19", "v18", "18-й версии", "в 18 и в 19";
# не цепляет номера разделов (19.4) и числа внутри других чисел
_VERSION_MENTION = re.compile(
    r"(?<!\d)(?<!\d\.)(\d{1,2})(?:-?(?:[йяюе]|ой|ая|ую)|th)?(?!\w|\.\d)", re.IGNORECASE
)


def detect_versions(query: str, known: set[str]) -> list[str]:
    """Версии документации из known, упомянутые в вопросе, в порядке появления."""
    found = []
    for match in _VERSION_MENTION.finditer(query):
        version = match.group(1)
        if version in known and version not in found:
            found.append(version)
    return found
