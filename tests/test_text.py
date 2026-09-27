from app.text import TextNormalizer


def test_normalize_text():
    text = """
        საცხოვრებელი სახლის მშენებლობა
        დასრულდება    2026 წელს.


        სამუშაოების   ღირებულება შეადგენს
        50 000 ლარს.
    """

    normalizer = TextNormalizer()

    result = normalizer.normalize(text)

    assert result == (
        "საცხოვრებელი სახლის მშენებლობა\n"
        "დასრულდება 2026 წელს.\n\n"
        "სამუშაოების ღირებულება შეადგენს\n"
        "50 000 ლარს."
    )
