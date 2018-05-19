import re


def text_from_path(path):
    return __clean_text(open(path).read())


def get_chars(text):
    return sorted(list(set(text)))


def get_char_indices(chars):
    return dict((char, chars.index(char)) for char in chars)


def __clean_text(text):
    return (
        re.sub(r'\s+', ' ', text)
        .replace('“', '"')
        .replace('”', '"')
        .replace('‘', '\'')
        .replace('’', '\'')
        .lower()
    )
