# rag/utils.py
import unicodedata

def utf8_safe(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    # normalise et remplace le tiret cadratin par un simple '-'
    s = unicodedata.normalize("NFC", s)
    s = s.replace("\u2014", "-")
    # Si une lib force l'ASCII en dessous, on garde quand même tout ce qui est encodable UTF-8
    return s.encode("utf-8", "ignore").decode("utf-8", "ignore")