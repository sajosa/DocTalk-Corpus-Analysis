import pandas as pd
import re
from pathlib import Path

files = [
    Path("outputs/confidential/cleaned_corpus_tables/D_utterances_clean_lexical.csv"),
    Path("outputs/confidential/cleaned_corpus_tables/G_utterances_clean_lexical.csv"),
]

pattern = re.compile(
    r"\bkeine?\s+(aktive[n]?|akute[n]?)?\s*to[\s-]?dos?\b|\bkeine?\s+(aktive[n]?|akute[n]?)?\s*todos?\b",
    flags=re.IGNORECASE,
)

rows = []

for path in files:
    df = pd.read_csv(path)
    corpus = "direct" if path.name.startswith("D_") else "group"

    for _, row in df.iterrows():
        original = str(row.get("text_original", ""))
        cleaned = str(row.get("text_clean_lexical", ""))

        if pattern.search(original):
            rows.append({
                "corpus": corpus,
                "id": row.get("id", ""),
                "conversation_id": row.get("conversation_id", ""),
                "text_original": original,
                "text_clean_lexical": cleaned,
                "contains_kein_Todo_after_cleaning": "kein_Todo" in cleaned,
            })

out = pd.DataFrame(rows)
out.to_excel(
    "outputs/results/collocations_v2/validate_original_kein_aktives_todo.xlsx",
    index=False
)

print(out.shape)
print(out[["corpus", "contains_kein_Todo_after_cleaning"]].value_counts())



import pandas as pd

path = "outputs/results/collocations_v2/validate_original_kein_aktives_todo.xlsx"
df = pd.read_excel(path)

false_case = df[df["contains_kein_Todo_after_cleaning"] == False]

print(false_case[["corpus", "id", "conversation_id", "text_original", "text_clean_lexical"]].to_string(index=False))