import pandas as pd
import re
from pathlib import Path

input_path = Path(
    "outputs/confidential/cleaned_corpus_tables/"
    "utterances_for_collocation_clean_lexical_v2.csv"
)

output_path = Path(
    "outputs/confidential/validation_tables/"
    "validate_original_kein_aktives_todo_v2.xlsx"
)
output_path.parent.mkdir(parents=True, exist_ok=True)

pattern = re.compile(
    r"\bkeine?\s+(aktive[n]?|akute[n]?)?\s*to[\s-]?dos?\b|"
    r"\bkeine?\s+(aktive[n]?|akute[n]?)?\s*todos?\b",
    flags=re.IGNORECASE,
)

df = pd.read_csv(input_path)

rows = []

for _, row in df.iterrows():
    original = str(row.get("text_original", ""))
    cleaned_v1 = str(row.get("text_clean_lexical", ""))
    cleaned_v2 = str(row.get("text_clean_lexical_v2", ""))

    if pattern.search(original):
        rows.append({
            "corpus": row.get("direction", ""),
            "id": row.get("id", ""),
            "conversation_id": row.get("conversation_id", ""),
            "text_original": original,
            "text_clean_lexical": cleaned_v1,
            "text_clean_lexical_v2": cleaned_v2,
            "contains_kein_Todo_after_v1_cleaning": "kein_Todo" in cleaned_v1,
            "contains_kein_Todo_after_v2_cleaning": "kein_Todo" in cleaned_v2,
        })

out = pd.DataFrame(rows)
out.to_excel(output_path, index=False)

print(out.shape)

print("\nV1 result:")
print(out[["corpus", "contains_kein_Todo_after_v1_cleaning"]].value_counts())

print("\nV2 result:")
print(out[["corpus", "contains_kein_Todo_after_v2_cleaning"]].value_counts())

false_case_v2 = out[out["contains_kein_Todo_after_v2_cleaning"] == False]

print("\nFalse cases after v2 cleaning:")
if false_case_v2.empty:
    print("None")
else:
    print(
        false_case_v2[
            [
                "corpus",
                "id",
                "conversation_id",
                "text_original",
                "text_clean_lexical_v2",
            ]
        ].to_string(index=False)
    )

print(f"\nSaved validation table to: {output_path}")