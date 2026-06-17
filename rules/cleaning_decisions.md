# Cleaning decision log

## 2026-06-14

Manual validation showed that `@<Nachname>` refers to colleague addressing in the analyzed corpus and must therefore be standardized as `Mention_KolName`, not as `PatName`.

Manual validation also showed that combined full-name placeholders such as `<Vorname><Nachname>` refer to colleagues and are therefore standardized as `KolName`, whereas remaining standalone `<Nachname>` placeholders are treated as patient-name references and standardized as `PatName`.

Rule order was adjusted accordingly:
1. salutation + `<Nachname>` → `PatName`
2. `@<Vorname><Nachname>` / `@<Nachname>` → `Mention_KolName`
3. `<Vorname><Nachname>` → `KolName`
4. remaining `<Nachname>` → `PatName`


## Manual validation: negated ToDo possessive variants

Manual validation showed that variants such as `kein_Todo 's` occurred after initial ToDo normalization. These forms were standardized to `kein_Todo` to avoid artificial token fragmentation in lexical frequency and N-gram analyses.

## Manual validation: quotation marks and apostrophe remnants

Manual validation of the frequency tables showed that quotation marks and apostrophe remnants were preserved as part of tokens, for example `"Wie`, `"antriggern"` or `Fortschritt'`. Residual quotation marks and apostrophe characters were therefore added to the final punctuation-removal step to avoid artificial token fragmentation in frequency and N-gram analyses.

## Manual validation: artefact from gender-inclusive user references

Manual validation of token and N-gram tables showed that the artefact `inname` occurred in sequences such as `Benutzer inname`. These cases originated from split gender-inclusive user references. To avoid artificial token fragmentation, `Benutzer inname` was standardized to `Benutzer_innen`.


## Manual validation: gender-inclusive colleague and user references

Manual validation of token and N-gram tables showed that gender-inclusive forms such as `Kolleg:innen` and `Benutzer:innen` were split into artificial token sequences such as `Kolleg innen` and `Benutzer inname`. In the analyzed clinical communication corpus, these forms referred to colleagues or platform users within the care team. They were therefore subsumed under the standardized token `KolName` rather than retained as separate gender-inclusive surface forms. This decision reduced artificial token fragmentation while preserving the analytical category of colleague/user references.