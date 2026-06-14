# Cleaning decision log

## 2026-06-14

Manual validation showed that `@<Nachname>` refers to colleague addressing in the analyzed corpus and must therefore be standardized as `Mention_KolName`, not as `PatName`.

Manual validation also showed that combined full-name placeholders such as `<Vorname><Nachname>` refer to colleagues and are therefore standardized as `KolName`, whereas remaining standalone `<Nachname>` placeholders are treated as patient-name references and standardized as `PatName`.

Rule order was adjusted accordingly:
1. salutation + `<Nachname>` → `PatName`
2. `@<Vorname><Nachname>` / `@<Nachname>` → `Mention_KolName`
3. `<Vorname><Nachname>` → `KolName`
4. remaining `<Nachname>` → `PatName`