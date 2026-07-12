# DocTalk-Corpus-Analysis
Analysis of the DocTalk Chat corpus 


- Kontext der Korpusanalyse
- Bezug zur Masterarbeit
- Installationsanleitung
- Ausführen der Notebooks
- Datenverfügbarkeit
- Datenschutz/Ethik
- Zitation

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt


Cleaning decisions are documented in `rules/cleaning_decision_log.md`. The log
records corpus-specific manual validation decisions that informed placeholder
normalization, gender-inclusive form normalization, ToDo negation handling, and
other rule-based lexical cleaning steps.