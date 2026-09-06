# Synthetic demonstration corpus

This directory contains a small, fully synthetic corpus for testing the
repository pipeline.

The messages were created from templates and were not derived from the
confidential DocTalk corpus. They do not represent real patients, employees,
clinical cases, conversations, or events.

The synthetic sample mirrors the intermediate corpus tables produced by 01_build_corpus.py. It can therefore be used to test the workflow from lexical cleaning onward. The original raw export structure cannot be shared.

## Purpose

The synthetic sample can be used to:

- inspect the expected input structure;
- test lexical cleaning rules;
- validate marker normalization;
- execute selected analysis scripts;
- demonstrate the distinction between direct and group communication.

The sample is not intended to reproduce the frequencies, effect sizes,
collocations, or other empirical results reported in the associated study.

## Generation

The files can be regenerated with:

```bash
python scripts/00_generate_synthetic_sample.py