# SFT Data v1 Audit

## Scope

- Date: 2026-08-22
- Fixed audit file: `data/processed/sft_audit_100.jsonl`
- Audit SHA-256: `3aac420c693718ef7b321929de05c8f4178c0dd4c2a5a090dda78a7e69c12a2d`
- Samples: 100 unique problems, all `verified=true`
- Source distribution: TACO 58, APPS 28, CodeContests 11, open-r1/Codeforces 3
- Difficulty distribution: easy 24, medium 27, hard 15, unknown 34
- Total-token range: maximum 15,028; no sample exceeds 16,384

## Checks performed

Each fixed sample was reviewed through its problem statement, reasoning summary, final C++ program, provenance and token counts. Automated checks additionally confirmed:

- the problem is not empty or the OCR2 `-` placeholder;
- `response` exactly renders Output Protocol v1 from `reasoning` and `code`;
- the response contains a complete C++ program with `main`;
- every program passed the preparation-time GNU C++17 compile check;
- every sample has OCR2 `judgement=right`, `pass_rate >= 0.8` and a restored upstream problem;
- no sample is an interactive task or explicitly lacks its problem statement;
- no sample exceeds the 16,384-token training limit;
- no retained sample matches the frozen Eval v1 exact/near-duplicate checks.

## Findings and remediation

The first audit pass found one blocking sample whose statement explicitly said the problem statement was missing. Its reasoning guessed the hidden operation from examples. A full-data follow-up found 14 additional interactive problems, which are incompatible with the project's stdin/stdout protocol. The preparation pipeline now rejects both categories, then refills from the ordered verified candidate pool. The regenerated 10K dataset and fixed 100-sample audit contain none of them.

No blocking issue remains in the regenerated audit. Non-blocking quality limitations retained and recorded are:

- reasoning is often verbose and can repeat exploratory steps;
- 34/100 audited samples have unknown source difficulty;
- a small number of deliberately named “mystery” tasks infer a simple rule from examples; these were retained only when the statement provides the intended sample-based contract and the verified solution follows it;
- source statements vary in formatting and editorial quality;
- tags are unavailable in v1, so balancing uses difficulty and platform only.

Passing upstream tests and compilation is stronger evidence than manual inspection but is not a proof that every reasoning sentence is correct. This audit therefore records format, alignment, task suitability and visible quality defects separately from upstream execution verification.

## Final dataset identity

- `sft_1k.jsonl`: `f572d6aed4d8ba72467185ffcdc3efa127521f84e054f0f44af83d9e246c9aed`
- `sft_5k.jsonl`: `de8acef7eb175efdf681444c67bef0a38cf649f9ecd0afe0574225e33cbc8aa7`
- `sft_10k.jsonl`: `16d25b5ad5780b4b5925a6a504210c11c7d39f35b535e7c783e6f3e9398a3581`

The sets are strictly nested by ordered prefix (`1K ⊂ 5K ⊂ 10K`), contain 10,000 unique problem IDs at the largest size, retain zero Eval matches, and contain zero samples over 16,384 total tokens. M5 is accepted for entry into M6 SFT smoke testing.
