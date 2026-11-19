# nutrition-label-llm

LoRA fine-tune of **Qwen2.5-14B-Instruct** (via [Unsloth](https://github.com/unslothai/unsloth)) that parses raw OCR text from EU nutrition labels (Regulation EU 1169/2011) into structured JSON.

Built for the nutrition-label scanning feature of Macro Micro, my nutrition and fitness tracking app. Version 1 of the app is retired.

## What it does

Input: noisy OCR text from a food label, in any EU-market language (Romanian, German, French, ...). This is the first example of the test set (`data/dataset/test.jsonl`). The OCR read the label as three blocks, misread `536 kJ` as `S36 kJ` and `20,0 g` as `ZO,O g`, and the third block holds the per-portion values, which the model must ignore:

```
BLOCK 1: Valoare energetica
Grasimi
din care: saturati
Hidrati de carbon
din care: zaharuri
Proteina
Continut de sare
BLOCK 2: la 100g
S36 kJ. / 128.0 cal
5,0g
2,0gr.
0,5g
0,5 gr
ZO,O g
2,0 gr
BLOCK 3: la portie (148g)
189,0 g
7,4gr
3,0 g
0,7 gr.
0,7 G
29,6 gr
3,0 G
```

Expected output: normalized JSON, per 100 g. The label has no fiber line, so `fiber` is `null`. Sodium is 2.0 g of salt divided by 2.5:

```json
{"calories": 128.0, "fats": 5.0, "saturated_fat": 2.0, "carbs": 0.5, "sugar": 0.5, "protein": 20.0, "sodium": 0.8, "fiber": null}
```

Handles comma/point decimal separators, unit variations, label-language differences and sodium ↔ salt conversion (sodium = salt / 2.5).

## Training

- **Dataset:** 50,000 synthetic examples (40k train / 5k val / 5k test), generated to cover multiple languages, formats and OCR noise levels — see `data/`
- **Method:** 4-bit QLoRA (r=16, alpha=32) with Unsloth + TRL `SFTTrainer`
- **Hardware:** single RTX 5090 (32 GB VRAM)

## Files

- `train.py` — training script (config at the top)
- `setup.py` — environment setup
- `data/dataset11.zip` — the synthetic dataset
