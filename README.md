# nutrition-label-llm

LoRA fine-tune of **Qwen2.5-14B-Instruct** (via [Unsloth](https://github.com/unslothai/unsloth)) that parses raw OCR text from EU nutrition labels (Regulation EU 1169/2011) into structured JSON.

Built for the nutrition-label scanning feature of [Macro Micro](https://macromicro.ro), my nutrition & fitness tracking app.

## What it does

Input: noisy OCR text from a food label, in any EU-market language (Romanian, German, French, ...):

```
Brennwert: 891KJ/213kcal
Fett: 13,4 gr
davon gesattigte Fettsauren: 1,7 gr
KH: 13,5 g
Zucker: 6,0 G
Protein: 9,7 gr
Salzgehalt: 0,1 g
```

Output: normalized JSON, per 100 g:

```json
{"calories": 213, "fats": 13.4, "saturated_fat": 1.7, "carbs": 13.5, "sugar": 6.0, "protein": 9.7, "sodium": 0.048, "fiber": 3.8}
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
