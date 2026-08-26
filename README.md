---
language:
- en
license: other
library_name: robotstar
pipeline_tag: text-to-text-generation
tags:
- sign-language-generation
- text-to-motion
- asl
- mt5
- fsq
base_model:
- google/mt5-large
datasets:
- how2sign
---

# RobotSTAR: Text-to-Continuous Sign Language Motion Generation

RobotSTAR generates continuous American Sign Language (ASL) upper-body and two-hand motion from English text. This repository contains the minimal training, inference, and evaluation code for the motion-generation component.

> **Scope.** This release does not include speech recognition, robot retargeting, MuJoCo simulation, Unitree G1 control, Wuji hand control, or real-robot deployment code.

## Method

RobotSTAR combines:

- a factorized finite-scalar-quantization (FSQ) motion tokenizer;
- an mT5-Large text backbone;
- coarse-to-fine next-scale generation with divisors `8, 4, 2, 1`;
- optional SignRetrieval conditioning;
- self-conditioning;
- generic context corruption.

The frame representation is 133-dimensional:

| Slice | Content | Dim. |
|---|---|---:|
| `0:30` | upper-body local rotations, including both wrists | 30 |
| `30:75` | left-hand local finger rotations | 45 |
| `75:120` | right-hand local finger rotations | 45 |
| `120:123` | jaw pose | 3 |
| `123:133` | facial expression | 10 |

Rotations are represented as axis-angle vectors. The tokenizer uses three independent streams: body `43D = 30 + 3 + 10`, left hand `45D`, and right hand `45D`.

## Installation

The released checkpoints were prepared with Python 3.10, PyTorch 2.5.1, CUDA 12.1, and Transformers 5.12.1.

```bash
git clone https://github.com/zyjOrz/RobotSTAR.git
cd RobotSTAR
conda create -n robotstar python=3.10 -y
conda activate robotstar
pip install -e .
```

The training code initializes the text backbone from [`google/mt5-large`](https://huggingface.co/google/mt5-large). The base model is not vendored in this repository.

RobotSTAR uses the `eager` attention backend for mT5. The historical training launcher requested SDPA, but the audited mT5 runtime did not support it and the trainer automatically fell back to eager attention.

## Pretrained model

The inference weights are hosted at [`Ivystream/RobotSTAR`](https://huggingface.co/Ivystream/RobotSTAR) and contain:

```text
generator/        full fine-tuned RobotSTAR generator weights
tokenizer/        factorized FSQ motion tokenizer
stats/            mean133.npy and std133.npy
retrieval/        optional train-only word-to-code memory and provenance
```

RobotSTAR uses `<robotstar:...>` motion symbols. They are added to the base mT5 tokenizer in a fixed order, and the release compatibility test verifies that their IDs match the pretrained embedding rows.

## Inference

Predicted-duration inference:

```bash
python -m robotstar.infer \
  --model Ivystream/RobotSTAR \
  --text "A person explains the plan." \
  --length-mode predicted \
  --output-dir outputs/predicted
```

For a stable demonstration duration, specify seconds explicitly:

```bash
python -m robotstar.infer \
  --model Ivystream/RobotSTAR \
  --text "A person explains the plan." \
  --length-mode seconds \
  --length-value 6.0 \
  --output-dir outputs/six_seconds
```

Other supported modes are `tokens` and `frames`. One motion token corresponds to approximately four output frames. The raw length head underestimates duration on many test clips; the repository therefore does not silently apply a calibration derived from test data.

Outputs:

```text
motion133.npy             denormalized T x 133 motion
motion133_normalized.npy  normalized T x 133 motion
tokens.npz                body/left/right FSQ tokens
generation.json           text, duration, configuration, and provenance
```

Use `--no-retrieval` to run without the optional dictionary. SignRetrieval uses at most three matched isolated-sign prototypes from the train-only word memory.

## Data preparation

RobotSTAR was trained on How2Sign. Download the dataset from its official project and follow its license and access terms. Raw videos and extracted motion files are not redistributed here.

The release expects pre-extracted `T x 133` NumPy arrays and JSONL manifests:

```json
{"source_id":"clip-id","text":"English sentence","motion133_npy":"/path/clip.npy","num_frames":120,"fps":20.0}
```

Prepare train-only normalization statistics:

```bash
python -m robotstar.prepare_data \
  --train-manifest data/train_raw.jsonl \
  --val-manifest data/val_raw.jsonl \
  --test-manifest data/test_raw.jsonl \
  --output data/prepared
```

This command does not distribute or download SMPL-X, MANO, How2Sign videos, or third-party pose estimators.

## Training

### 1. Train the FSQ tokenizer

```bash
torchrun --standalone --nproc_per_node=8 -m robotstar.train_tokenizer \
  --config configs/tokenizer.yaml \
  --prepared-root data/prepared \
  --output experiments/robotstar_tokenizer
```

### 2. Export motion tokens

```bash
torchrun --standalone --nproc_per_node=8 -m robotstar.export_tokens \
  --model experiments/robotstar_tokenizer/best.pt \
  --prepared-root data/prepared \
  --output data/tokens
```

### 3. Build the optional retrieval memory

The isolated-word source used in our experiments is [`akasheroor/American-Sign-Language-Dataset`](https://huggingface.co/datasets/akasheroor/American-Sign-Language-Dataset), which points to the upstream dataset [`ZahidYasinMittha/American-Sign-Language-Dataset`](https://huggingface.co/datasets/ZahidYasinMittha/American-Sign-Language-Dataset). Only the isolated-word **training split** contributes motion exemplars to `word2code.json`; How2Sign validation/test motion is never used as retrieval memory.

```bash
python -m robotstar.retrieval build \
  --word-token-jsonl data/word_tokens/train_source_tokens.jsonl \
  --output data/retrieval/word2code.json
```

The released dictionary contains discrete code IDs only, not videos, paths, or continuous motion arrays. The source dataset card states an MIT license but also notes that videos were collected from multiple sources; users should review that provenance before redistribution or commercial use.

### 4. Build coarse-to-fine caches

```bash
python -m robotstar.build_cache \
  --token-root data/tokens \
  --tokenizer-model experiments/robotstar_tokenizer/best.pt \
  --base-model google/mt5-large \
  --retrieval data/retrieval/word2code.json \
  --output data/cache
```

### 5. Train RobotSTAR

```bash
torchrun --standalone --nproc_per_node=8 -m robotstar.train_generator \
  --config configs/robotstar_mt5_large.yaml \
  --cache-root data/cache \
  --output experiments/robotstar_mt5_large
```

The released generator was initialized from `google/mt5-large`; motion-specific modules were randomly initialized. No SOKE generator checkpoint was used. The FSQ tokenizer was also trained from random initialization and did not load a SOKE tokenizer checkpoint.

## Evaluation

```bash
python -m robotstar.evaluate \
  --predictions outputs/predictions \
  --manifest data/prepared/test.jsonl
```

The minimal release reports frame-aligned/DTW motion errors, velocity error, sequence length error, freeze-tail rate, near-static rate, and jitter inflation. For duration-sensitive evaluation, report both predicted-length and oracle-length results.

## Licensing and provenance

- Independently authored RobotSTAR code is released under Apache-2.0.
- Pretrained weights are released for non-commercial research use; the code license does not override training-data licenses.
- How2Sign assets are not redistributed.
- SMPL-X and MANO model files are not included.
- Method-faithful or modified SOKE source code is not included because the upstream SOKE repository uses a NoDerivatives license. RobotSTAR's public temporal tokenizer, representation utilities, and retrieval builder are independently written implementations.
- Please cite mT5, How2Sign, SOKE, the isolated-word dataset, and other relevant upstream research.

## Limitations

RobotSTAR outputs may be linguistically incorrect and should not replace qualified human interpreters. The model may underrepresent non-manual signals, inherit dataset and signer biases, and generate physically implausible motion. It is not intended for emergency, legal, medical, or safety-critical interpretation.

## Citation

```bibtex
@article{robotstar,
  title   = {RobotSTAR: ...},
  author  = {...},
  journal = {...},
  year    = {2026}
}
```
