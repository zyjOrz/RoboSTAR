<p align="center">
  <img src="assets/robostar_logo.png" alt="RoboSTAR Icon" width="80"/>
</p>
<h1 align="center"><strong>RoboSTAR: Next-Scale Autoregressive Sign Language Motion Translation for Humanoid Robots</strong></h1>
  <p align="center">
              <a href="https://www.yujiazeng.com/">Yujia Zeng<sup>*</sup></a>
              <a href="https://pengchensheng.com/">Chensheng Peng<sup>*</sup></a>
              <a href="https://thomaschen98.github.io/">Yuxin Chen<sup>*</sup></a>
              <a href="https://alexshao.net/">Alex Shao<sup></sup></a>
              <a href="https://www.linkedin.com/in/nathan-jew-a31314273/">Nathan Jew<sup></sup></a>
              <a href="https://me.berkeley.edu/people/masayoshi-tomizuka/">Masayoshi Tomizuka<sup></sup></a>    <br>
    <sup></sup>UC Berkeley <br> <sup>*</sup>Equal contribution
</p>

<p align="center">
  <img src="https://img.shields.io/badge/arXiv-Coming_Soon-blue">

  <a href="https://zyjorz.github.io/RoboSTAR/">
    <img src="https://img.shields.io/badge/Project-Page-blue">
  </a>

  <a href="https://github.com/zyjOrz/RoboSTAR">
    <img src="https://img.shields.io/badge/GitHub-Code-black">
  </a>

  <img src="https://img.shields.io/badge/🤗_HuggingFace-Coming_Soon-orange">
</p>

<p align="center">
  <em>
    RoboSTAR translates English text or audio into continuous American Sign Language (ASL) motion. <br>
    This release contains the text-to-sign-motion component.
  </em>
</p>



https://github.com/user-attachments/assets/b2433425-9a14-41b7-96aa-a83644eb061c



## Installation


```bash
git clone https://github.com/zyjOrz/RoboSTAR.git
cd RoboSTAR
conda create -n robostar python=3.10 -y
conda activate robostar
pip install -e .
```


## Inference

Predicted-duration inference:

```bash
python -m robostar.infer \
  --model Ivystream/RoboSTAR \
  --text "A person explains the plan." \
  --length-mode predicted \
  --output-dir outputs/predicted
```

For a stable demonstration duration, specify seconds explicitly:

```bash
python -m robostar.infer \
  --model Ivystream/RoboSTAR \
  --text "A person explains the plan." \
  --length-mode seconds \
  --length-value 6.0 \
  --output-dir outputs/six_seconds
```

Other supported modes are `tokens` and `frames`. 


## Data preparation


```bash
python -m robostar.prepare_data \
  --train-manifest data/train_raw.jsonl \
  --val-manifest data/val_raw.jsonl \
  --test-manifest data/test_raw.jsonl \
  --output data/prepared
```

## Training

### 1. Train the FSQ tokenizer

```bash
torchrun --standalone --nproc_per_node=8 -m robostar.train_tokenizer \
  --config configs/tokenizer.yaml \
  --prepared-root data/prepared \
  --output experiments/robostar_tokenizer
```

### 2. Export motion tokens

```bash
torchrun --standalone --nproc_per_node=8 -m robostar.export_tokens \
  --model experiments/robostar_tokenizer/best.pt \
  --prepared-root data/prepared \
  --output data/tokens
```

### 3. Build the optional retrieval memory

The isolated-word source used in our experiments is [`akasheroor/American-Sign-Language-Dataset`](https://huggingface.co/datasets/akasheroor/American-Sign-Language-Dataset).

```bash
python -m robostar.retrieval build \
  --word-token-jsonl data/word_tokens/train_source_tokens.jsonl \
  --output data/retrieval/word2code.json
```


### 4. Build coarse-to-fine caches

```bash
python -m robostar.build_cache \
  --token-root data/tokens \
  --tokenizer-model experiments/robostar_tokenizer/best.pt \
  --base-model google/mt5-large \
  --retrieval data/retrieval/word2code.json \
  --output data/cache
```

### 5. Train RoboSTAR

```bash
torchrun --standalone --nproc_per_node=8 -m robostar.train_generator \
  --config configs/robostar_mt5_large.yaml \
  --cache-root data/cache \
  --output experiments/robostar_mt5_large
```


## Evaluation

```bash
python -m robostar.evaluate \
  --predictions outputs/predictions \
  --manifest data/prepared/test.jsonl
```


## 🔗 Citation

If you find our work useful, please consider citing:

```bibtex
@article{robostar,
  title   = {RoboSTAR: ...},
  author  = {...},
  journal = {...},
  year    = {2026}
}
```

## 🙏 Acknowledgements

- [SOKE](https://github.com/2000ZRL/SOKE): Our work follows SOKE in adopting its motion representation and sign retrieval formulation.

- [How2Sign Dataset](https://how2sign.github.io/): Our work uses the How2Sign dataset for training and evaluation.

- [HandMDM](https://imagine.enpc.fr/~leore.bensabath/HandMDM/): Our human-mesh rendering and visualization pipeline is inspired by HandMDM.
