<div align="center">

# 字符识别算法：PARSeq <br/> 语言拓展：中文训练与应用


[**原项目地址**](https://github.com/baudm/parseq) [**论文**](https://arxiv.org/pdf/2207.06966.pdf)

[环境安装](#环境安装) | [数据准备](#数据准备) | [训练](#getting-started) | [评估](#frequently-asked-questions) | [部署](#training)

</div>

场景文本识别 (STR) 模型使用语言上下文来增强对噪声或损坏图像的鲁棒性。 最近的方法（例如 ABINet）使用独立或外部语言模型 (LM) 来进行预测细化。 在这项工作中，我们表明，外部 LM（需要预先分配专用计算能力）对于 STR 而言效率低下，因为其性能与成本特征较差。 我们提出了一种使用置换自回归序列（PARSeq）模型的更有效的方法。 请查看我们的 [海报](https://drive.google.com/file/d/19luOT_RMqmafLMhKQQHBnHNXV7fOCRfw/view) 和 [PPT](https://drive.google.com/file/d/11VoZW4QC5tbMwVIjKB44447uTiuCJAAD/view) 以获取简要概述。

![PARSeq](.github/gh-teaser.png)

**NOTE:** 更多信息请查看原项目
## 环境安装

## 数据准备
1. 按照文件树准备数据集
```python
data
├── gt.txt
└── test
    ├── word_1.png
    ├── word_2.png
    ├── word_3.png
    └── ...
```
gt.txt：数据集的标签文件，其中每行文本为：{图像路径}\t{标签}\n,例如
```python
test/word_1.png 这里
test/word_2.png 那里
test/word_3.png 嘟嘟嘟
...
```
<br/>

2. 调用脚本生成lmdb样式数据集
```python
pip3 install fire
python3 create_lmdb_dataset.py --inputPath data/ --gtFile data/gt.txt --outputPath result/
...
```
### Demo
An [interactive Gradio demo](https://huggingface.co/spaces/baudm/PARSeq-OCR) hosted at Hugging Face is available. The pretrained weights released here are used for the demo.

### Installation
Requires Python >= 3.8 and PyTorch >= 1.10 (until 1.13). The default requirements files will install the latest versions of the dependencies (as of June 1, 2023).
```bash
# Use specific platform build. Other PyTorch 1.13 options: cu116, cu117, rocm5.2
platform=cpu
# Generate requirements files for specified PyTorch platform
make torch-${platform}
# Install the project and core + train + test dependencies. Subsets: [train,test,bench,tune]
pip install -r requirements/core.${platform}.txt -e .[train,test]
 ```
#### Updating dependency version pins
```bash
pip install pip-tools
make clean-reqs reqs  # Regenerate all the requirements files
 ```
### Datasets
Download the [datasets](Datasets.md) from the following links:
1. [LMDB archives](https://drive.google.com/drive/folders/1NYuoi7dfJVgo-zUJogh8UQZgIMpLviOE) for MJSynth, SynthText, IIIT5k, SVT, SVTP, IC13, IC15, CUTE80, ArT, RCTW17, ReCTS, LSVT, MLT19, COCO-Text, and Uber-Text.
2. [LMDB archives](https://drive.google.com/drive/folders/1D9z_YJVa6f-O0juni-yG5jcwnhvYw-qC) for TextOCR and OpenVINO.

### Pretrained Models via Torch Hub
Available models are: `abinet`, `crnn`, `trba`, `vitstr`, `parseq_tiny`, and `parseq`.
```python
import torch
from PIL import Image
from strhub.data.module import SceneTextDataModule

# Load model and image transforms
parseq = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval()
img_transform = SceneTextDataModule.get_transform(parseq.hparams.img_size)

img = Image.open('/path/to/image.png').convert('RGB')
# Preprocess. Model expects a batch of images with shape: (B, C, H, W)
img = img_transform(img).unsqueeze(0)

logits = parseq(img)
logits.shape  # torch.Size([1, 26, 95]), 94 characters + [EOS] symbol

# Greedy decoding
pred = logits.softmax(-1)
label, confidence = parseq.tokenizer.decode(pred)
print('Decoded label = {}'.format(label[0]))
```

## Frequently Asked Questions
- How do I train on a new language? See Issues [#5](https://github.com/baudm/parseq/issues/5) and [#9](https://github.com/baudm/parseq/issues/9).
- Can you export to TorchScript or ONNX? Yes, see Issue [#12](https://github.com/baudm/parseq/issues/12#issuecomment-1267842315).
- How do I test on my own dataset? See Issue [#27](https://github.com/baudm/parseq/issues/27).
- How do I finetune and/or create a custom dataset? See Issue [#7](https://github.com/baudm/parseq/issues/7).
- What is `val_NED`? See Issue [#10](https://github.com/baudm/parseq/issues/10).

## Training
The training script can train any supported model. You can override any configuration using the command line. Please refer to [Hydra](https://hydra.cc) docs for more info about the syntax. Use `./train.py --help` to see the default configuration.

<details><summary>Sample commands for different training configurations</summary><p>

### Finetune using pretrained weights
```bash
./train.py pretrained=parseq-tiny  # Not all experiments have pretrained weights
```

### Train a model variant/preconfigured experiment
The base model configurations are in `configs/model/`, while variations are stored in `configs/experiment/`.
```bash
./train.py +experiment=parseq-tiny  # Some examples: abinet-sv, trbc
```

### Specify the character set for training
```bash
./train.py charset=94_full  # Other options: 36_lowercase or 62_mixed-case. See configs/charset/
```

### Specify the training dataset
```bash
./train.py dataset=real  # Other option: synth. See configs/dataset/
```

### Change general model training parameters
```bash
./train.py model.img_size=[32, 128] model.max_label_length=25 model.batch_size=384
```

### Change data-related training parameters
```bash
./train.py data.root_dir=data data.num_workers=2 data.augment=true
```

### Change `pytorch_lightning.Trainer` parameters
```bash
./train.py trainer.max_epochs=20 trainer.accelerator=gpu trainer.devices=2
```
Note that you can pass any [Trainer parameter](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html),
you just need to prefix it with `+` if it is not originally specified in `configs/main.yaml`.

### Resume training from checkpoint (experimental)
```bash
./train.py +experiment=<model_exp> ckpt_path=outputs/<model>/<timestamp>/checkpoints/<checkpoint>.ckpt
```

</p></details>

## Evaluation
The test script, ```test.py```, can be used to evaluate any model trained with this project. For more info, see ```./test.py --help```.

PARSeq runtime parameters can be passed using the format `param:type=value`. For example, PARSeq NAR decoding can be invoked via `./test.py parseq.ckpt refine_iters:int=2 decode_ar:bool=false`.

<details><summary>Sample commands for reproducing results</summary><p>

### Lowercase alphanumeric comparison on benchmark datasets (Table 6)
```bash
./test.py outputs/<model>/<timestamp>/checkpoints/last.ckpt  # or use the released weights: ./test.py pretrained=parseq
```
**Sample output:**
| Dataset   | # samples | Accuracy | 1 - NED | Confidence | Label Length |
|:---------:|----------:|---------:|--------:|-----------:|-------------:|
| IIIT5k    |      3000 |    99.00 |   99.79 |      97.09 |         5.09 |
| SVT       |       647 |    97.84 |   99.54 |      95.87 |         5.86 |
| IC13_1015 |      1015 |    98.13 |   99.43 |      97.19 |         5.31 |
| IC15_2077 |      2077 |    89.22 |   96.43 |      91.91 |         5.33 |
| SVTP      |       645 |    96.90 |   99.36 |      94.37 |         5.86 |
| CUTE80    |       288 |    98.61 |   99.80 |      96.43 |         5.53 |
| **Combined** | **7672** | **95.95** | **98.78** | **95.34** | **5.33** |
--------------------------------------------------------------------------

### Benchmark using different evaluation character sets (Table 4)
```bash
./test.py outputs/<model>/<timestamp>/checkpoints/last.ckpt  # lowercase alphanumeric (36-character set)
./test.py outputs/<model>/<timestamp>/checkpoints/last.ckpt --cased  # mixed-case alphanumeric (62-character set)
./test.py outputs/<model>/<timestamp>/checkpoints/last.ckpt --cased --punctuation  # mixed-case alphanumeric + punctuation (94-character set)
```

### Lowercase alphanumeric comparison on more challenging datasets (Table 5)
```bash
./test.py outputs/<model>/<timestamp>/checkpoints/last.ckpt --new
```

### Benchmark Model Compute Requirements (Figure 5)
```bash
./bench.py model=parseq model.decode_ar=false model.refine_iters=3
<torch.utils.benchmark.utils.common.Measurement object at 0x7f8fcae67ee0>
model(x)
  Median: 14.87 ms
  IQR:    0.33 ms (14.78 to 15.12)
  7 measurements, 10 runs per measurement, 1 thread
| module                | #parameters   | #flops   | #activations   |
|:----------------------|:--------------|:---------|:---------------|
| model                 | 23.833M       | 3.255G   | 8.214M         |
|  encoder              |  21.381M      |  2.88G   |  7.127M        |
|  decoder              |  2.368M       |  0.371G  |  1.078M        |
|  head                 |  36.575K      |  3.794M  |  9.88K         |
|  text_embed.embedding |  37.248K      |  0       |  0             |
```

### Latency Measurements vs Output Label Length (Appendix I)
```bash
./bench.py model=parseq model.decode_ar=false model.refine_iters=3 +range=true
```

### Orientation robustness benchmark (Appendix J)
```bash
./test.py outputs/<model>/<timestamp>/checkpoints/last.ckpt --cased --punctuation  # no rotation
./test.py outputs/<model>/<timestamp>/checkpoints/last.ckpt --cased --punctuation --rotation 90
./test.py outputs/<model>/<timestamp>/checkpoints/last.ckpt --cased --punctuation --rotation 180
./test.py outputs/<model>/<timestamp>/checkpoints/last.ckpt --cased --punctuation --rotation 270
```

### Using trained models to read text from images (Appendix L)
```bash
./read.py outputs/<model>/<timestamp>/checkpoints/last.ckpt --images demo_images/*  # Or use ./read.py pretrained=parseq
Additional keyword arguments: {}
demo_images/art-01107.jpg: CHEWBACCA
demo_images/coco-1166773.jpg: Chevrol
demo_images/cute-184.jpg: SALMON
demo_images/ic13_word_256.png: Verbandsteffe
demo_images/ic15_word_26.png: Kaopa
demo_images/uber-27491.jpg: 3rdAve

# use NAR decoding + 2 refinement iterations for PARSeq
./read.py pretrained=parseq refine_iters:int=2 decode_ar:bool=false --images demo_images/*
```
</p></details>

## Tuning

We use [Ray Tune](https://www.ray.io/ray-tune) for automated parameter tuning of the learning rate. See `./tune.py --help`. Extend `tune.py` to support tuning of other hyperparameters.
```bash
./tune.py tune.num_samples=20  # find optimum LR for PARSeq's default config using 20 trials
./tune.py +experiment=tune_abinet-lm  # find the optimum learning rate for ABINet's language model
```

## Citation
```bibtex
@InProceedings{bautista2022parseq,
  title={Scene Text Recognition with Permuted Autoregressive Sequence Models},
  author={Bautista, Darwin and Atienza, Rowel},
  booktitle={European Conference on Computer Vision},
  pages={178--196},
  month={10},
  year={2022},
  publisher={Springer Nature Switzerland},
  address={Cham},
  doi={10.1007/978-3-031-19815-1_11},
  url={https://doi.org/10.1007/978-3-031-19815-1_11}
}
```
