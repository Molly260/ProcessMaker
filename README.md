# <div align="center" >ProcessMaker: A Generalized Process Visualization Framework with Adaptive Sequence Steps on Diffusion Transformers <div align="center">

## <div align="center">🎉🎉 **CVPR 2026** 🎉🎉</div>

<div align="center">
  <p>
    <a href="">Mengling Xu</a><sup>1</sup>
    <a href="">Sisi You</a><sup>1</sup>
    <a href="">Yaning Li</a><sup>1</sup>
    <a href="https://www.scholat.com/bkbao.en">Bing-Kun Bao</a><sup>1,2,✉</sup>
  </p>
  <p>
    <sup>1</sup>Nanjing University of Posts and Telecommunications &nbsp;&nbsp;
    <sup>2</sup>Peng Cheng Laboratory &nbsp;&nbsp;
    <sup>✉</sup>Corresponding author
  </p>
</div>

## ⚙️ Setup
####  Step 1: Environment setup

```python
cd ProcessMaker
conda create -n processmaker python=3.11
conda activate processmaker

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install --upgrade -r requirements.txt

accelerate config


```
#### Step 2: Download the pretrained checkpoint of Flux.1


```bash
hf download Comfy-Org/flux1-dev --local_dir ./ckpt/
```

#### Step 3: Prepare the datasets

Download the dataset from <a href="https://github.com/showlab/MakeAnything">MakeAnything</a>
Place the data under `./datasets`.

**Directory of training dataset:**

```
dataset/
├── portrait_001.png
├── portrait_001.caption
├── portrait_002.png
├── portrait_002.caption
├── lego_001.png
├── lego_001.caption
```


#### Step 4: Train
**Train Stage 1**
```bash
bash scripts/train_stage1.sh
```
**Train Stage 2**
```bash
bash scripts/train_stage2.sh
```

#### Step 5: Inference
**Inference Stage 1**
```bash
bash scripts/sample_stage1.sh
```
**Inference Stage 2**

```bash
bash scripts/sample_stage2.sh
```

## Acknowledgement
We would like to express our gratitude to the MakeAnything and Flux.1 for their codes. Their contributions have been instrumental to the development of this project.

## 📚 Citation
Will coming soon
