# MYR NLP Project

项目描述一下

# Installation Dependencies

```
pip install -r requirements.txt
```

# Data

待马老师补充

# Pretraining Model

We use 
[RoBERTa-base](https://huggingface.co/xlm-roberta-base), 
[MacBERT-base](https://huggingface.co/hfl/chinese-macbert-base), 
[PERT-base](https://huggingface.co/hfl/chinese-pert-base), 
[LERT-base](https://huggingface.co/hfl/chinese-lert-base)
as the pretraining model.


# Train

After modifying the data path and pretraining model path, execute

```
bash train.sh
```

# Result

Pretraining Model | Accuracy
:-------------------------:|:-------------------------:
roberta |  93.04
macbert |  93.33
pert |  92.87
lert |  93.27


# Inference

After modifying the data path and pretraining model path, execute

```
python infer.py
```
