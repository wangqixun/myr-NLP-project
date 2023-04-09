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

Pretraining Model | Accuracy | Checkpoint
:-------------------------:|:-------------------------:|:-------------------------:
roberta |  93.04 | [link](https://pan.baidu.com/s/1WhSAp374W7bQiiY6En5mPA?pwd=7v4j):7v4j
macbert |  93.33 | [link]():xxxx
pert |  92.87 | [link]():xxxx
lert |  93.27 | [link]():xxxx


# Inference

After modifying the data path and pretraining model path, execute

```
python infer.py
```



