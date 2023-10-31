# Prototype-based Feature Selection with the Nafes Package
[Nana A. Otoo](https://github.com/naotoo1)

This repository contains the code for the paper Prototype-based Feature Selection with the sofes Package (under construction)


## Abstract
This paper presents a soft prototype-based feature selection package designed as a wrapper
centered on the highly interpretable Matrix Robust Soft Learning Vector Quantization
(MRSLVQ) classification algorithm and its local variant (LMRSLVQ).The determination of feature relevance using sofes follows similar analogy enshrined in the Nafes package with the only discrepancy existing in the utilization of prototype-based induction learners inspired by a probabilistic framework. 
[https://vixra.org/abs/2308.0112](https://vixra.org/abs/2308.0112)


The implementation requires Python 3.11.5 and above. The author recommends to use a virtual environment or Docker image.
The details of the implementation and results evaluation can be found in the paper.

To install the Python requirements use the following command:

```python
pip install -r requirements.txt 
```

To replicate results for WDBC in the paper run the default parameters:

```python
python train.py --dataset wdbc --model gmlvq --eval_type ho
python train.py --dataset wdbc --model gmlvq --eval_type mv
python train.py --dataset wdbc --model lgmlvq --eval_type ho --reject_option
python train.py --dataset wdbc --model lgmlvq --eval_type mv --reject_option

```

To replicate results for Ozone Layer in the paper run the default parameter:
```python
python train.py --dataset ozone --model gmlvq --eval_type ho
python train.py --dataset ozone --model gmlvq --eval_type mv
python train.py --dataset ozone --model lgmlvq --eval_type ho --reject_option
python train.py --dataset ozone --model lgmlvq --eval_type mv --reject_option

```
 

