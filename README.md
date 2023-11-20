# Prototype-Based Soft Feature Selection Package
[Nana A. Otoo](https://github.com/naotoo1)

This repository contains the code for the paper Prototype-based Feature Selection with the [Sofes](https://pypi.org/project/sofes/) Package


## Abstract
This paper presents a prototype-based soft feature selection package ([Sofes](https://pypi.org/project/sofes/)) wrapped around the
highly interpretable Matrix Robust Soft Learning Vector Quantization (MRSLVQ) and the Local
MRSLVQ algorithms. The process of assessing feature relevance with Sofes aligns with a comparable
approach established in the Nafes package, with the primary distinction being the utilization of
prototype-based induction learners influenced by a probabilistic framework. The numerical evaluation
of test results aligns Sofes’ performance with that of the Nafes package.
[https://vixra.org/abs/2308.0112](https://vixra.org/abs/2311.0089)



The implementation requires Python >=3.6 . The author recommends to use a virtual environment or Docker image.
The details of the implementation and results evaluation can be found in the paper.

To install the Python requirements use the following command:

```python
pip install -r requirements.txt 
```

To replicate results for WDBC in the paper run the default parameters:

```python
python train.py --dataset wdbc --model mrslvq --eval_type ho
python train.py --dataset wdbc --model mrslvq --eval_type mv
python train.py --dataset wdbc --model lmrslvq --eval_type ho --reject_option
python train.py --dataset wdbc --model lmrslvq --eval_type mv --reject_option

```

To replicate results for Ozone Layer in the paper run the default parameter:
```python
python train.py --dataset ozone --model mrslvq --eval_type ho
python train.py --dataset ozone --model mrslvq --eval_type mv
python train.py --dataset ozone --model lmrslvq --eval_type ho --reject_option
python train.py --dataset ozone --model lmrslvq --eval_type mv --reject_option

```

```python
usage: train.py [-h] [--ppc PPC] [--dataset DATASET] [--model MODEL]
                [--sigma SIGMA] [--regularization REGULARIZATION]
                [--eval_type EVAL_TYPE] [--max_iter MAX_ITER]
                [--verbose VERBOSE] [--significance] [--norm_ord NORM_ORD]
                [--evaluation_metric EVALUATION_METRIC]
                [--perturbation_ratio PERTURBATION_RATIO]
                [--termination TERMINATION]
                [--perturbation_distribution PERTURBATION_DISTRIBUTION]
                [--reject_option] [--epsilon EPSILON]
```

 

