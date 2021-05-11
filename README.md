# Continual Knowledge Graph Embedding (CKGE)
Supporting code for [CKGE](https://arxiv.org/abs/2101.05850) paper. This repo contains three CKGE benchmark datasets as well as scripts to auto-generate new datasets from
existing knowledge graphs.

## Pre-requisites
1. This repo has been tested for a system running Ubuntu 18.04 LTS, PyTorch (1.2.0), and 
hardware CPU or Nvidia GPU (GeForce GTX 1060 6GB or better).
2. For GPU functionality Nvidia drivers, CUDA, and cuDNN are required.

## Installation
All dependencies are installed to a virtual environment using `virtualenv` to protect your system's
current configuration. Install the virtual environemnt and dependencies by running `./setup_repo.sh`
in terminal. This script should only be executed ONCE for the life of the repo.

## Source the environment
You must source your environment each time it is deactivated. This is done via `source ./setup_env.sh`. You 
environment is sourced when `(py36_venv)` appears as the first part of the terminal prompt. You can unsource via
`deactivate`.

## Check install
After sourcing the environment, run `python`. Python version 3.6 should run. Next, check if `import torch` works.
Next, for GPU usage check if `torch.cuda.is_available()` is `True`. If all these checks passed, the installation should
be working. 

## Repo Conents
This repo contains two knowledge graph embedding models, three CKGE datasets, two learning settings, and CKGE approaches.

- Graph-embedding Models: [TrasnE](http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf)
& [Analogy](http://proceedings.mlr.press/v70/liu17d.html)
- Datasets: [WN18RR](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/download/17366/15884), 
[FB15K237](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/download/17366/15884),
[THOR](https://adaruna3.github.io/robocse/)
- Learning Settings:
    1. standard: follows precedents & assumptions from knowledge graph embedding [community](http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf).
    2. continual: implements CKGE as described in the [paper](https://arxiv.org/abs/2101.05850).
- CKGE Approaches:
    1. [Progressive Neural Networks](./models/pnn_models.py)
    1. [Copy Weight with Re-Init](./models/cwr_models.py)
    1. [L2 Regularization](./models/l2_models.py)
    1. [Synaptic Intelligence](./models/si_models.py)
    1. [Generative Replay](./models/dgr_models.py)
    
## Run the Paper Experiments
The following scripts run the experiments presented in the submission. The final results of the scripts are PDF files
containing plots from evaluations and text containing the outputs of statistical significance tests.

### Generate the CKGE datasets from the paper (triple sampling in paper appendix)
1. Run `./experiments/scripts/run_generate_dataset.sh`.
2. Output datasets will be placed in `./datasets/` directory.
3. PDFs containing the dataset statistics will be generated and placed at the root directory of the repo.

### Generate entity and relation relation sampling datasets from the appendix
1. Comment/uncomment corresponding lines from `./experiments/scripts/run_generate_dataset.sh`

### Benchmark Evaluations (training time for 5 runs is more than 3 days)
1. Train the models by running `./experiments/scripts/run_continual_setting_experiment_benchmark_train_TS.sh`.
2. Test the models, produce plots, and run statistical tests by running `./experiments/scripts/run_continual_setting_experiment_benchmark_test_TS.sh`.
3. Output is a PDF in the root directory of the repo and several text files with outputs of the statistical tests.
4. Similarly for entity or relation sampling, run 1-3 as above but using either corresponding `*_ES.sh` or `*_RS.sh` files, respectively.

### Unconstrained Robot Evaluation Setting (training time for 5 runs is ~1.5 days)
1. Train the models by running `./experiments/scripts/run_continual_setting_experiment_robot_train_uncon.sh`.
2. Test the models, produce plots, and run statistical tests by running `./experiments/scripts/run_continual_setting_experiment_robot_test_uncon.sh`.
3. Output is a PDF in the root directory of the repo and several text files with outputs of the statistical tests.

### Data Constrained Robot Evaluation Setting (training time for 5 runs is ~1.5 days)
1. Train the models by running `./experiments/scripts/run_continual_setting_experiment_robot_train_dcon.sh`.
2. Test the models, produce plots, and run statistical tests by running `./experiments/scripts/run_continual_setting_experiment_robot_test_dcon.sh`.
3. Output is a PDF in the root directory of the repo and several text files with outputs of the statistical tests.

### Data & Time Constrained Robot Evaluation Setting (training time for 5 runs is <1 day)
1. Train the models by running `./experiments/scripts/run_continual_setting_experiment_robot_train_dtcon.sh`.
2. Test the models, produce plots, and run statistical tests by running `./experiments/scripts/run_continual_setting_experiment_robot_test_dtcon.sh`.
3. Output is a PDF in the root directory of the repo and several text files with outputs of the statistical tests.

* After beginning any training program, you can check the progress of your training session by starting tensorboard in 
another terminal via `tensorboard --logdir=logger`. Remember to source the environment. As training progresses and the 
model achieves new best performance levels, model checkpooints are saved to `./models/checkpoints`.

* Note that in the Data & Time Constrained Robot Evaluation setting, DGR sometimes does not save checkpoints model 
because performance does not improve. This will cause errors in evaluation. The issue is fixed by replacing line 381 
of [model_utils.py](./models/model_utils.py) with `if True:`.

## Hyper-paramter Tuning:
We use Adagrad SGD to train the knowledge graph embeddings (TransE and Analogy). We tune all the hyper-parameters of 
knowledge graph embeddings simultaneously using grid search with the original knowledge graphs (WN18RR, FB15K237, 
AI2Thor). For Analogy, we tune the learning rate {0.1,0.01,0.001}, negative sampling ratio {1,25,50,100}, and 
embedding hidden size dimensions (d_E/d_R) {25,50,100,200}. For TransE we also tune the hyper-parameter margin (gammea) 
{2,4,8}. The hyper-parameter settings and performance on the original knowledge graphs are shown below.

|  Dataset |  Model  | Embedding Hidden Dim | Negative Sampling Ratio | Learning Rate | Margin | MRR | Hits@10 |
|:--------:|:-------:|:--------------------:|:-----------------------:|:-------------:|:------:|:---:|:-------:|
|  WN18RR  |  TransE |          100         |            25           |      0.1      |   8.0  |  23 |    48   |
|  WN18RR  | Analogy |          200         |            1            |      0.01     |    -   |  41 |    46   |
| FB15K237 |  Transe |          200         |            1            |      0.01     |   8.0  |  24 |    39   |
| FB15K237 | Analogy |          200         |            25           |      0.01     |    -   |  26 |    41   |
|  AI2Thor |  Transe |          25          |            1            |      0.1      |   2.0  |  61 |    85   |
|  AI2Thor | Analogy |          100         |            50           |      0.1      |    -   |  66 |    88   |

We also tune the CKGE methods' hyper-parameters. The regularization strength scaling term (lambda) 
{0.0001, 0.001, 0.01, 0.1, 1, 10, 100} is set using grid search for L2R and SI. The settings are shown below.

|  Dataset |  Model  | L2 - Lambda | SI - Lambda |
|:--------:|:-------:|:-----------:|:-----------:|
|  WN18RR  |  TransE |     1.0     |     0.1     |
|  WN18RR  | Analogy |     0.1     |     1.0     |
| FB15K237 |  TransE |     10.0    |     1.0     |
| FB15K237 | Analogy |    0.0001   |    0.0001   |
|  AI2Thor |  TransE |     1.0     |     0.01    |
|  AI2Thor | Analogy |     10.0    |     1.0     |

Due to the large number of hyper-parameters in VAE architecture for DGR, the relevant parameters are tuned manually.
These included embedding layer dimensions, hidden layer dimensions, latent layer dimensions, anneal slope (lambda_as), 
anneal max (lambda_am), and anneal position (lambda_ap). The used settings are provided below.

|  Dataset | Embedding Layer Dim | Hidden Layer Dim | Latent Layer Dim | Anneal Slope | Anneal Max | Anneal Position |
|:--------:|:-------------------:|:----------------:|:----------------:|:------------:|:----------:|:---------------:|
|  WN18RR  |         200         |        150       |        100       |     0.06     |     0.8    |      200.0      |
| FB15K237 |         200         |        150       |        100       |     0.06     |     0.8    |      200.0      |
|  AI2Thor |         100         |        75        |        50        |     0.06     |     0.8    |      200.0      |
