# FLyer Reconstruction Report

## What was reconstructed

The paper describes a four-tier edge/cloud architecture in which four edge servers train local LSTM models, encrypt model updates with AES-256, and send them to a cloud server for weighted Federated Averaging. Local data are retained at the edge, while the cloud returns the aggregated model. The paper also states that local data storage uses AES-256 and RSA. The repository implements those components as a single-process simulation with explicit encrypted serialization, preserving the algorithmic behavior while avoiding a requirement for four physical machines and a TCP deployment.

## Paper-to-code mapping

| Paper component | Repository implementation |
|---|---|
| Four edge servers | `load_federated_data(..., num_clients=4)` |
| Local LSTM and global LSTM | `FLyerLSTM` in `flyer/model.py` |
| One 64-unit LSTM layer | `nn.LSTM(..., hidden_size=64)` |
| Two 64-unit dense layers | `dense1` and `dense2` |
| Sparse categorical cross-entropy | PyTorch `CrossEntropyLoss` with integer class labels |
| Local training | `FederatedTrainer._train_one` |
| Random client subset | `client_fraction` and per-round sampling |
| Weighted FedAvg | Sample-count weighted state-dict aggregation |
| AES-256 gradient/update encryption | `UpdateCipher` using AES-256-GCM |
| RSA + AES local storage encryption | `LocalStorageCipher` using RSA-OAEP and AES-256-GCM |
| Accuracy, precision, recall, F1 | `FederatedTrainer.evaluate` |
| Confusion matrix | Included in `runs/*/metrics.json` |

## Dataset interpretation

The paper links a Western Maharashtra crop-and-fertilizer recommendation dataset containing 4,513 samples. The public dataset page describes a classification task with a `Crop` field, not a continuous yield target. Therefore, the implementation faithfully reproduces the paper’s actual reported experimental task as crop classification. It does not silently convert the dataset into a yield-regression problem.

## Validation

The package compiled successfully, and all three regression tests passed. An end-to-end run was completed against the public paper-linked CSV with two federated rounds and one local epoch. The run produced model weights, preprocessing metadata, round history, cryptographic audit records, and evaluation metrics under `runs/validation/`. The short smoke run achieved 19.49% accuracy after two rounds; this is expected not to match the paper’s reported >99% result because the paper omits critical choices such as exact split construction, sequence encoding, optimizer settings, epoch count, and possibly preprocessing/label handling. The result is intentionally reported rather than hidden.

## Limitations

The authors’ original source code is not included with the paper, so “exact code base” is not technically recoverable. The implementation is a transparent, runnable reconstruction. It does not claim that AES encryption alone prevents model-inference attacks, and it does not implement differential privacy, secure aggregation, or a formal gradient leakage metric. Network latency and energy are recorded only for the local simulated encryption/training path; reproducing the paper’s hardware-specific latency and energy plots would require the original testbed and measurement scripts.

## Source

The primary source is the supplied paper, “FLyer: Federated Learning-Based Crop Yield Prediction for Agriculture 5.0,” IEEE Transactions on Artificial Intelligence, 2025, DOI `10.1109/TAI.2025.3534149`. The linked dataset is available at <https://www.kaggle.com/datasets/sanchitagholap/crop-and-fertilizer-dataset-for-westernmaharashtra>.
