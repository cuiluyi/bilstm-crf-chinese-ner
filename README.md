# BiLSTM-CRF Chinese NER

A Chinese Named Entity Recognition (NER) model implementation using BiLSTM + CRF architecture, with character-level input processing and sequence labeling optimization.

## Project Overview

This repository provides a complete implementation of a Chinese NER system:
- **Model Architecture**: Character Embedding → Bidirectional LSTM → Linear Projection → CRF
- **Labels**: Supports 7 entity types in BIO format: `O, B-PER, I-PER, B-LOC, I-LOC, B-ORG, I-ORG`
- **Training Pipeline**: End-to-end training with BPTT optimization and checkpoint management

## Model Architecture

![Model Architecture](https://tianchou.oss-cn-beijing.aliyuncs.com/img/20251109123122181.png)

### Key Components

1. **Input Layer**  
   Converts raw Chinese characters to dense vector representations via embedding layer.

2. **Bidirectional LSTM**  
   
   - Forward LSTM computes contextual features from left-to-right
   
   - Backward LSTM computes contextual features from right-to-left  
   
   - Hidden states concatenated for bidirectional context modeling:  
     $$
     P_i = \text{softmax}(\tanh(W_c [F(h_i), B(h_i)]))
     $$
     
   
3. **CRF Layer**  
   Performs sequence decoding with transition constraints:  
   
   - Sequence score calculation: 
     $$
     s(X,y) = \sum_{i=0}^{L} A_{y_i,y_{i+1}} + \sum_{i=1}^{L} P_{i,y_i}
     $$
     
   - Viterbi algorithm for optimal label sequence prediction: 
     $$
     y^* = \operatorname{argmax}_{\tilde{y} \in Y_X} s(X,\tilde{y})
     $$
   
4. **Training Objective**  
   Cross-entropy loss with joint optimization of:  
   
   - LSTM weights ($W1, U1, W2, U2$)  
   - Concatenation layer weights ($W_c$)  
   - CRF transition matrix ($A$)

## Dataset Structure

### Data Files
```
data/
├── train_corpus.txt    # Training text corpus
├── train_label.txt     # Training labels
├── train.txt           # Combined training data
├── test_corpus.txt     # Test text corpus
├── test_label.txt      # Test labels
└── test.txt            # Combined test data
```

### Vocabulary Construction
- `char2id`: Character-to-index mapping built from training/test corpus
- `tag2label`: Fixed label mapping:  
  `{"O":0, "B-PER":1, "I-PER":2, "B-LOC":3, "I-LOC":4, "B-ORG":5, "I-ORG":6}`

### Dataset Class
`NERDataset` handles data loading with:
- Line-by-line text parsing with empty-line sentence segmentation
- Direct tensor conversion without padding (`BATCH_SIZE=1`)

## Quick Start

### Environment Setup
```bash
# Clone repository
git clone git@github.com:cuiluyi/bilstm-crf-chinese-ner.git
cd bilstm-crf-chinese-ner

# Install dependencies
pip install -r requirements.txt
```

### Training
```bash
# Basic training
python train.py

# With GPU and logging (recommended)
bash scripts/train.sh
```
- Hyperparameters configured in `constants.py`
- Checkpoints saved to `ckpts/model_epoch_{n}.pth`
- Training logs written to `logs/train.log`

### Evaluation
```bash
# Evaluate specific checkpoint
python test.py --model_path ckpts/model_epoch_1.pth

# Batch evaluation
bash scripts/test.sh
```
Output metrics:
- Overall Accuracy (including "O" class)
- Accuracy (excluding "O" class for meaningful entity detection evaluation)

### Inference
```bash
# Interactive prediction
python predict.py --model_path ckpts/model_epoch_5.pth

# Run with logging
bash scripts/predict.sh
```
Enter Chinese sentences interactively to get BIO-tagged entity predictions.

## Experimental Results

| Epoch | Accuracy (Overall) | Accuracy (Ignore "O") |
|-------|--------------------|-----------------------|
| 1     | 97.1002%           | 89.1687%              |
| 2     | 97.4994%           | 87.3307%              |
| 3     | 97.6263%           | 88.1358%              |
| 4     | 97.5945%           | 88.6165%              |
| 5     | 97.3169%           | 88.0750%              |
| 6     | 97.6535%           | 87.8508%              |
| 7     | 97.5585%           | 86.8199%              |
| 8     | 97.6089%           | 88.1359%              |
| 9     | 97.3140%           | 84.5975%              |
| 10    | 97.5261%           | 86.7769%              |

### Key Observations
- Peak entity detection performance at **Epoch 1** (89.17% accuracy ignoring "O" class)
- Overall accuracy consistently high (>97%) due to dominant "O" class in dataset
- Best trade-off between overall and entity accuracy at **Epoch 6** (97.65% overall)

## Repository Structure
```
.
├── README.md           # Project documentation
├── constants.py        # Hyperparameter configuration
├── ner_dataset.py      # Dataset loading and preprocessing
├── model.py            # BiLSTM-CRF architecture implementation
├── train.py            # Training script
├── test.py             # Evaluation script
├── predict.py          # Inference script
├── data/               # Training/test data
├── ckpts/              # Model checkpoints
├── logs/               # Training/evaluation logs
└── scripts/            # Bash scripts for batch operations
```

## License
[MIT License](LICENSE)