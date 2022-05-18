# MHN
This is the PyTorch Implementation of our paper "[Multilevel Hierarchical Network with Multiscale Sampling for Video Question Answering](https://arxiv.org/abs/2205.04061)". (accepted by IJCAI’22)

![alt text](docs/fig2.png 'overview of the network')

# Platform and dependencies
Ubuntu 14.04  Python 3.7  CUDA10.1  CuDNN7.5+  
pytorch>=1.7.0

# Data Preparation
* Download the dataset  
  MSVD-QA: [link](https://github.com/xudejing/video-question-answering)   
  MSRVTT-QA: [link](https://github.com/xudejing/video-question-answering)   
  TGIF-QA: [link](https://github.com/YunseokJANG/tgif-qa)   
* Preprocessing
  1. To extract questions or answers Glove Embedding, please ref [here](https://github.com/thaolmk54/hcrn-videoqa).  
  Take the action task in TGIF-QA dataset as an example, we have features at the path /QAfeatures:
  TGIF/word/action/TGIF_action_train_questions.pt
  TGIF/word/action/TGIF_action_val_questions.pt
  TGIF/word/action/TGIF_action_test_questions.pt
  TGIF/word/action/TGIF_action_vocab.json
  2. To extract appearance and motion feature, use the pretrained models [here](https://drive.google.com/open?id=1xbYbZ7rpyjftI_KCk6YuL-XrfQDz7Yd4).  
  for the action task, we have features at the path /Vfeatures:  
  `TGIF/SpatialFeatures/action/video0/Features.pkl` (shape is 2^level-1,16,2048)  
  `TGIF/SpatialFeatures/action/video1/Features.pkl`  
  `...`  
  `TGIF/TemporalFeatures/action/video0/Features.pkl` (shape is 2^level-1,2048)  
  `TGIF/TemporalFeatures/action/video1/Features.pkl`  
  `...`  
  In our paper, level is set to 3 by default.
  
# train and test
run `python train.py` in your terminal.  The trained models for the action task can be download in [here](https:).  
