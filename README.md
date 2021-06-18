# ml
My ML Experiments

Deep Learning Models are built with either Pytorch or Keras (Tensorflow)

Machine Learning Models are built with Scikit-Learn.

Most build the model from scratch.

A few use open-source model implementations with pre-trained parameters.

All of these use popular publicly available datasets, including several from Kaggle competitions.

Discriminative Learning Rates. Tensorboard integration. Kaggle Dataset.

### Computer Vision
- [Pointnet 3D](apps/3D%20Objects%20Pointnet.ipynb) (Pytorch): Build a Pointnet model to classify 3D objects. Source data in Mesh format is converted to Point Cloud. Interactive visualization with Plotly.
- [Facial Keypoints](apps/Facial%20Keypoints.ipynb) (Pytorch): Build a CNN architecture and process video files to identify facial keypoints. Can also input images.
- [Image Caption](apps/Image%20Caption.ipynb) (Keras): Build three different architectures to generate Image Captions - Multi-modal CNN + RNN network, Encoder-Decoder with Attention and Encoder-Decoder with Transformers. Use Transfer Learning to encode images.
- [Image Classification](apps/Image%20Classification.ipynb) (Pytorch): Build an enhanced ResNet model (Resnet18 through Resnet152) to classify images. The architecture is based on Amazon's "[Bag of Tricks](https://arxiv.org/pdf/1812.01187.pdf)" paper.
- [Image OCR](apps/Image%20OCR.ipynb) (Keras): Build CNN and RNN model for OCR of text content in images using CTC algorithm and Beam Search.
- [Image Segmentation Mask R-CNN](apps/Image%20Segmentation%20MaskRCNN.ipynb) (Pytorch): Use open-source Mask R-CNN implementation with pre-trained model. Process video files for semantic segmentation.
- [Image Segmentation](apps/Image%20Segmentation.ipynb) (Pytorch): Built a UNet architecture for semantic segmentation of images taken from Kaggle Carvana dataset.
- [Object Detection Retinanet](apps/Object%20Detection%20Retinanet.ipynb)(Pytorch): Build Facebook's state-of-the-art Retinanet architecture with a Feature Pyramid and FocalLoss, for doing Object Detection with Pascal VOC dataset.
- [Object Tracking Deepsort](apps/Object%20Tracking%20DeepSort.ipynb) (NA): Detection and Tracking Objects in motion in a video using open source implementation of Deepsort algorithm and pre-trained YOLO3 model.
- [Pose Estimation](apps/Pose%20Estimation%20Mask%20R-CNN%20with%20TF%201.ipynb) (Keras): Pose Estimation in images and video based on open source Mask R-CNN implementation (with added Pose Estimation layers) and pre-trained model.

### NLP
- [Sequence-to-Sequence Word](apps/Seq-to-Seq-Word.ipynb) (Keras): Build Sequence-to-Sequence Encoder-Decoder LSTM model for Machine Translation. With English-to-French dataset.
- [Sequence-to-Sequence Char](apps/Seq-to-Seq.ipynb) (Keras): Build Sequence-to-Sequence Encoder-Decoder LSTM model for character-by-character Machine Translation. With English-to-French dataset.
- [Text Generation](apps/Text%20Generation.ipynb) (Keras): Build GRU-based character-by-character Text Generation model.
- [Transformer Time Embeddings](apps/Transformer%20Time%20Embeddings.ipynb) (Keras): Convert stock price time series data into Time Embeddings using Time2Vector. Build a Transformer architecture to process the time embeddings to predict stock prices. 
- [Transformer Translation](apps/Transformer%20Translation.ipynb) (Pytorch): Build Transformer architecture from scratch for Machine Translation with French to English dataset.
- [Language Model](apps/Language%20Model.ipynb) (Keras): Build a LSTM-based Sequence-to-Sequence Language Model to predict the expected next word in a sentence.
- [NLP Transfer Learning](apps/NLP%20Transfer%20Learning%20ULMFit.ipynb) (Pytorch): NLP Transfer Learning using ULMFit (Universal Language Model Fine Tuning) technique. Build a Language Model architecture using a AWD-LSTM with several types of RNN regularization methods. Pre-train this model on a Wikitext corpus. Then fine tune it for Text Classification to do Sentiment Analysis on an IMDB movie review dataset.
- [Chatbot](apps/Chatbot%20DialogGPT.ipynb) (Hugging Face library): Dialog Chatbot using a pre-trained DialoGPT language model implementation from Hugging Face's transformers library.
- [Text Translation](apps/Attention%20Bi-directional%20LSTM.ipynb) (Pytorch): Build Sequence-to-Sequence Encoder Decoder architecture using Bidirectional LSTM with Attention, for French to English Machine Translation. Write custom Bleu Metric implementation for evaluation.

### Tabular Data
- [Tabular Home Credit](apps/Tabular%20Home%20Credit.ipynb) (Pytorch): Deep-learning model for tabular structured data to predict Loan Approvals using Kaggle Home Credit dataset. Uses Categorical Embeddings, rather than encodings. Basic Exploratory Data Analysis and Visualization with Seaborn.
- [Tabular Random Forest](apps/Tabular%20Random%20Forest.ipynb) (scikit): Predict bulldozer prices with tabular structured data using Decision Tree and Random Forest machine learning algorithms. Uses Kaggle Blue-Book for Bulldozers dataset. Selects the most relevant features based on Feature Importance.
- [Tabular Rossman](apps/Tabular%20Rossman.ipynb) (Pytorch): Deep-learning model for tabular structured data with time series for Store Sales Forecasting using Kaggle Rossman dataset. Uses Categorical Embeddings. EDA with Seaborn.

### Audio
- [Audio Classification](apps/Audio%20Classification.ipynb) (Pytorch): Build a CNN architecture to classify audio clips using Mel Spectrograms for two applications: (1) Identify voices from short utterances and (2) Classify day-to-day urban sounds
- [Speech-to-Text](apps/Speech%20To%20Text.ipynb) (Pytorch): Build Baidu Deep-Speech-2 model with combined CNN + RNN architecture for Automatic Speech Recognition. Uses CTC Loss algorithm and WER/CER metrics.

### Recommendation Systems
- [Recommendation Systems](apps/Collaborative%20Filtering.ipynb) (Pytorch): Build two deep learning architectures for Recommendation Systems for movie recommendations. One model is based on Collaborative Filtering using Matrix Factorization. The second model uses a Neural Collaborative Filtering architecture with a linear network.

### GAN
- [Cycle GAN](apps/Cycle%20GAN.ipynb) (Pytorch): Build a Cycle GAN architecture to transform images of horses into zebras (while preserving the backgrounds) and vice versa.

### Reinforcement Learning
- [Actor-Critic](apps/RL%20A3C.ipynb) (Keras): Reinforcement Learning to play a video game using A3C distributed asynchronous multi-worker algorithm
- [Deep Q Networks](apps/RL%20DQN.ipynb) (Keras): Reinforcement Learning to play a video game using Deep Q Networks
- [Q-Learning](apps/RL%20QLearning.ipynb) (Keras): Reinforcement Learning to play a simple game using Q-Learning algorithm

### Machine Learning

### Other
- [Geo Location](apps/Geo%20Location.ipynb) (Scikit): Predict trip durations using Random Forest and XGBoost algorithms for geo-location data with Kaggle NYC Taxi dataset. Location-based clusters with both K-means and probabilistic Gaussian Mixture Model. Location features with Geopandas and interactive maps with Folium.
- [Auto Encoder](apps/Auto%20Encoder.ipynb) (Keras): Build a Variational Auto Encoder model to generate images of handwritten digits, using MNIST dataset.

### Library
This contains much of the commonly-used functionality when building deep learning applications with Pytorch. It incorporates a number of useful techniques and best practices to make it easy to build and run these models with only a few lines of code.

Although the implementation is different, many of these ideas and techniques were inspired by Fastai.

Deep Learning library for advanced Pytorch models. Provides functionality for data preparation in a generic way for datasets in different input formats. Learning Rate Finder.
- [Data](lib/data_lib.ipynb) - Declaratively specify steps to prepare data from a variety of source input formats
- [Training](lib/training_lib.ipynb) - Custom Training loop with Callbacks and Schedulers. 