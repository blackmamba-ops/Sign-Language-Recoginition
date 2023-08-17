# Sign-Language-Recoginition

**sign-language-gesture-recognition-from-video-sequences**

# Sign-Language-Recognition

## Description
Welcome to the Sign Language Gesture Recognition project! This project focuses on recognizing sign language gestures from video sequences using a combination of RNN and CNN models.

## Updates
- Code has been cleaned up and made more understandable.
- Command line arguments have been added for more flexible usage.
- Bugs related to changes in model operations' names have been fixed.
- Successfully tested the code on a dummy dataset of three classes using Google Colab.

## Citation
If you find this project useful, please consider citing it in your work.

## DataSet Used
We have used the [Argentinian Sign Language Gestures dataset](https://facundoq.github.io/datasets/lsa64/). It's important to note that the dataset is made available strictly for academic purposes by the dataset owners. Make sure to carefully read and adhere to the dataset's license terms. If you plan to use the dataset, don't forget to cite their paper as well.

## Requirements
To run this project, you need the following dependencies:
- OpenCV (Follow [installation instructions](https://docs.opencv.org/trunk/d7/d9f/tutorial_linux_install.html))
- TensorFlow: `pip install tensorflow`
- TFLearn: `pip install tflearn`

## Training and Testing

### 1. Data Preparation
Create two folders: `train_videos` and `test_videos` in the project root directory. Each of these folders should contain subfolders for each gesture category, and within these subfolders, you should place the corresponding videos.

### 2. Extracting Frames
Use the `video-to-frame.py` script to extract frames from gesture videos.
- For training videos: `python3 video-to-frame.py train_videos train_frames`
- For test videos: `python3 video-to-frame.py test_videos test_frames`

### 3. Retrain the Inception v3 model
- Download `retrain.py`: `curl -LO https://github.com/tensorflow/hub/raw/master/examples/image_retraining/retrain.py`
- Retrain the model using the extracted frames:

### 4. Intermediate Representation of Videos
Use the `predict_spatial.py` script to generate intermediate representations of videos.
- Approach 1 (softmax-based representation):
- Train: `python3 predict_spatial.py retrained_graph.pb train_frames --batch=100`
- Test: `python3 predict_spatial.py retrained_graph.pb test_frames --batch=100 --test`
- Approach 2 (pool layer-based representation):
- Train: `python3 predict_spatial.py retrained_graph.pb train_frames --output_layer="module_apply_default/InceptionV3/Logits/GlobalPool" --batch=100`
- Test: `python3 predict_spatial.py retrained_graph.pb test_frames --output_layer="module_apply_default/InceptionV3/Logits/GlobalPool" --batch=100 --test`

### 5. Train the RNN
Use the `rnn_train.py` script to train the RNN model.
- Approach 1: `python3 rnn_train.py predicted-frames-final_result-train.pkl non_pool.model`
- Approach 2: `python3 rnn_train.py predicted-frames-GlobalPool-train.pkl pool.model`

### 6. Test the RNN Model
Use the `rnn_eval.py` script to test the trained RNN model.
- Approach 1: `python3 rnn_eval.py predicted-frames-final_result-test.pkl non_pool.model`
- Approach 2: `python3 rnn_eval.py predicted-frames-GlobalPool-test.pkl pool.model`

Enjoy building and testing your Sign Language Gesture Recognition model! 😄🤟
