# kaggle competition project: digit recognizer
this is a kaggle competition practice.

keywords: kaggle digit recognizer mnist tensorflow keras estimator cnn data augmentation tensorboard

## 1. what matters

 (1) more and more data, duh

 (2) good cnn model and learning rate

 (3) data augmentation

 (4) ensemble learning

## 2. notes for versions


(1) keras v1 model is a working cnn model with data augmentation that gives acc 0.99 ish. 

(2) keras V2 model implements ensemble learning method. This increases kaggle score from 0.99457 to 0.99628, ranking from top 22% to top 11%

(3) estimator v1 is modified on a job from gcp course. it has the same cnn model as keras version. a custom estimator is created. model trained locally.

(4) estimator v2 is simplified from v1 and transfrom the complied keras model to estimator instead of createing a custom one. model trained locally. acc = 0.990 epochs =20

(5) estimator v3 is a completed version for local train, eval and predict. decay learning rate and data augmentation are implemented. TensorBoard is also activiated too monitor progress. the prediction results in similar kaggle score. acc=0.994

