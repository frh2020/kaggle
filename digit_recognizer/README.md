# Multi-class image classification: digit recognizer
This is a kaggle competition practice.

tags: kaggle, digit recognizer, mnist, tensorflow, keras, estimator, cnn, decay learning rate, data augmentation, tensorboard, google cloud, ml-engline, ai-platform, submit, train 

## 1. What matters

 (1) More and more data, duh

 (2) Good cnn model and learning rate

 (3) Data augmentation

 (4) Ensemble learning

## 2. Notes for versions


(1) Keras v1 model is a working cnn model with data augmentation that gives acc 0.99-ish. 

(2) Keras v2 model implements ensemble learning method. This increases kaggle score from 0.99457 to 0.99628, ranking from top 22% to top 11%.

(3) Estimator v5 is a completed version for local train, eval and predict. decay learning rate and data augmentation are implemented without tf.contrib.layers which is deprecated in tf2.0. a key is added to input format. TensorBoard is also activiated too monitor progress. the prediction results in similar kaggle score. acc=0.994

(4) Keras v1 codes are converted to tf 2.0. kaggle-digit-cnn-keras-tf2-v1.ipynb

## 3. Run on Cloud

Run-on-google-cloud-platform package is built from estimator v5. althought this project is small enough to run on local machine, it's also a good practice to use it for gcp. online prediction and batch prediction are both tested out on the deployed model. it generated prediction with similar kaggle score as the local versions.

https://github.com/frh2020/kaggle/blob/master/digit_recognizer/run_on_gcp/README.md
