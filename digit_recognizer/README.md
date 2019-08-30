# kaggle competition project: digit recognizer
this is a kaggle competition practice.

tags: kaggle, digit recognizer, mnist, tensorflow, keras, estimator, cnn, decay learning rate, data augmentation, tensorboard, google cloud, ml-engline, ai-platform, submit, train 

## 1. what matters

 (1) more and more data, duh

 (2) good cnn model and learning rate

 (3) data augmentation

 (4) ensemble learning

## 2. notes for versions


(1) keras v1 model is a working cnn model with data augmentation that gives acc 0.99 ish. 

(2) keras V2 model implements ensemble learning method. This increases kaggle score from 0.99457 to 0.99628, ranking from top 22% to top 11%

(3) estimator v3 is a completed version for local train, eval and predict. decay learning rate and data augmentation are implemented. TensorBoard is also activiated too monitor progress. the prediction results in similar kaggle score. acc=0.994

## 3. next steps

(1) run-on-google-cloud-platform version is coming. althought this project is small enough to run on local machine, it's also a good practice to use it for gcp

https://github.com/frh2020/kaggle/tree/master/digit_recognizer/run_on_gcp

(2) there is a better way to implement data augmentation by using iterator.

(3) carefully upgrade codes to tf 2.0 
