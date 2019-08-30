run-on-gcp version

tags: kaggle, digit recognizer, mnist, tensorflow, keras, estimator, cnn, decay learning rate, data augmentation, tensorboard, google cloud, ml-engline, ai-platform, submit, train 

gcp guide 

https://cloud.google.com/ml-engine/docs/tensorflow/getting-started-training-prediction#tensorboard-local

project structure

<img src="recommended-project-structure.png">

modifications from local machine version:

(1) add REQUIRED_PACKAGE= ['GCSFS' ] in setup.py

(2) export model at the end, instead of each epoch.

(3) launch TensorBoard in cloud shell on gcp console.

