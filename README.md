TensorFlow2.0-from-zero-to-advanced
====

This repo is the excises of course `TensorFlow2.0-from-zero-to-advanced`. Thanks for the tutor of this course.

The outline of this course and excises lists:




### Chapter01 Tensorflow Introduction and Environment Setup
  1. What is Tensorflow?
  2. Tensorflow version changes and tf1.0 architecture
  3. Tensorflow2.0 architecture
  4. Tensorflow VS pytorch
  5. Tensorflow environment configuration
  6. Google_cloud without GPU environment
  7. Google_cloud_Remote jupyter_notebook configuration
  8. Google_cloud_gpu_tensorflow configuration
  9. Google_cloud_gpu_tensorflow mirror configuration
  10. AWS cloud platform environment configuration
### Chapter02 Tensorflow `keras` Excises
  1. tfkeras brief introduction
  2. Classification regression and objective function
  3. Data reading and display of actual classification model
  4. Model Construction of Classification Model
  5. Data normalization of classification model
  6. Callbacks
  7. Regression model
  8. Neural netorks
  9. Deep neural network
  10. Batch normalization, activation function, dropout
  11. wide_deep model
  12. Function API to implement wide & deep model
  13. Subclass API to implement wide & deep model
  14. Multi-input and multi-output excises of wide & deep model
  15. Hyperparameter search
  16. Manual implementation of hyperparameter search
  17. sklearn package keras model
  18. sklearn hyperparameter search
### Chapter03 Tensorflow `Basic API` Usage
  1. `tf.constant`
  2. `tf.strings` and `ragged tensor`
  3. `sparse tensor` and `tf.Variable`
  4. self define `loss function` and `DenseLayer`
  5. Use `subclasses` and `lambdas` to define levels separately
  6. `tf.function` function conversion
  7. `@tf.function` function conversion
  8. Function signature and graph structure
  9. Approximate derivative
  10. `tf.GradientTape` Basic Usage
  11. `tf.GradientTape`与`tf.keras`结合使用
### Chapter04 Tensorflow `dataset` Usage
  1. `tf_data` basic API usage
  2. Generate csv file
  3. `tf.io.decode_csv` usage
  4. `tf.data` reads a csv file and uses it with `tf.keras`
  5. `tfrecord` basic API usage
  6. Generate `tfrecords` file
  7. `tf.data` reads tfrecord file and uses it with tf.keras
### Chapter05 Tensorflow `Estimator` Usage and  tf1.0
  1. Titanic problem analysis
  2. `feature_column` usage
  3. `keras_to_estimator`
  4. Predefined estimator usage
  5. Cross feature excises
  6. TF1.0 computational graph construction
  7. TF1.0 model training
  8. TF1_dataset usage
  9. TF1 self defined estimator
### Chapter06 Convolutional Neural Network
  1. Problems solved by convolution
  2. Calculation of convolution
  3. Pooling operation
  4. CNN excises
  5. Deep separable convolutional network
  6. Deep separable convolutional network excises
  7. 10monkeys dataset
  8. Keras generator reads data
  9. `10monkeys` basic model building and training
  10. 10monkeys model fine-tuning
  11. Keras generator reads `cifar10` dataset
  12. Model training and prediction
### Chapter07 Recurrent Neural Network
  1. RNN and embedding
  2. Data set loading and construction of vocabulary index
  3. Data padding, model construction and training
  4. Sequential problems and recurrent neural networks
  5. Text Classification by RNN
  6. Data processing for text generation
  7. Model construction for text generation
  8. Sample text for text generation
  9. LSTM
  10. Text classification and text generation by LSTM
  11. Dataset loading and tokenizer for subword text classification
  12. Dataset transformation and model training for subword text classification
### Chapter08 Tensorflow Distribution
  1. GPU settings
  2. GPU default settings
  3. Memory growth and virtual device excises
  4. GPU manual settings excises
  5. Distribution strategy
  6. Keras distribution excises
  7. Estimator distribution excises
  8. Self define process excises
  9. Distributed self define process excises
### Chapter09 Tensorflow Model Saving and Deployment
  1. `TFLite_x264`
  2. Save model structure plus parameters and save parameters in practice
  3. Keras model convert to `SavedModel`
  4. `Signature function` convert to `SavedModel`
  5. Signature function, SavedModel and Keras model convert to concrete function
  6. `tflite` preservation and interpretation and quantification
  7. `tensorflowjs` convert model
  8. `tensorflowjs` build server and load model
  9. `Android` deployment model
### Chapter10 Machine Translation
  1. `seq2seq+attention` model
  2. Data preprocessing and reading
  3. Convert string to id and dataset generation
  4. Build Encoder
  5. Build attention
  6. Build decoder
  7. Loss function and one step trainning
  8. Model training
  9. Model prediction
  10. `Transformer`
  11. Encoder-Decoder with Zoom click attention
  12. Multi head attention with position encoding
  13. Data preprocessing and dataset generation
  14. Position encoding
  15. Build mask
  16. Build Zoom click attention
  17. Build Multi head attention
  18. Feedforward layer
  19. Encoder layer
  20. Decoder layer
  21. Encoder model
  22. Decoder model
  23. Transformer
  24. Self define learning rate
  25. Model training and evaluation
