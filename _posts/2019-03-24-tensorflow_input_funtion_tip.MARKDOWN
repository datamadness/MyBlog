---
layout: post
current: post
cover:  assets/images/input_function/Fort_Gibson_dam_2.jpg
navigation: True
title: TensorFlow Tip - input_fn with custom parameters
date: 2019-03-24 12:00:00
tags: [Python]
class: post-template
subclass: 'post tag-Python'
author: DATAmadness
mathjax: true
---

The TensorFlow Estimators API is great for quickly building your custom models, but you might have noticed there is no  obvious way to pass custom parameters from your estimator into the input_fn. That can be a limiting factor when building more complicated input functions that include a data pipeline in them([see this post](https://datamadness.github.io/tensorflow_estimator_large_dataset_feed)).
This short post offers a quick tip on how to use the pythonic lambda function to solve the problem in a single line of code.

You can get the full python example from [my GitHub repo](https://github.com/datamadness/Feeding-TensorFlow-from-drive-MNIST-Example) for more details and practical demo. Specifically, you want to look into the input function in this file:

`MNIST_CNN_with_TFR_iterator_example.py`

#### Standard input_fn
Below is an example of standard *train_input_fn* from the MNIST classifier that shows how the input_fn is used for training an estimator:
```python
mnist_classifier.train(input_fn=train_input_fn, steps=1000)
```
You can see that the *input_fn* does not accept any parameters when called from the estimators *train* method.

#### Custom intput_fn with parameters
The problem can be easily solve via the ever-so-helpful *lambda* statement with our custom *dataset_input_fn* and an arbitrary number of input parameters:

```python
mnist_classifier.train(
    input_fn=lambda : dataset_input_fn(train_folder, train = True, 
		batch_size = batch_size, num_epochs=num_epochs),
    steps=training_steps,
    hooks=[logging_hook])
```

Enjoy!
