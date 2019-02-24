---
layout: post
current: post
cover:  assets/images/Skew_autotransform/header_spyder.png
navigation: True
title: Python function to automatically transform skewed data in Pandas DataFrame
date: 2018-05-27 08:00:00
tags: [Python]
class: post-template
subclass: 'post tag-Python'
author: DATAmadness
mathjax: true
---

When I stumble on a cool new dataset, I often find myself excitedly prototyping a quick machine learning models to see what I can get out of the latest find. This is mostly easy thanks to plenty awesome packages such as scikit-learn or Keras. As a result, I spend most of my prototyping effort on cleaning and transforming the data.

To speed up the initial transformation, I wrote a small python function that takes a Pandas DataFrame and automatically transforms any columns that exceed specified skewness. You can get it from [my GitHub repo](https://github.com/datamadness/Automatic-skewness-transformation-for-Pandas-DataFrame). Specifically these two python files:

`skew_autotransform.py`<br>
`TEST_skew_autotransform.py`

The first file lets you import the *skew_autotransform()* function and use it in your project:
<pre><code class="nohighlight">from skew_autotransform import skew_autotransform
skew_autotransform(DF, include = None, exclude = None, plot = False, threshold = 1, exp = False)
</code></pre>

#### Feature Overview

* Analyzes all columns in Pandas DataFrame and transforms the data to improve skewness if the original skewness exceeds specified threshold
* Allows to specify list of columns that should be processed or excluded
* Selection between Box-Cox transformation or log / exponential transformation
* Recognizes positive / negative skewness and applies appropriate transform (log / exp)
* Plots before and after comparison

#### Input parameters summary
* **DF**: Pandas DataFrame, mandatory
* **threshold**: skewness threshold, default value = 1, optional
* **include**: list of columns to process, optional
* **exclude**: list of columns to exclude, optional
* **exp**: If true, applies log / exponential transformation, default value is False that applies Box-Cox transformation, optional

#### Example #1
Import Boston housing dataset and apply Box-Cox transformation on any column that has absolute value of skewness larger than 0.5:

<pre><code class="nohighlight">import pandas as pd
import numpy as np
from skew_autotransform import skew_autotransform

#Import test dataset - Boston huosing data
from sklearn.datasets import load_boston

exampleDF = pd.DataFrame(load_boston()['data'], columns = load_boston()['feature_names'].tolist())

transformedDF = skew_autotransform(exampleDF.copy(deep=True), plot = True, exp = False, threshold = 0.5)

print('Original average skewness value was %2.2f' %(np.mean(abs(exampleDF.skew()))))
print('Average skewness after transformation is %2.2f' %(np.mean(abs(transformedDF.skew()))))
</code></pre>

Output:

Couple samples of the Before and After histograms that are automatically generated for each column(out of 13):
![image post](/assets/images/Skew_autotransform/example1_CRIM.png)
![image post](/assets/images/Skew_autotransform/example1_DIS.png)
<pre><code class="nohighlight"> 'CRIM' had 'positive' skewness of 5.22

 Transformation yielded skewness of 0.41
 ------------------------------------------------------

 'ZN' had 'positive' skewness of 2.23

 Transformation yielded skewness of 1.10
 ------------------------------------------------------

 NO TRANSFORMATION APPLIED FOR 'INDUS' . Skewness = 0.30
 ------------------------------------------------------

 'CHAS' had 'positive' skewness of 3.41

 Transformation yielded skewness of 3.41
 ------------------------------------------------------

 'NOX' had 'positive' skewness of 0.73

 Transformation yielded skewness of 0.36
 ------------------------------------------------------

 NO TRANSFORMATION APPLIED FOR 'RM' . Skewness = 0.40
 ------------------------------------------------------

 'AGE' had 'negative' skewness of -0.60

 Transformation yielded skewness of 0.94
 ------------------------------------------------------

 'DIS' had 'positive' skewness of 1.01

 Transformation yielded skewness of 0.15
 ------------------------------------------------------

 'RAD' had 'positive' skewness of 1.00

 Transformation yielded skewness of 0.29
 ------------------------------------------------------

 'TAX' had 'positive' skewness of 0.67

 Transformation yielded skewness of 0.33
 ------------------------------------------------------

 'PTRATIO' had 'negative' skewness of -0.80

 Transformation yielded skewness of 0.52
 ------------------------------------------------------

 'B' had 'negative' skewness of -2.89

 Transformation yielded skewness of -1.13
 ------------------------------------------------------

 'LSTAT' had 'positive' skewness of 0.91

 Transformation yielded skewness of -0.32
 ------------------------------------------------------
Original average skewness value was 1.55
Average skewness after transformation is 0.74
</code></pre>

#### Example #2
Import Boston housing dataset and apply log and exponential transformation on any column that has absolute value of skewness larger than 0.7. Exclude 'B' and 'LSTAT' column from the operation:

<pre><code class="nohighlight">import pandas as pd
import numpy as np
from skew_autotransform import skew_autotransform

#Import test dataset - Boston huosing data
from sklearn.datasets import load_boston

exampleDF = pd.DataFrame(load_boston()['data'], columns = load_boston()['feature_names'].tolist())

transformedDF = skew_autotransform(exampleDF.copy(deep=True), plot = True, exp = False, threshold = 0.5)

print('Original average skewness value was %2.2f' %(np.mean(abs(exampleDF.skew()))))
print('Average skewness after transformation is %2.2f' %(np.mean(abs(transformedDF.skew()))))
</code></pre>

Output:

Couple samples of the Before and After histograms that are automatically generated for each column(out of 13):
![image post](/assets/images/Skew_autotransform/example1_DIS_exp.png)
![image post](/assets/images/Skew_autotransform/example1_PTRATIO_exp.png)
 <pre><code class="nohighlight">------------------------------------------------------

 'CRIM' had 'positive' skewness of 5.22

 Transformation yielded skewness of 0.41

 ------------------------------------------------------

 'ZN' had 'positive' skewness of 2.23

 Transformation yielded skewness of 1.10

 ------------------------------------------------------

 NO TRANSFORMATION APPLIED FOR 'INDUS' . Skewness = 0.30

 ------------------------------------------------------

 'CHAS' had 'positive' skewness of 3.41

 Transformation yielded skewness of 3.41

 ------------------------------------------------------

 'NOX' had 'positive' skewness of 0.73

 Transformation yielded skewness of 0.36

 ------------------------------------------------------

 NO TRANSFORMATION APPLIED FOR 'RM' . Skewness = 0.40

 ------------------------------------------------------

 NO TRANSFORMATION APPLIED FOR 'AGE' . Skewness = -0.60

 ------------------------------------------------------

 'DIS' had 'positive' skewness of 1.01

 Transformation yielded skewness of 0.15

 ------------------------------------------------------

 'RAD' had 'positive' skewness of 1.00

 Transformation yielded skewness of 0.29

 ------------------------------------------------------

 NO TRANSFORMATION APPLIED FOR 'TAX' . Skewness = 0.6

 ------------------------------------------------------

 'PTRATIO' had 'negative' skewness of -0.80

 Transformation yielded skewness of 0.52


Original average skewness value was 1.55
Average skewness after transformation is 0.92
</code></pre>


You can see on these examples that both cases allowed me to very quickly improve the skewness of the data from 1.5 to more reasonable 0.7 and 0.9 respective. While it is not perfect, it is generally good enough for a first prototype. 
I would also recommend to quickly check which transformation works better for your specific dataset.

Enjoy!