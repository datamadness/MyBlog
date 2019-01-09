---
layout: post
current: post
cover:  assets/images/tags.jpg
navigation: True
title: DRAFT Support Data Generator in Python
date: 2018-05-27 08:00:00
tags: [Statistics]
class: post-template
subclass: 'post tag-Statistics'
author: DATAmadness
mathjax: true
---

We all have been there - it is Sunday evening, you have couple of fresh ideas for a new customer centric strategy and you want to test how it would hold up in the real world. Unfortunately, it might be hard to get real or at least somewhat realistic customer support ticket datasets for specific business models and company size. 

If this issue plagues you on regular basis, head to [this repository on my GitHub](https://github.com/datamadness/Support-ticket-data-generator.git) and download the two python files:

`main_generate_ticket_data.py`<br>
`generate_daily_data.py`

This code will allow you to input few numeric parameters and quickly generate custom support ticket datasets that reflect what you could expect in real world business operations.
#### Feature Overview

* A python function to generate one year worth of support data(call repeatedly for x years worth of data)
* Generates varied, but statistically relevant number of support tickets for each day of the year
* Effects of business days vs weekends
* Capable to simulate an impact of arbitrary number of busy seasons through the year(e.g. Christmas in retail or tax periods in accounting)
* Simulates realistic, but easy to control statistical distribution of logged time for each support ticket / case.
* Control number of customer accounts to capture desired support volume vs customer base size

#### Input parameters summary

Daily ticket volume through the year controls:

* Seasonal peaks
* Weekend volumes
* Daily volume range (population mean, approximate min and max outliers)

Effort / logged time per ticket controls:

* Median
* Minimum logged time
Advanced / Optional:
* Skewness factor
* Degrees of freedom
* Non-centrality (impacts variance and kurtosis)

Customer base controls:

- Number of unique accounts

#### Modelling daily tickets volumes through the year
The number of daily tickets is typically random variable that has reasonably defined mean and variance. Moreover, the mean is likely to move through the year to reflect seasonal effects such as major holidays or annual business cycles. 
To model this behavior, the code is using truncated normal continuous variable function from scipy package to generate the number of tickets on given day:
<pre><code class="nohighlight">        #Calculate distribution function
        dist = stats.truncnorm((annual_minimum - mu) / sigma, (annual_maximum - mu) / sigma, loc=mu, scale=sigma)
        #Generate number of tickets for given day
        ticket_num = int(dist.rvs())
</code></pre>
Simply specify the approximate minimum and maximum number of cases per day through the year. The mean `mu` and variance `sigma` for given day then will be calculated based the seasonal effect function:
<pre><code class="nohighlight">        #Calculate mean for given day
        mu = (annual_maximum - annual_minimum)/2 + seasonal_factor * (busy_level - 0.5) * ((annual_maximum - annual_minimum)/2) 
        #Calculate standard deviation for given day
        sigma = (annual_maximum - annual_minimum) * 0.1
</code></pre>
The seasonal effect requires you to input the peak months as integer values. Here is an example for simulating two seasonal peaks in February and November:
busy_months = np.array([2,11])

Moreover, input `seasonal_factor` (float <0,1>) allows you to control the magnitude of seasonal effects:

* 0 no seasonal effects
* 1 maximum seasonal effects

The following plot demonstrates the annual volumes and seasonality dependency with all other parameters being constant:
<pre><code class="nohighlight">    annual_minimum = 5		# ~ annual daily minimum of tickets
    annual_maximum = 50       	# ~ annual daily maximum of tickets           
    busy_months = np.array([2,7])       # February and July are busy months
</code></pre>
*Seasonal factor impact comparison:*
![image test](/assets/images/support_data_generator/weekly_ticket_totals_comparison.png)







###### Impacts of weekends:
Weekends can have a major impact on your ticket volumes. This can be particularly pronounced if your company is B2B oriented and your clients are off work while having little to no impact if you are operating in B2C model. To model this, select the appropriate value of the weekend_factor parameter (float - <0,1>):
* 0 - no ticket volume on weekends
* 1 - weekends are the same as business days




###### Modelling logged time for each opened ticket
Crucial part of planning your operations is to have an understanding of how much time is spent on each ticket. Unfortunately, this data cannot be simulated by a simple normal distribution function. While you will be able to find reasonable mean in your real world data, you will see few large outliers due to complex support problems and more or less pronounced positive skew. Fortunately, this can be quite well approximated by the non-central chi-squared distribution([Wikipedia](https://en.wikipedia.org/wiki/Noncentral_chi-squared_distribution )) that can generalize the k - degrees of freedom of chi-squared function via noncentrality parameter.

Probability Density Function(PDF) for vanila non-central chi-squared function:

$$f_{X}(x;k,\lambda )=\sum _{i=0}^{\infty }{\frac {e^{-\lambda /2}(\lambda /2)^{i}}{i!}}f_{Y_{k+2i}}(x)$$

Non-central chi-squared PDF visualization (source [en.wikipedia.com](https://en.wikipedia.org/wiki/Noncentral_chi-squared_distribution#/media/File:Chi-Squared-(nonCentral)-pdf.png)):
<img src="/assets/images/support_data_generator/Chi-Squared-(nonCentral)-pdf.png" width="600" height="450">


In the proposed python code, the logged time data is generated by function daily_data imported from generate_daily_data.py. This function is called by the main function with the following parameters:

###### Mandatory:
* size - automatically determined by the main function
* logged_time_median_desired - median logged time for the population in minutes
* Logged_time_minimum - minimum logged time in minutes

###### Optional:
* skewness_factor - [0,1] value;  0 for default chi^2 distribution; 1 for large positive skew
* k - degrees of freedom
* nonc - non-centrality (stats typically uses lambda letter) 



The following plot shows histogram and kernel density estimation(KDE) for the logged time data of ~8000 tickets over 12 months:
![image test](/assets/images/support_data_generator/logged_time_distribution_plot.png)
