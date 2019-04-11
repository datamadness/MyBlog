---
layout: post
current: post
cover:  assets/images/startup-desktop.jpg
navigation: True
title: Use Statistics to Forecast Resource Allocation in Operations Teams
date: 2019-4-1 10:32:00
tags: [Statistics]
class: post-template
subclass: 'post tag-Statistics'
author: DATAmadness
mathjax: true
---
Planning staff allocation in an environment geared heavily towards operations can be challenging due to the erratic nature of the demand from both internal or external clients. The predictability issues affect a wide range of teams - from sales or support to development and data analytics departments. If your team is exposed to any type of ad-hoc requests on a regular basis, you are likely familiar with having your carefully estimated project plans derailed by an unexpected workload. 

This article will demonstrate that it is possible to accurately predict commitments even in a very erratic environment as long as we choose the right time frame and management approach. A simple, yet powerful methodology is proposed for your operations team planning. As opposed to the common go-to Machine Learning solution, we'll use simple statistics that will let us to get away with a small dataset that is readily available at most businesses.

> For managers, the message is that it is not always necessary to micromanage the team and require keeping up with daily or weekly  deadlines for secondary project work in operations, but it is certainly possible to plan and hold the team responsible for delivering results on a monthly basis.

The proposed resource planning framework will allow you to estimate:

- The fraction of team resources required to accommodate seemingly random ad-hoc tasks without being overwhelmed
- How much of your team's time can be safely committed to project work


The data used in this example are generated using a statistical model described in [this article](https://datamadness.github.io/Support-Data-Generator). You can use the provided tool to model your own data for quick experiments.

#### Business Scenario
You are a manager of a data analytics team with 20 team members. The team provides business intelligence, reporting and data consultation services to several departments in your large corporation X. Most requests come to the team on an ad-hoc basis and are difficult to predict. The rest of the team's time is dedicated to project work such as developing more advanced data science and ML solutions to increase efficiency.

**The Problem:** The team often misses project deadlines because they are busy with a surge of analytics and reporting requests that must be done with high priority.

#### Exploring the available data
The team logs time they spend on providing the data analytics requests for the company. We can explore the data from the past six years in this interactive visualization(you can display it fullscreen):
<iframe width="800" height="600" src="https://app.powerbi.com/view?r=eyJrIjoiY2ZlODU5YjAtZTQxMC00MzUxLTgyNjgtOWY5MjcyZGU1ZGY1IiwidCI6ImZmZTRhZjdkLWIxODgtNDIzZi1iMmQ5LTUwZmIzYjQ2NjU2ZiJ9" frameborder="0" allowFullScreen="true"></iframe>

<br>
**Key takeaways from 2017 data:**

- The effort is highly cyclical with more requests coming in around quarter ends when there is a higher need for reporting
- Most requests come from Investor Relations and the Sales and Marketing departments
- The maximum monthly effort in 2017 was logged in December with 1663 hours
- The minimum monthly effort in 2017 was logged in January with 719 hours
- Annual monthly median is around 1180 Hours


#### Challenge: Accurately Predict Team Resources Allocation for Next Month
Lets imagine it is the end of April 2018 and we want to accurately predict May 2018 numbers:

1. How much time the team will spend on ad-hoc requests
1. How much time we can safely commit to project work while using the team's time efficiently

To do this, we will first look at the previous months data:
<iframe width="800" height="600" src="https://app.powerbi.com/view?r=eyJrIjoiZTE2NzVjZjgtOTRkOC00OGY5LTgyZmUtOTVlNDRlMDIwYTdjIiwidCI6ImZmZTRhZjdkLWIxODgtNDIzZi1iMmQ5LTUwZmIzYjQ2NjU2ZiJ9" frameborder="0" allowFullScreen="true"></iframe>

We see that on daily basis the team spent:
Average of 

$$ \mu_{april} = 45.5 Hours$$ 

on reports with Standard Deviation 

$$ \sigma_{april} = 17.5 Hours$$

This allows us to calculate the Standard Deviation of the Mean (SEM) as:

$$ SEM = \frac \sigma {\sqrt n} = \frac {17.5} {\sqrt 30} = 3.2 Hours $$

This is where we get the first big assistance from statistics: While the daily logged time in April went from a minimum of 17 hours a day to a maximum of 84 hours a day, with 30 days in a month the standard deviation of the mean(SEM) is only 3.2 Hours. That is very little!

The second huge accuracy boost comes from the [Central Limit Theorem](https://en.wikipedia.org/wiki/Central_limit_theorem) that tells us that even our skewed data will have a normally distributed mean when the sample size is large enough. The rule of thumb is to have 30 or more samples. Luckily for us, one month has 30 days! (You might want to increase the prediction time frame if you want to give your employees weekends off :))

As a result, we can use a simple normal distribution statistics to calculate the 95% Confidence Interval(CI) for April and get:

$$ 1358 \le \bar x \le 1370$$

We actually know for a fact that the true mean is in this range, but the goal is to use this interval for extrapolation for the May 2018 Confidence Interval. As I mentioned, we'll keep things very simple and look at 2017 to see that May was ~8% higher than April. Hence, we simply add 8% to the April CI to predict the range for May 2018 mean as:

$$ 1466 \le \bar x \le 1480$$

This means that our team of 20 with a total of 4000 man hours available will spend approximately:
- **1470 hours** on ad-hoc requests for the corporation
- **2530 hours** can be safely dedicated to other work such as projects

Notice how we went from a standard daily deviation of 17.5 hours to predicting the monthly effort within an interval of 14 hours!!!

**Bonus: Poisson Distribution**

We can also use Poisson distribution to predict the probability that we'll receive certain number of request on any given day. For example, we can calculate the probability of receiving 6 or more requests on any given day by plugging in the average rate of 3.4 requests/day into the cumulative poisson distribution function. This can be done in Python, Excel or any other tool of your choice. The resulting probability $p = 0.06$ tells us that we are most likely to have one or two days with 6 or more requests.


#### May 2018 Prediction vs Reality
So, how did we do with our predictions when compared to reality? Here is the May 2018 overview:
<iframe width="800" height="600" src="https://app.powerbi.com/view?r=eyJrIjoiZDUwM2EyOTUtYWM2NS00YTNlLWFkNGItMmIxMzljYWJiOTJlIiwidCI6ImZmZTRhZjdkLWIxODgtNDIzZi1iMmQ5LTUwZmIzYjQ2NjU2ZiJ9" frameborder="0" allowFullScreen="true"></iframe>

It may be hard to believe, but we were able to predict the entire month's effort required for ad-hoc analytics and reporting requests with more than 99% accuracy, being off only by 11 hours(predicted ~1470 vs actual 1459).

#### Conclusion

The experiment shows that if we select the correct time frame for our planning, we can use the strength of simple statistics to predict future effort with high accuracy even for teams engaged in seemingly unpredictable business. While we would be hard pressed to predict the effort required for the next day with 20 h accuracy, we could predict an entire month's effort with only 11 hour error.

For managers, the message is that it is not necessary to micromanage the team and require keeping up with daily or weekly deadlines for secondary project work, but it is certainly possible to plan and hold the team responsible for delivering results on a monthly basis.

Data, calculations, and visualization files used in this example can be downloaded from [this GitHub repo](https://github.com/datamadness/Statistics-for-planning-team-reasources-in-operations).
