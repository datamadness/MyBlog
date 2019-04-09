---
layout: post
current: post
cover:  assets/images/startup-desktop.jpg
navigation: True
title: Forecasting Operations Team Resources Using Statistics
date: 2018-06-24 10:32:00
tags: [Statistics]
class: post-template
subclass: 'post tag-Statistics'
author: DATAmadness
---
Planning staff allocation in an environment geared heavily towards operations can be challenging due to the erratic nature of the demand from the internal or external clients. The predictability issues affects wide range of teams - from sales or support teams to development teams or analytics teams. If your team is exposed to any type of ad-hoc requests on regular basis, you are likely familiar with having your carefully estimated project plans derailed by unexpected workload.

This article will propose a team resources planning framework that allows you to estimate:

- The fraction of team resources required to accommodate the seemingly random ad-hoc tasks without being overwhelmed
- How much of your team's time can be safely committed to project work 

To tackle this task, we will use basics statistics to guide your staffing and time allocation decisions by using basic data that are typically available in your company or can be even estimated and modeled. 

The data used in this example are generated using a statistical model described in [this article](https://datamadness.github.io/Support-Data-Generator). You can use the provided tool to model your own data for quick experiments.

## Exploring the available data
<iframe width="800" height="600" src="https://app.powerbi.com/view?r=eyJrIjoiY2ZlODU5YjAtZTQxMC00MzUxLTgyNjgtOWY5MjcyZGU1ZGY1IiwidCI6ImZmZTRhZjdkLWIxODgtNDIzZi1iMmQ5LTUwZmIzYjQ2NjU2ZiJ9" frameborder="0" allowFullScreen="true"></iframe>



### April Recorded
<iframe width="800" height="600" src="https://app.powerbi.com/view?r=eyJrIjoiZTE2NzVjZjgtOTRkOC00OGY5LTgyZmUtOTVlNDRlMDIwYTdjIiwidCI6ImZmZTRhZjdkLWIxODgtNDIzZi1iMmQ5LTUwZmIzYjQ2NjU2ZiJ9" frameborder="0" allowFullScreen="true"></iframe>


### May Predicted vs Actual
<iframe width="800" height="600" src="https://app.powerbi.com/view?r=eyJrIjoiZDUwM2EyOTUtYWM2NS00YTNlLWFkNGItMmIxMzljYWJiOTJlIiwidCI6ImZmZTRhZjdkLWIxODgtNDIzZi1iMmQ5LTUwZmIzYjQ2NjU2ZiJ9" frameborder="0" allowFullScreen="true"></iframe>