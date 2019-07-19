# SurvivalAnalysis
NOTE: github's rendering in notebook doesn't always work, if the ipynb file doesn't load, you can see it in https://nbviewer.jupyter.org/github/YIZHE12/SurvivalAnalysis/blob/master/EDA_survival_analysis.ipynb

## Background:
Your team has been tasked with diagnosing why and when employees from your subsidiaries leave. 

You need a tangible data-driven recommendation for each of the ten Presidents of your subsidiaries. 

What are your recommendations and why?

## Quick look:
This is a survival analysis tasks that I solved using [Kaplan Meier plot](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3059453/) and [Cox Proportional-Hazards Model](http://www.sthda.com/english/wiki/cox-proportional-hazards-model). There are some data cleaning to do as the datedata is several formats. There are also outliers in the data. For examples, two data points have seniority of 90 years which is not likely as we don't expect someone who have worked for 90 years. For more information, you can have a look at the pdf file in the repo.

The notebook here included all the analysis. The data is the txt file uploaded.

## Prerequisites
pandas, numpy, lifelines, matplotlib, seaborn
```
pip install numpy
pip install pandas
pip install matplotlib
pip install seaborn
pip install lifelines
```
