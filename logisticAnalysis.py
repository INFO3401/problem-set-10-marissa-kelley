#Logistic Analysis
#Due 11.11.18

#Import things
#Import all the things
import pandas as pd
import csv 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

################################################################################
# MONDAY WORK 
################################################################################
#I worked with Jacob Paul, Hannah Weber, Taylor Lawrence, Jack Sandberg 

import regressionAnalysis.py

#1. Add a function to the LogisticAnalysis object called runSimpleAnalysis. This function should take in an AnalysisData object as a parameter and should use this object to compute which variable best predicts whether or not a candy is chocolate using logistic regression. Print the variable name and the resulting fit. Do the two functions find the same optimal variable? Which method best fits this data? Make sure your best predictor is NOT the same as the targetY variable.

#2. Add a function to the LogisticAnalysis object called runMultipleRegression. This function should take in an AnalysisData object as a parameter and should use this object to compute a multiple logistic regression using all of the possible independent variables in your dataset to predict whether or not a candy is chocolate (note, you should not use your dependent variable as an independent variable). Print the variable name and resulting fit. In your testing code, create a new LogisticAnalysis object and use it to run this function on your candy data. Compare the outcomes of this and the simple logistic analysis. Which model best fits the data? Why? 

#3. Write the equations for your linear, logistic, and multiple logistic regressions. Hint: Use the equations from the slides from Monday's lecture to work out what a logistic regression equation might look like. 

################################################################################
# WEDNESDAY WORK 
################################################################################
#I worked with Jacob Paul, Hannah Weber, Taylor Lawrence, Jack Sandberg 