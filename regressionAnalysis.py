#Problem-Set-9 
#Due 11.4.18

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


#PART A: AnalysisData

class AnalysisData:
    #^ This is the initialization function
    #__ private reserved function that goes into Analysis
        
#Initialize (init) any attributes and handle preprocessing 
    def __init__(self):
        self.dataset= [] #expected this to be a list 
        self.variables= [] 
    
# Define a function and put it into self.variables 
    def parseFile(self,filename):
        #open the file
        self.dataset = pd.read_csv(filename)
        for column in self.dataset.columns.values:
            if column != "competitorname":
                self.variables.append(column)

#PART B: LinearAnalysis

#align initialized functions 
class LinearAnalysis:
    def __init__(self,targetY_input):
        self.bestX = "" 
        self.targetY = targetY_input 
        #self.fit = ""
        
    def runSimpleAnalysis(self, data): 
        #regr = LinearRegression()
        best_r2 = -1
        best_var = ""
        #regr.fit(<candy>, <sugar>)
        #regr.predict(<candy>)
        #r2_score(<sugar>, <predicted values>)
        for column in data.variables:
            if column != self.targetY:
                #take column and pull all values from it and set it to independent_var
                independent_var = data.dataset[column].values
                #reshapes it into a 2D array from a 1D array 
                independent_var = independent_var.reshape(len(independent_var),1)
                
                regr = LinearRegression()
                regr.fit(independent_var, data.dataset[self.targetY])
                pred = regr.predict(independent_var)
                r_score = r2_score(data.dataset[self.targetY],pred)
                if r_score > best_r2:
                    best_r2 = r_score
                    best_var = column
        self.bestX = best_var
        print(best_var, best_r2)

#PART C: Logistic Analysis
class LogisticAnalysis:
    def __init__(self,target_Y):
        self.target_Y= _targetY
        self.bestX = ""

# PROBLEM 1. Using the candy-data.csv file in the repo, populate an AnalysisData object that will hold the data you'll use for today's problem set. You should read in the data from the CSV, store the data in the dataset variable, and initialize the xs (column name) and targetY variables appropriately. targetY should reference the variable describing whether or not a candy is chocolate.

data = AnalysisData()
data.parseFile("candy-data.csv")
#data.read_csv('candy-data.csv')

#PROBLEM 2. Create a function to initialize a LinearAnalysis object that takes a targetY as its input parameter. Create the same function for LogisticAnalysis. Note that you will use the LinearAnalysis object to try to predict the amount of sugar in the candy and the LogisticAnalysis object to predict whether or not the candy is chocolate.

#See parts B and C

################################################################################
# WEDNESDAY & FRIDAY WORK 
################################################################################
#PROBLEM 3

# Runs the __init__ function which runs the targetY function 
line_analysis = LinearAnalysis('sugarpercent')
line_analysis.runSimpleAnalysis(data)