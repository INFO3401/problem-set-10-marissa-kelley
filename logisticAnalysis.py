#Logistic Analysis
#Due 11.11.18
#I worked with Jacob Paul, Hannah Weber, Taylor Lawrence, Jack Sandberg 

#Import all the things
import pandas as pd
import csv 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

################################################################################
# LAST WEEK'S WORK (PROBLEM SET 6)
################################################################################
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
        
################################################################################
# MONDAY & WEDNESDAY WORK 
################################################################################
#PROBLEM 1
#import regression analysis 
#import regressionAnalysis

class LogisticAnalysis:
    def __init__(self,targetY_input):
        self.target_Y= targetY_input
        self.bestX = ""
        self.fit = ""
    
    def runSimpleAnalysis(self, data): 
        best_r2 = -1
        best_var = ""
        
        for column in data.variables:
            if column != self.target_Y:
                #take column and pull all values from it and set it to independent_var
                independent_var = data.dataset[column].values
                #reshapes it into a 2D array from a 1D array 
                independent_var = independent_var.reshape(len(independent_var),1)
                
        regr = LogisticRegression()
        regr.fit(independent_var, data.dataset[self.target_Y])
        pred = regr.predict(independent_var)
        r_score = r2_score(data.dataset[self.target_Y],pred)
        if r_score > best_r2:
            best_r2 = r_score
            best_var = column
        
        self.bestX = best_var
        print(best_var, best_r2)
        
#Last Week
dataAD = AnalysisData()
dataAD.parseFile("candy-data.csv")

#Last Week
line_analysis = LinearAnalysis('sugarpercent')
line_analysis.runSimpleAnalysis(dataAD)

#This week
logisticAnalysis = LogisticAnalysis("chocolate")
logisticAnalysis.runSimpleAnalysis(dataAD)
 
#Do the two functions find the same optimal variable? 
    #They both find the same optimal variable. 
#Which method best fits this data? 
    #The method that best fits the data is the linear analysis which printed out the pricepercent 0.10870630201695808 while the logistic analysis printed out the winpercent 0.04279279279279302. 


#PROBLEM 2. Add a function to the LogisticAnalysis object called runMultipleRegression. This function should take in an AnalysisData object as a parameter and should use this object to compute a multiple logistic regression using all of the possible independent variables in your dataset to predict whether or not a candy is chocolate (note, you should not use your dependent variable as an independent variable). Print the variable name and resulting fit. In your testing code, create a new LogisticAnalysis object and use it to run this function on your candy data. Compare the outcomes of this and the simple logistic analysis. Which model best fits the data? Why? 

def runMultipleRegression(self, data): 
        best_r2 = -1
        best_var = ""
        
        for column in data.variables: #creating a new dataframe each time 
            if column != self.target_Y:
                #take column and pull all values from it and set it to independent_var
                independent_var = data.dataset[column].values
                #reshapes it into a 2D array from a 1D array 
                independent_var = independent_var.reshape(len(independent_var),1)
                
            regr = LogisticRegression()
            regr.fit(independent_var, data.dataset[self.target_Y])
            pred = regr.predict(independent_var)
            r_score = r2_score(data.dataset[self.target_Y],pred)
            if r_score > best_r2:
                best_r2 = r_score
                best_var = column
        
            self.bestX = best_var
            print(best_var, best_r2)
            if r_score > best_r2:
            best_r2 = r_score
            best_var = column
        
        self.bestX = best_var
        print(best_var, best_r2)

#Which model best fits the data? Why? 
    #The model that best fits the data is... 
    
    
#PROBLEM 3. Write the equations for your linear, logistic, and multiple logistic regressions. Hint: Use the equations from the slides from Monday's lecture to work out what a logistic regression equation might look like. 
#Y = sugar percent

################################################################################
# FRIDAY WORK 
################################################################################
#4. Identify the independent variable(s) and its type (e.g., categorical, continuous, or discrete), the dependent variable and its type, and the null hypothesis for each of the following scenarios: 

#(a) What candies contain more sugar, those with caramel or those with chocolate?
    #Independent Variable: Candy 
    #Type: Categorical 
    #Dependent Variable: Amount of sugar 
    #Type: Continuous 
    #Null Hypothesis: Both types of candies contain the same amount of sugar 

#(b) Are there more split ticket voters in blue states or red states? 
    #Independent Variable: The state (blue vs.red)
    #Type: Categorical- not an order to it 
    #Dependent Variable: Split ticket voters 
    #Type: Continuous
    #Null Hypothesis: Both states contain the same amount of split ticket voters

#(c) Do phones with longer battery life sell at a higher or lower rate than other phones?
    #Independent Variable: Phone 
    #Type: Categorical 
    #Dependent Variable: Selling rate 
    #Type: Continuous 
    #Null Hypothesis: Phones with the longer battery life sell at the same rate as other phones. 