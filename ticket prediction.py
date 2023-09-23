"""
Name: Eric Zheng
Date:11/14/2021

"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def cleanReg(reg):

    regList = {'PAS' : 'PAS', 'COM' : 'COM'}
    return(regList.get(reg.upper(),'OTHER'))           #.get returns second value if nothing is foundin first 

def cleanColor(c):
    abbrev = [('GRAY', ['GY','GREY','GRAY','SILVE', 'SIL', 'SL']),\
              ('WHITE',['WH','WHITE']),\
              ('BLACK',['BK', 'BLACK', 'BL']),\
              ('BLACK',['BLUE']),\
              ('RED',['RED, RD']),\
              ('GREEN',['GR, GREEN']),\
              ('BROWN',['BROWN, TAN'])
             ]
    colors = {}
    for (a1,a2) in abbrev:
        for a in a2:
            colors[a] = a1

    return colors.get(c.upper(), 'OTHER')

def addIndicators(df,cols=['Registration', 'Color', 'State']):
	newdf = pd.get_dummies(df, columns=cols ,drop_first=True )
	return newdf


def build_clf(df, xes, y_col = "NumTickets", test_size = 0.25, random_state=17):

	clf =LinearRegression()
	#regr.fit(df[xes], df[y_col])
	X_train, X_test, y_train, y_test = train_test_split(df[xes], df[y_col], test_size=test_size, random_state=random_state)
	clf.fit(X_train,y_train)
	y_predict = clf.predict(X_test)
	score = clf.score(X_test, y_test)
	return score ,clf
	
	
	

	
df = pd.read_csv('Parking_Q1_2021_Lexington.csv')
df = df[['Plate ID','Plate Type','Registration State','Issue Date','Vehicle Color']]
df = df.dropna()
print(f'Your file contains {len(df)} parking violations.')
df['Plate Type'] = df['Plate Type'].apply(cleanReg)
df['Vehicle Color'] = df['Vehicle Color'].apply(cleanColor)
newDF =  df.groupby('Plate ID').agg(NumTickets =
    pd.NamedAgg(column = 'Plate ID', aggfunc = 'count'),
    Registration = pd.NamedAgg(column = 'Plate Type', aggfunc = 'first'),
    State = pd.NamedAgg(column = 'Registration State', aggfunc = 'first'),
    Color = pd.NamedAgg(column = 'Vehicle Color', aggfunc = 'first')
)
print(newDF)	
	
	
xes = ['State_NY','Registration_OTHER', 'Registration_PAS', 'Color_GRAY', 'Color_OTHER', 'Color_WHITE']
y_col = 'NumTickets'
sc,clf = build_clf(newDF, xes)
print(f'Score is {sc}.')
predicted = clf.predict([[1,0,0,0,0,1]])[0]
print(f'NY state, white commercial vehicle (encoded as: [1,0,0,0,0,1])\n\twill get {predicted:.2f} tickets.')
predicted = clf.predict([[1,0,1,1,0,0]])[0]
print(f'NY state, gray passenger vehicle (encoded as: [1,0,1,1,0,0])\n\twill get {predicted:.2f} tickets.')	
	
	
	
	
	
	
	
