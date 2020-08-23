import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.write('# Diabetes Data Analytics')

"""
Web App Which Gives a Complete Analysis of the Diabetes Data.This has been collected using direct questionnaires from the patients.

"""

data=pd.read_csv('Case study 3-diabetes_data_upload.csv')
st.write(data.head())

from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(drop='first',sparse=False)
final_data=pd.DataFrame(ohe.fit_transform(data.iloc[:,1:-1]),columns=data.columns[1:-1])
final_data['class']=data['class'].map({'Positive':1,'Negative':0})
final_data=pd.concat([data['Age'],final_data],axis=1)
a=[]
chi_col=[]
for t in final_data.columns[1:]:
	x,y,i,z=chi2_contingency(pd.crosstab(final_data[t],final_data['class']).values)
	a.append((t,x,y,y<=0.05))
	if y<=0.05:
		chi_col.append(t)
ohe1=OneHotEncoder(drop='first',sparse=False)
final_data11=pd.DataFrame(ohe1.fit_transform(data.loc[:,np.array(chi_col)[:-1]]),columns=chi_col[:-1])
final_data11['class']=data['class'].map({'Positive':1,'Negative':0})
final_data11=pd.concat([data['Age'],final_data11],axis=1)

	
ded=pd.DataFrame(np.array(a),columns=['Col_Name','Chi Square','p-value','<=0.5-T,>0.5F'])

st.write('## Exploratory Data Analysis')
option=st.selectbox('What do you want to do?',('Chi Square','Cross Tabs','Plot'))
if option=='Cross Tabs':
	col1=st.selectbox('Select the column 1',(data.columns))
	col2=st.selectbox('Select the column 2',(data.columns))
	st.write(pd.crosstab(data[col1],data[col2]))
if option=='Plot':
	col1=st.selectbox('Select the column 1',(data.columns))
	fig=px.histogram(data,x=col1,color='class')
	st.plotly_chart(fig)
if option=='Chi Square':
	col1=st.selectbox('Select the column 1',(*final_data.columns[1:],'all'))
	if col1 != 'all':
		x,y,i,z=chi2_contingency(pd.crosstab(final_data[col1],final_data['class']).values)
		st.write(f'Chi-Square Statistic={x} and p value={y}') 
		if y<=0.05:
    			st.write(f'Since P value is less than 0.05.We can say that the {col1} is related in predicting whether a person has diabetes or Not')
		if y>0.05:
    			st.write(f'Since P value is Greater than 0.05.We can say that the {col1} is  not related in predicting whether a person has diabetes or Not')
	if col1=='all':
		st.write(ded)
st.write('## Prediction')
option1=st.checkbox('Use only Related Columns (from chi2)')

if option1:
	x=final_data11.iloc[:,:-1]
	y=final_data11.iloc[:,-1]
	test_si=st.slider('test_size',0.0,0.30,0.25)
	x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=test_si,random_state=180)
	st.write(f'The Train Size is {x_train.shape}')
	st.write(f'The Test Size is {x_test.shape}')
	rf=RandomForestClassifier()
	rf.fit(x_train,y_train)
	st.write('The Data is fitted to the RandomForestClassifier Model')
	st.write("## Let's Predict")
	val1=st.text_input(x.columns[0],50)
	val2=st.selectbox(x.columns[1],('Male','Female'))
	val3=st.selectbox(x.columns[2],('Yes','No'))
	val4=st.selectbox(x.columns[3],('Yes','No'))
	val5=st.selectbox(x.columns[4],('Yes','No'))
	val6=st.selectbox(x.columns[5],('Yes','No'))
	val7=st.selectbox(x.columns[6],('Yes','No'))
	val8=st.selectbox(x.columns[7],('Yes','No'))
	val9=st.selectbox(x.columns[8],('Yes','No'))
	val10=st.selectbox(x.columns[9],('Yes','No'))
	val11=st.selectbox(x.columns[10],('Yes','No'))
	val12=st.selectbox(x.columns[11],('Yes','No'))
	val13=st.selectbox(x.columns[12],('Yes','No'))
	val=ohe1.transform([[val2,val3,val4,val5,val6,val7,val8,val9,val10,val11,val12,val13]])
	button=st.button('Predict')
	if button:
		pred=np.c_[int(val1),val]
		if np.where(rf.predict_proba(pred)[:,0]>=0.3,0,1)==1:
			st.header('The Person has Diabetes')
		elif np.where(rf.predict_proba(pred)[:,0]>=0.3,0,1)==0:
			st.header('The Person does not have Diabetes')
else:
	x=final_data.iloc[:,:-1]
	y=final_data.iloc[:,-1]
	test_si=st.slider('test_size',0.0,0.30,0.25)
	x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=test_si,random_state=180)
	st.write(f'The Train Size is {x_train.shape}')
	st.write(f'The Test Size is {x_test.shape}')
	rf=RandomForestClassifier()
	rf.fit(x_train,y_train)
	st.write('The Data is fitted to the RandomForestClassifier Model')
	st.write("## Let's Predict")
	val1=st.text_input(x.columns[0],50)
	val2=st.selectbox(x.columns[1],('Male','Female'))
	val3=st.selectbox(x.columns[2],('Yes','No'))
	val4=st.selectbox(x.columns[3],('Yes','No'))
	val5=st.selectbox(x.columns[4],('Yes','No'))
	val6=st.selectbox(x.columns[5],('Yes','No'))
	val7=st.selectbox(x.columns[6],('Yes','No'))
	val8=st.selectbox(x.columns[7],('Yes','No'))
	val9=st.selectbox(x.columns[8],('Yes','No'))
	val10=st.selectbox(x.columns[9],('Yes','No'))
	val11=st.selectbox(x.columns[10],('Yes','No'))
	val12=st.selectbox(x.columns[11],('Yes','No'))
	val13=st.selectbox(x.columns[12],('Yes','No'))
	val14=st.selectbox(x.columns[13],('Yes','No'))
	val15=st.selectbox(x.columns[14],('Yes','No'))
	val16=st.selectbox(x.columns[15],('Yes','No'))
	val=ohe.transform([[val2,val3,val4,val5,val6,val7,val8,val9,val10,val11,val12,val13,val14,val15,val16]])
	button=st.button('Predict')
	if button:
		pred=np.c_[int(val1),val]
		if np.where(rf.predict_proba(pred)[:,0]>=0.3,0,1)==1:
			st.header('The Person has Diabetes')
		elif np.where(rf.predict_proba(pred)[:,0]>=0.3,0,1)==0:
			st.header('The Person does not have Diabetes')





	


	

	