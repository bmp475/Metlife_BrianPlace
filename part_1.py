import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

part1a=pd.read_csv('appointments.csv', na_values = '\N')
part1b=pd.read_csv('patients.csv', na_values = '\N')

part1a.head()
part1b.head()


#rename 'patient id' as 'id' to match first record
part1b.rename(columns={'patient_id':'id'}, inplace=True)
part1b.head()

#1 merge on id
part1=pd.merge(left=part1a, right=part1b, left_on='id', right_on='id')

part1.head()

#2a
part1['appointment_date'].dtype


#convert to datetime
part1['appointment_date']=pd.to_datetime(part1['appointment_date'], infer_datetime_format=True)

#add day of week
part1['day_of_week']=part1['appointment_date'].dt.weekday_name
part1.head()

#2b
part1.groupby('day_of_week')['id'].nunique()

#3
part1['age'].max()
part1['age'].min()

#bin into 10 groups: max age 113, increments of 11.3. Add .1 to include max and min values
bins=[-.1,11.3,22.6,33.9,45.2,56.5,67.8,79.1,90.4,101.7,113.1]
names=['0-11.3', '11.3-22.6','22.6-33.9','33.9-45.2','45.2-56.5','56.5-67.8','67.8-79.1','79.1-90.4','90.4-101.7','101.7-113']
binnames=pd.cut(part1['age'], bins, labels=names)
part1['binnames']=pd.cut(part1['age'], bins,labels=names)

#count number in each bin
part1['binnames'].value_counts()

#plot histogram of bins. Last column not visible on scale since only 14 people
part1.plot(x='binnames', y='age', kind='hist', rwidth=.9)


#4
#EDA, check for missing values, outliers, distribution
part1.describe()
part1.hist()
part1.corr()

#Fix age values less than 0
part1.loc[part1['age']<0, 'age']=0
part1['age'].min()

#Convert response  variable 'status' to 0,1
part1['status'].replace(('Show-Up', 'No-Show'), (1,0), inplace=True)

part1.dtypes


#MODEL
#split data into 80% train, 20% test. categorical variables need conversion for model integration, much more refininement of this model
x=part1.ix[:,(5,7,8,9,10,11)].values
y=part1.ix[:,3].values
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=.2)
print x_train.shape, x_test.shape


l=LogisticRegression()
l.fit(x_train, y_train)


#evaluate model. Ok at true negatives, terrible true positives
pred=l.predict(x_test)

cm= confusion_matrix(y_test,pred)
print cm
print(classification_report(y_test, pred))
