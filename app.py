import streamlit as st 
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
st.set_option('deprecation.showfileUploaderEncoding', False)
# Load the pickled model
pickle_in = open("Model.pkl","rb")
model=pickle.load(pickle_in)
dataset= pd.read_csv('Data.csv')
X = dataset.iloc[ : ,1:10].values


# Taking care of missing data
#handling missing data (Replacing missing data with the mean value)  
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.NAN, strategy= 'constant', fill_value='Female', verbose=1, copy=True)
#Fitting imputer object to the independent variables x.   
imputer = imputer.fit(X[:, 2:3]) 
#Replacing missing data with the calculated mean value  
X[:, 2:3]= imputer.transform(X[:, 2:3])  


# Taking care of missing data
#handling missing data (Replacing missing data with the mean value)  
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.NAN, strategy= 'mean', fill_value=None, verbose=1, copy=True)
#Fitting imputer object to the independent variables x.   
imputer = imputer.fit(X[: , 3:9]) 
#Replacing missing data with the calculated mean value  
X[:, 3:9]= imputer.transform(X[:, 3:9])    

# Encoding Categorical data:
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 2] = labelencoder_X.fit_transform(X[:, 2])

# Encoding Categorical data:
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)


def predict_note_authentication(CreditScore,Geography,Gender,Age,Tenure,Balance,HasCrCard,IsActiveMember,EstimatedSalary):
  output= model.predict(sc.transform([[CreditScore,Geography,Gender,Age,Tenure,Balance,HasCrCard,IsActiveMember,EstimatedSalary]]))
  print("Heart Disease Category is",output)
  if output==[0]:
    prediction="SVM Predict that new customer will leave the bank "
   

  if output==[1]:
    prediction="SVM Predict that new customer will not leave the bank "
    
    
  print(prediction)
  return prediction
def main():
    
    html_temp = """
   <div class="" style="background-color:blue;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:40px;color:white;margin-top:10px;">Poornima Institute of Engineering & Technology</p></center> 
   <center><p style="font-size:30px;color:white;margin-top:10px;">Department of Computer Engineering</p></center> 
   <center><p style="font-size:20px;color:white;margin-top:10px;">MID TERM 1 Practical by PIET18CS170</p></center> 
   </div>
   </div>
   </div>
   """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.header("Predict new customer will leave the bank or not using  SVM ")
    CreditScore = st.number_input('Enter  Credit Score',300,1000)
    Geography = st.number_input('Insert Geography 0 for France and 1 for Spain',0,1)
    Gender = st.number_input('Insert gender 0 for Male and 1 for Female',0,1)
    Age = st.number_input('Insert a Age',18,80)
    Tenure = st.number_input('Insert a Tenure',0,9)
    Balance = st.number_input('Enter your Account Balance')
    HasCrCard= st.number_input('Inster hasCard or not 1 for yes 0 for no ',0,1)
    IsActiveMember= st.number_input('Insert a member is active or not 1 for yes 0 for no',0,1)
    EstimatedSalary  = st.number_input('Enter Estimated salary ',0,10000000)
   
    
    resul=""
    if st.button("Predict"):
      result=predict_note_authentication(CreditScore,Geography,Gender,Age,Tenure,Balance,HasCrCard,IsActiveMember,EstimatedSalary)
      st.success('Model has predicted {}'.format(result))
    if st.button("About"):
      st.subheader("Developed by Shruti Mittal 1st Mid-Term ")
      st.subheader("C-Section,PIET")

if __name__=='__main__':
  main()
   
