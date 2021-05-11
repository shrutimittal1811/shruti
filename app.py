import streamlit as st 
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
st.set_option('deprecation.showfileUploaderEncoding', False)
# Load the pickled model
pickle_in = open("ShrutiMittalPIET18CS170.pkl","rb")
model=pickle.load(pickle_in)
dataset= pd.read_csv('Classification Dataset1.csv')


X = dataset.iloc[:, 1:10].values

#handling missing data (Replacing missing data with the constant value)  
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.NAN, strategy= 'constant', fill_value="Female", verbose=1, copy=True)
#Fitting imputer object to the independent variables x.   
imputer = imputer.fit(X[:, 2:3]) 
#Replacing missing data with the calculated constant value  
X[:, 2:3]= imputer.transform(X[:,2:3]) 


# Encoding Categorical data:
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 2] = labelencoder_X.fit_transform(X[:, 2])

# Encoding Categorical data:
# Encoding the Independent Variable

labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)


def predict_note_authentication(CreditScore,Geography,Gender,Age, Tenure, Balance, HasCrCard, IsActiveMembe ,EstimatedSalary):
  output= model.predict(sc.transform([[CreditScore,Geography,Gender,Age, Tenure, Balance, HasCrCard, IsActiveMembe ,EstimatedSalary]]))
  print("modal is predicted ",output)
  if output==[0]:
    prediction="Customer will leave"
   

  if output==[1]:
    prediction="Customer will not leave"
    
    
  print(prediction)
  return prediction
def main():
    
    html_temp = """
   <div class="" style="background-color:blue;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:40px;color:white;margin-top:10px;">Poornima Institute of Engineering & Technology</p></center> 
   <center><p style="font-size:30px;color:white;margin-top:10px;">Department of Computer Engineering</p></center> 
   <center><p style="font-size:20px;color:white;margin-top:10px;">MID TERM 1 pactice by PIET18CS170 Shruti Mittal</p></center> 
   </div>
   </div>
   </div>
   """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.header("SVM modal is predicted whether the customer will leave or not")
    CreditScore = st.number_input('Enter Credit Score',0,900)
    Geography = st.number_input('Insert country 0 for Spain and 1 for France',0,1)
    Gender = st.number_input('Insert gender 0 for male and 1 for female',0,1)
    Age = st.number_input('Enter Age',18,100)
    Tenure = st.number_input('Enter Tenure ',0,10)
    Balance = st.number_input('Enter Balance ')
    HasCrCard = st.number_input('Enter 0 if no Credit card 1 if have Credit card ',0,1)
    IsActiveMembe= st.number_input('Enter 0 if not member 1 if member ',0,1)
    EstimatedSalary=st.number_input('Enter EstimatedSalary')
   
    
    resul=""
    if st.button("Predict"):
      result=predict_note_authentication(CreditScore,Geography,Gender,Age, Tenure, Balance, HasCrCard, IsActiveMembe ,EstimatedSalary)
      st.success('Model has predicted {}'.format(result))
    if st.button("About"):
      st.subheader("Developed by Shruti Mittal 1st mid term ")
      st.subheader("C-Section,PIET")

if __name__=='__main__':
  main()
   
