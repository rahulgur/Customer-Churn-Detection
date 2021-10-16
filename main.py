import boto3
import argparse
import numpy as np
import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import sys
from io import StringIO
import shutil
from sklearn.compose import ColumnTransformer
from sklearn.externals import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import Binarizer, StandardScaler, OneHotEncoder
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score,classification_report
from sklearn.metrics import f1_score
from sklearn.ensemble import GradientBoostingClassifier


# Dictionary to convert indices to labels
INDEX_TO_LABEL = {
    0: 'Non-Churn',
    1: 'Churn'
}




"""
model_fn
    model_dir: (sting) specifies location of saved model

This function is used by AWS Sagemaker to load the model for deployment. 
It does this by simply loading the model that was saved at the end of the 
__main__ training block above and returning it to be used by the predict_fn
function below.
"""
def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir,"test_model.joblib"))
    return model

"""
input_fn
    request_body: the body of the request sent to the model. The type can vary.
    request_content_type: (string) specifies the format/variable type of the request

This function is used by AWS Sagemaker to format a request body that is sent to 
the deployed model.
In order to do this, we must transform the request body into a numpy array and
return that array to be used by the predict_fn function below.

Note: Oftentimes, you will have multiple cases in order to
handle various request_content_types. Howver, in this simple case, we are 
only going to accept text/csv and raise an error for all other formats.
"""



def input_fn(request_body,content_type):
    return request_body
    
    '''samples = []
    for i in request_body.split('\n'):
        list1 = []
        for j in i.split(','):
            if isfloat(j)==True:
                list1.append(float(j))
            else:
                list1.append(j)
        samples.append(list1)
    return np.array(samples,dtype='object')'''
    

"""
predict_fn
    input_data: (numpy array) returned array from input_fn above 
    model (sklearn model) returned model loaded from model_fn above

This function is used by AWS Sagemaker to make the prediction on the data
formatted by the input_fn above using the trained model.
"""
def predict_fn(input_data, model):
    
    def isfloat(value):
      try:
        float(value)
        return True
      except ValueError:
        return False
    
    def isint(value):
      try:
        int(value)
        return True
      except ValueError:
        return False
    
    '''samples = []
    for i in input_data.split('\n'):
        list1 = []
        for j in i.split(','):
            if isint(j)==True:
                list1.append(int(j))
            elif isfloat(j)==True:
                list1.append(float(j))
            else:
                list1.append(j)
        samples.append(list1)
    samples = np.array(samples)'''

    samples = []
    for row in input_data.split('\n'):
        list1 = []
        if row=="":
            continue
        row = str(row).replace("b'", "") 
        row = row.replace("\\n'", "") 
        row_to_list = row.split(',')
        for i in row_to_list:
            if isint(i)==True:
                list1.append(int(i))
            elif isfloat(i)==True:
                list1.append(float(i))
            else:
                list1.append(i)
        samples.append(list1)
    
    df = pd.DataFrame(samples,columns=   ['Customer_Age','Gender','Dependent_count','Education_Level','Marital_Status','Income_Category','Card_Category','Months_on_book','Total_Relationship_Count','Months_Inactive_12_mon','Contacts_Count_12_mon','Credit_Limit','Total_Revolving_Bal','Avg_Open_To_Buy','Total_Amt_Chng_Q4_Q1','Total_Trans_Amt','Total_Trans_Ct','Total_Ct_Chng_Q4_Q1','Avg_Utilization_Ratio'])

    return model.predict(df)

"""
output_fn
    prediction: the returned value from predict_fn above
    content_type: (string) the content type the endpoint expects to be returned

This function reformats the predictions returned from predict_fn to the final
format that will be returned as the API call response.

Note: While we don't use content_type in this example, oftentimes you will use
that argument to handle different expected return types.
"""
def output_fn(prediction, content_type):
    return '|'.join([INDEX_TO_LABEL[t] for t in prediction])





if __name__ =='__main__':
   parser = argparse.ArgumentParser()

   parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
   parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
   parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
   parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))

   args = parser.parse_args()


   
   cr_data = pd.read_csv(os.path.join(args.train,'train.csv'), index_col=0, engine="python")
    
   
   X=cr_data.drop(['CLIENTNUM','Churn_Status'],axis=1)
   y=cr_data[['Churn_Status']]

  
   numerical_features = ['Customer_Age',  'Dependent_count', 'Months_on_book','Total_Relationship_Count', 'Months_Inactive_12_mon','Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal','Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt','Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio']


   
   categorical_features = ['Gender','Education_Level','Marital_Status','Income_Category','Card_Category']


   

  
   

   #Numerical operations contain data cleaning and feature engineering
   numeric_transformer = Pipeline(steps=[
      ('imputer', SimpleImputer(strategy='median')),
      ('scaler', StandardScaler())
         
   ])

   #Categorical operations contain data cleaning and feature engineering
   categorical_transformer = Pipeline(steps=[
         ('imputer', SimpleImputer(strategy='constant'))
         ,('encoder',  OneHotEncoder())
   ])








   #transformation performing on the data
   preprocessor = ColumnTransformer(
      transformers=[
      ('numeric', numeric_transformer, numerical_features)
      ,('categorical', categorical_transformer, categorical_features)
   ])
   #print(preprocessor)



   #Splitting the data in train test
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=167566, stratify=y)

   #Loading the model 
   #model = XGBClassifier(n_estimators=1000, learning_rate=0.1, objective='binary:logistic' )
   #clf = RandomForestClassifier(n_estimators=10)
   clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)




   #composing all the steps 
   model_pipeline = Pipeline(steps = [
                  ('preprocess',preprocessor),
               ('classifier',clf)
            ])


   #Data passing to model for traing
   model_pipeline.fit(X_train,y_train)

   #saving the train model with pipeline
   print(args.model_dir)
   joblib.dump(model_pipeline, os.path.join(args.model_dir, "test_model.joblib"))

   




