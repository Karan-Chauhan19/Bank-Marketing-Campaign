'''
Author : Karan Chauhan
github : @Karan-Chauhan19
Organization : L.J University
'''
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE



class featureEngineering :
    def __init__(self) :
        pass
    def clean_data(self) :
        df = pd.read_csv('/home/karan-chauhan/WorkStation/Project/Bank-Marketing-Campaign/Data/bank.csv')

        #Rename column name
        df = df.rename(columns={'y':'deposit','housing':'house_loan','loan':'personal_loan','previous':'pre_campaign'})
        df = df.drop(columns=['balance','age','day','default'])

        return df

    def get_clean_data(self) :
        df = featureEngineering().clean_data()
        df_processed = df.copy()

        # Define columns for preprocessing
        categorical_column_ohe = ['job','marital','education','contact','poutcome','month','house_loan','personal_loan']
        numerical_column = ['duration','campaign','pdays','pre_campaign']

        # Create a copy of the DataFrame before preprocessing to avoid in-place modifications
        df_processed = df.copy()  

        # Apply LabelEncoder to the target variable in the copied DataFrame
        le = LabelEncoder()
        df_processed['deposit'] = le.fit_transform(df_processed['deposit']) 

        # Create the preprocessing pipeline
        preprocessor = ColumnTransformer(transformers=[
            ('trf1',OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first'), categorical_column_ohe), # Added sparse=False
            ('trf2',StandardScaler(), numerical_column)
        ])

        # Create the pipeline
        model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor)
        ])

        # Fit and transform the data using the copied DataFrame
        X = model_pipeline.fit_transform(df_processed.drop(columns=['deposit']))

        # Get feature names
        categorical_feature_names = model_pipeline.named_steps['preprocessor'].named_transformers_['trf1'].get_feature_names_out(categorical_column_ohe)
        numerical_feature_names = model_pipeline.named_steps['preprocessor'].named_transformers_['trf2'].get_feature_names_out(numerical_column)
        all_feature_names =  list(categorical_feature_names) + list(numerical_feature_names) 

        # Create the transformed DataFrame
        X_transformed = pd.DataFrame(X, columns=all_feature_names)
        X_transformed['deposit'] = df_processed['deposit']
        #For handle imbalanced data
        smote = SMOTE()
        X_resampled,y_resampled = smote.fit_resample(X_transformed.iloc[:,0:-1],X_transformed.iloc[:,-1])

        smote_df = pd.DataFrame(X_resampled, columns=all_feature_names)
        smote_df['deposit'] = y_resampled

        return smote_df
    
