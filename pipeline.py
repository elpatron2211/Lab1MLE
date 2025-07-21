import pandas as pd

data = pd.read_csv('loan_data.csv')

data['gender'] =  data['person_gender'].map({'male':1, 'female':0})

education_map = {
    'High School': 0,
    'Associate': 1,
    'Bachelor': 2,
    'Master': 3,
    'Doctorate': 4
}
data['education'] = data['person_education'].map(education_map)

data = pd.get_dummies(data, columns=['person_home_ownership'])
data = pd.get_dummies(data, columns=['loan_intent']) #remember to drop intent 'Personal' for logistic regression
data['previous_loan_defaults_on_file'] = data['previous_loan_defaults_on_file'].map({'Yes': 1, 'No': 0})

data.drop(columns=['person_home_ownership_OTHER'], inplace=True, errors='ignore')
data.drop(columns=['person_gender', 'person_education'], inplace=True, errors='ignore')
data.drop(columns = ['loan_int_rate'], inplace=True, errors='ignore') #se quitó porque no tiene relevancia para autorizar un préstamo

y = data['loan_status']

data.drop(columns=['loan_status'], inplace=True, errors='ignore')

data.to_json('processed_for_random_forest.json', index=False, orient='records')

data.drop(columns = ['loan_intent_PERSONAL'], inplace=True, errors='ignore') #para evitar multicolinealidad en regresión logística

data.to_json('processed_for_logistic_regression.json', index=False, orient='records')

y.to_json('target_variable.json', index=False, orient='records')