import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder

# Functions for cleaning input data
def make_input_df(inputslist):
    # Test list
    # rlist = ['50', '1', '2', 'F', 'Excellent', '<$50k', 'Very simple', 'Has health insurance', 'Not eligible',
    #          'A great deal', 'Very easy', 'Center City', 'Homeowner', 'Married', 'Retired', 'Less than high school',
    #          'No', 'Yes', 'Yes', 'No', 'No']

    cols = ['age', 'n_prescriptions', 'n_provider_visits', 'sex', 'general_health', 'income', 'med_burden',
            'have_health_insur', 'have_medicare',
            'understand_health_prob', 'can_afford_rx', 'metro', 'ownhome', 'mstatus', 'emply', 'educ', 'has_diabetes',
            'has_hyperten',
            'has_asthma_etc', 'has_heart_condition', 'has_hi_cholesterol']
    inputslist = np.array(inputslist).reshape(1, 21)

    inputslist = pd.DataFrame(inputslist, columns=cols)

    # Rearrange columns
    inputslist = inputslist[['sex','age', 'n_prescriptions', 'general_health', 'income', 'n_provider_visits', 'med_burden',
               'have_health_insur', 'have_medicare',
               'can_afford_rx', 'understand_health_prob', 'metro', 'ownhome', 'mstatus', 'emply', 'educ', 'has_diabetes',
               'has_hyperten',
               'has_asthma_etc', 'has_heart_condition', 'has_hi_cholesterol']]

    return inputslist


def process_data(X_original, x_input):
    combined_df = pd.concat([X_original.iloc[1:,], x_input], axis=0)

    combined_df = combined_df.sort_index()
    combined_df['age'] = combined_df['age'].astype('float64')
    combined_df['n_prescriptions'] = combined_df['n_prescriptions'].astype('float64')
    combined_df['n_provider_visits'] = combined_df['n_provider_visits'].astype('float64')

    X = pd.get_dummies(combined_df, drop_first=True)

    X = X.drop(columns=['med_burden_Unknown', 'educ_Unknown/Refused'], axis=1)
    return X

def scale_data(input_data):
    # Feature Scaling
    scaler = StandardScaler()
    input_data = scaler.fit_transform(input_data)

# Functions for getting the medical condition cost, burden, side effects (for pie charts)

def get_condition_cost(df, input):
    cost_vals = {
        '< $25 monthly' : 'Less than $25 monthly'
    }
    df['cost'].replace(cost_vals, inplace=True)
    cond = df[df['primary_condition']==input].groupby(['cost'])['cost'].describe().to_dict()['count']
    return cond.keys(), cond.values()

def get_condition_burden(df, input):
    cond = df[df['primary_condition']==input].groupby(['burden'])['burden'].describe().to_dict()['count']
    return cond.keys(), cond.values()

def get_condition_sideeffects(df, input):
    cond = df[df['primary_condition']==input].groupby(['side_effects'])['side_effects'].describe().to_dict()['count']
    return cond.keys(), cond.values()
