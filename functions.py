import pandas as pd

pd.options.display.max_columns = 20
pd.options.display.width = 200

import numpy as np
from _collections import OrderedDict

from sklearn.preprocessing import StandardScaler, LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from sklearn.impute import KNNImputer, SimpleImputer
from imblearn.over_sampling import SMOTE, ADASYN


# Functions for cleaning input data
def make_input_df(inputslist):
    # Test list
    # ['18', '0', '0', 'Within the past year', 'M', 'Excellent', '<$50k', 'Very simple', 'A great deal', 'Very easy',
    #  'Retired', 'Less than high school', 'Nonparent', 'No', 'No', 'No', 'No', 'No', 'Select a condition']

    cols = ['age', 'n_prescriptions', 'n_provider_visits',
            'first_got_rx', 'sex', 'general_health', 'income', 'med_burden',
            'understand_health_prob', 'can_afford_rx', 'emply', 'educ', 'parent',
            'has_diabetes', 'has_hyperten', 'has_asthma_etc', 'has_heart_condition', 'has_hi_cholesterol', ]

    inputslist = np.array(inputslist).reshape(1, 18)

    inputslist = pd.DataFrame(inputslist, columns=cols)

    # Rearrange columns
    inputslist = inputslist[['age', 'n_prescriptions', 'n_provider_visits',
                             'first_got_rx', 'income', 'general_health', 'med_burden', 'can_afford_rx',
                             'understand_health_prob', 'educ', 'sex', 'emply',
                             'has_diabetes', 'has_hyperten', 'has_asthma_etc', 'has_heart_condition',
                             'has_hi_cholesterol',
                             'parent']]

    return inputslist


# r = ['18', '0', '0', 'Within the past year', 'M', 'Excellent', '<$50k', 'Very simple', 'A great deal', 'Very easy',
#      'Retired', 'Less than high school', 'Nonparent', 'No', 'No', 'No', 'No', 'No', 'Select a condition']
# r = r[0:-1]
# r = make_input_df(r)


def encode_ordinals(df):
    # df = pd.DataFrame(df)
    # First started taking an rx on a regular basis
    first_started_taking_vals = {
        'Within the past year': 1,
        '1 to 2 years ago': 2,
        '3 to 5 years ago': 3,
        '6 to 10 years ago': 4,
        'More than 10 years ago': 5
    }
    df['first_got_rx'].replace(first_started_taking_vals, inplace=True)

    # General health
    gen_val = {
        'Excellent': 5,
        'Very good': 4,
        'Good': 3,
        'Fair': 2,
        'Poor': 1
    }
    df['general_health'].replace(gen_val, inplace=True)

    # Income
    income_vals = {
        'No response/Unknown': 0,
        '<$50k': 1,
        '$50k-75k': 2,
        '>$100k': 3
    }
    df['income'].replace(income_vals, inplace=True)

    # Med burden
    med_burden_vals = {
        'Very simple': 1,
        'Somewhat simple': 2,
        'Somewhat complicated': 3,
        'Very complicated': 4
    }
    df['med_burden'].replace(med_burden_vals, inplace=True)

    # Can afford rx
    can_afford_rx_vals = {
        'Very easy': 1,
        'Somewhat easy': 2,
        'Somewhat difficult': 3,
        'Very difficult': 4
    }
    df['can_afford_rx'].replace(can_afford_rx_vals, inplace=True)

    # Understand health prob
    understand_health_prob_vals = {
        'A great deal': 4,
        'Somewhat': 3,
        'Not so much': 2,
        'Not at all': 1,
        'Unknown/Refused': 0,
    }
    df['understand_health_prob'].replace(understand_health_prob_vals, inplace=True)

    # Education
    educ_vals = {
        'Less than high school': 1,
        'High school': 2,
        'Some college': 3,
        'Technical school/other': 4,
        'College graduate': 5,
        'Graduate school or more': 6,
    }
    df['educ'].replace(educ_vals, inplace=True)

    df = df.reset_index(drop=True)

    return df


def process_data(X_original, x_input):
    combined_df = pd.concat([X_original.iloc[1:, ], x_input], axis=0)

    combined_df = combined_df.sort_index()

    numeric_cols = ['age', 'n_prescriptions', 'n_provider_visits']
    ordinal_cols = ['first_got_rx', 'income', 'general_health', 'med_burden', 'can_afford_rx', 'understand_health_prob',
                    'educ', ]
    categorical_cols = ['sex',
                        # 'have_health_insur','have_medicare', 'metro',
                        'parent',
                        # 'US_region',
                        # 'ownhome',
                        # 'mstatus',
                        'emply',
                        'has_diabetes', 'has_hyperten', 'has_asthma_etc',
                        'has_heart_condition', 'has_hi_cholesterol']

    pipeline = Pipeline([
        ('scale_numeric', ColumnTransformer(
            [('scale', StandardScaler(), numeric_cols),
             ('encode_ord', FunctionTransformer(encode_ordinals), ordinal_cols),
             ('get dummies', OneHotEncoder(drop='first'), categorical_cols),
             ],
            remainder='passthrough'
        ))
    ])

    dummy_col_names = [str(i) for i in pd.get_dummies(combined_df[categorical_cols], drop_first=True).columns]
    columns_ = [numeric_cols + ordinal_cols + dummy_col_names]

    processed_df = pd.DataFrame(pipeline.fit_transform(combined_df), columns=columns_[0])

    return processed_df


def scale_data(input_data):
    # Feature Scaling
    scaler = StandardScaler()
    input_data = scaler.fit_transform(input_data)


from required_files import plm

# Functions for getting the medical condition cost, burden, side effects (for pie charts)
def get_cost(df, input, condition_or_drug):
    if condition_or_drug == 'condition':
        col = 'primary_condition'
    elif condition_or_drug == 'drug':
        col = 'drug_name'

    cost_vals = {
        '< $25 monthly': 'Less than $25 monthly'
    }
    df['cost'].replace(cost_vals, inplace=True)
    cond = df[df[col] == input].groupby(['cost'])['cost'].describe().to_dict()['count']
    total_vals = sum(cond.values())
    perc = [(i / total_vals) * 100 for i in cond.values()]
    # return cond.keys(), cond.values(), np.floor(perc)
    return cond.keys(), cond.values()


def get_burden(df, input, condition_or_drug):
    if condition_or_drug == 'condition':
        col = 'primary_condition'
    elif condition_or_drug == 'drug':
        col = 'drug_name'

    cond = df[df[col] == input].groupby(['burden'])['burden'].describe().to_dict()['count']
    return cond.keys(), cond.values()


def get_sideeffects(df, input, condition_or_drug):
    if condition_or_drug == 'condition':
        col = 'primary_condition'
    elif condition_or_drug == 'drug':
        col = 'drug_name'

    cond = df[df[col] == input].groupby(['side_effects'])['side_effects'].describe().to_dict()['count']
    return cond.keys(), cond.values()



from colour import Color

red = Color("red")
colors = list(red.range_to(Color("white"), 6))

blue = Color('blue')
colors = list(blue.range_to(Color("white"), 6))


def color_cost_chart(vals_labels):
    colors5 = ["#ffe6e6", "#ffb3b3", "#ff0000", '#cc0000', '#800000']
    costlist = []
    for i in vals_labels:
        if 'Less than $25 monthly' in i:
            i = list(i)
            i.append(colors5[0])
            costlist.append(tuple(i))
        elif '$25-49 monthly' in i:
            i = list(i)
            i.append(colors5[1])
            costlist.append(tuple(i))
        elif '$50-99 monthly' in i:
            i = list(i)
            i.append(colors5[2])
            costlist.append(tuple(i))
        elif '$100-199 monthly' in i:
            i = list(i)
            i.append(colors5[3])
            costlist.append(tuple(i))
        elif '$200+ monthly' in i:
            i = list(i)
            i.append(colors5[4])
            costlist.append(tuple(i))

    return costlist


def color_burden_chart(vals_labels):
    colors5 = ["#e8f8fc", '#47caeb', "#17aacf", "#1284a1"]
    burdenlist = []
    for i in vals_labels:
        if 'Not at all hard to take' in i:
            i = list(i)
            i.append(colors5[0])
            burdenlist.append(tuple(i))
        elif 'A little hard to take' in i:
            i = list(i)
            i.append(colors5[1])
            burdenlist.append(tuple(i))
        elif 'Somewhat hard to take' in i:
            i = list(i)
            i.append(colors5[2])
            burdenlist.append(tuple(i))
        elif 'Very hard to take' in i:
            i = list(i)
            i.append(colors5[3])
            burdenlist.append(tuple(i))

    return burdenlist


def color_side_effects_chart(vals_labels):
    colors5 = ["#f2e6ff", '#cc99ff', "#8c1aff", "#4d0099"]
    sideeffectslist = []
    for i in vals_labels:
        if 'None' in i:
            i = list(i)
            i.append(colors5[0])
            sideeffectslist.append(tuple(i))
        if 'Mild' in i:
            i = list(i)
            i.append(colors5[1])
            sideeffectslist.append(tuple(i))
        if 'Moderate' in i:
            i = list(i)
            i.append(colors5[2])
            sideeffectslist.append(tuple(i))
        if 'Severe' in i:
            i = list(i)
            i.append(colors5[3])
            sideeffectslist.append(tuple(i))

    return sideeffectslist


# For reporting the maximum value in each category
def return_max_value(type_, vals_labels):
    res = OrderedDict()

    max_ = max(vals_labels)
    total = sum([i[0] for i in vals_labels])

    res['type'] = type_
    res['val'] = max_[1].lower()
    res['percent'] = round(max_[0] / total * 100, 1)

    return res
