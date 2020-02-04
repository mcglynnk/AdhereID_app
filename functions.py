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
    cond = df[df['primary_condition'] == input].groupby(['burden'])['burden'].describe().to_dict()['count']
    return cond.keys(), cond.values()


def get_condition_sideeffects(df, input):
    cond = df[df['primary_condition'] == input].groupby(['side_effects'])['side_effects'].describe().to_dict()['count']
    return cond.keys(), cond.values()


#
#
#
#
# ##
#     # Test list
# rlist = ['50', '1', '2', 'F', 'Excellent', '<$50k', 'Very simple', 'Has health insurance', 'Not eligible',
#          'A great deal', 'Very easy', 'Center City', 'Homeowner', 'Married', 'Retired', 'Less than high school',
#          'No', 'Yes', 'Yes', 'No', 'No', 'epilepsy']
# colors = [
#                 "#F7464A", "#46BFBD", "#FDB45C", "#FEDCBA",
#                 "#ABCDEF", "#DDDDDD", "#ABCABC", "#4169E1",
#                 "#C71585", "#FF4500", "#FEDCBA", "#46BFBD"]
#
# costlabels, costvalues = get_condition_cost(plm, rlist[-1])
# # costdict = get_condition_cost(plm, rlist[-1])
# costcolors = colors[0:len(costlabels)]
#
# cl = list(zip(costvalues, costlabels, costcolors))
# bd = cl = list(zip(costvalues, costlabels, costcolors))
# cl2 = list(zip(costvalues, costlabels))
# dict(zip(costvalues, costlabels, costcolors))
#
# ##


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

# color_pie_charts('cost', vals_labels=cl2)


# def color_pie_charts(piechart):
#     if piechart=='cost':
#         new_dict = {}
#
#         colordict_ = {'Less than $25 monthly': '#ABCDEF',
#                       '$25-49 monthly':'#FDB45C'
#                       }
#
#         colorlist = []
#         for k in colordict_:
#             if k in costdict.keys():
#                 colorlist.append(colordict_[k])
#                 #new_dict[k] = new_dict[k] + v
#
#         print(colorlist)
#         print(costdict)
#         print(list(zip(costvalues, costlabels, colorlist)))
#
# color_pie_charts('cost')
