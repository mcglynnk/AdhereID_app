# Setup
from flask import Flask, render_template, request

# Initialize application
application = Flask(__name__)

# Setup
import pandas as pd
import numpy as np
import pickle

# Home page
@application.route('/', methods=['POST', 'GET'])
def home1():
    condfile = r'C:\Users\Kelly\Documents\Python\AdhereID_app_charts\cond_list.txt'
    with open(condfile, 'rb') as f:
        cond_list = pickle.load(f)

    return render_template("index.html", cond_list=cond_list)

# Load ML Model
filename = r'lr_model.sav'
with open(filename, 'rb') as file:
    lr_model = pickle.load(file)


# Import functions for cleaning input data
from functions import make_input_df, process_data
from required_files import X

# Import functions for getting the medical condition cost, burden, side effects (for pie charts)
from functions import get_condition_cost, get_condition_burden, get_condition_sideeffects
from required_files import plm


show_charts=False
@application.route('/result', methods=['POST', 'GET'])
def result1():
    global show_charts

    if request.method == 'POST':

        result_list = request.form.to_dict()
        result_list = list(result_list.values())
        print('result list: ', result_list)
        result_list_without_condition = result_list[0:-1]

        try:
            inputs_df = make_input_df(result_list_without_condition)
        except ValueError as e:
            return "Please fill out all the fields!"

        inputs_df_with_X = process_data(X, inputs_df)

        inputs_df_with_X_array = np.array(inputs_df_with_X.iloc[0,]).reshape(1, 52)

        predictions = lr_model.predict(inputs_df_with_X_array)

        if predictions == 1.0:
            predictions = "High risk of non-adherence!"
        elif predictions == 0:
            predictions = "Low risk of non-adherence!"
        else:
            None


        condfile = r'C:\Users\Kelly\Documents\Python\AdhereID_app_charts\cond_list.txt'
        with open(condfile, 'rb') as f:
            cond_list = pickle.load(f)

        if 'Select a condition' not in result_list:
            show_charts = True
        else:
            None
        print(result_list)
        if show_charts==True:
            colors = [
                "#F7464A", "#46BFBD", "#FDB45C", "#FEDCBA",
                "#ABCDEF", "#DDDDDD", "#ABCABC", "#4169E1",
                "#C71585", "#FF4500", "#FEDCBA", "#46BFBD"]

            costlabels, costvalues = get_condition_cost(plm, result_list[-1])
            costcolors = colors[0:len(costlabels)]

            burdenlabels, burdenvalues = get_condition_burden(plm, result_list[-1])
            burdencolors = colors[0:len(costlabels)]

            sideeffectslabels, sideeffectsvalues = get_condition_sideeffects(plm, result_list[-1])
            sideeffectscolors = colors[0:len(costlabels)]
        else:
            costlabels, costvalues, costcolors = [0],[0],[0]
            burdenlabels, burdenvalues, burdencolors = [0],[0],[0]
            sideeffectslabels, sideeffectsvalues, sideeffectscolors = [0],[0],[0]

        print(show_charts)
        # print(labels, values, colors)
        return render_template("result.html", max=17000,
                               predictions=predictions,
                               show_charts= show_charts,
                               costset=zip(costvalues, costlabels, costcolors),
                               burdenset=zip(burdenvalues, burdenlabels, burdencolors),
                               sideeffectsset=zip(sideeffectsvalues, sideeffectslabels, sideeffectscolors),
                               cond_list=cond_list,
                               result_list=result_list
                               )


if __name__ == '__main__':
    application.run(debug=True)
