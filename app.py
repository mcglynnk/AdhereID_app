# Setup
from flask import Flask, render_template, request

import pandas as pd
import sklearn
import numpy as np
import pickle
from required_files import res_url, conditions_list_file

# Initialize application
app = Flask(__name__)


# Home page
@app.route('/', methods=['POST', 'GET'])
def home1():
    # List of available medical conditions for the bottom drop-down menu
    condfile = conditions_list_file
    with open(condfile, 'rb') as f:
        cond_list = pickle.load(f)

    return render_template("index.html", cond_list=cond_list, res_url=res_url)

# Load ML Model
filename = r'files/lr_model.sav'
with open(filename, 'rb') as file:
    lr_model = pickle.load(file)

# Import functions for cleaning input data
from functions import make_input_df, process_data
from required_files import X

# Import functions for getting the medical condition cost, burden, side effects (for pie charts)
from functions import get_condition_cost, get_condition_burden, get_condition_sideeffects
from functions import color_cost_chart, color_burden_chart, color_side_effects_chart
from required_files import plm

show_charts = False


@app.route('/result', methods=['POST', 'GET'])
def result1():
    global show_charts

    if request.method == 'POST':
        # Get results after hitting submit
        result_list = request.form.to_dict()
        result_list = list(result_list.values())
        print('result list: ', result_list)
        result_list_without_condition = result_list[0:-1]

        # Goes to a blank html page containing "Please fill out all the fields!" if the user doesn't fill out the
        # whole form
        try:
            inputs_df = make_input_df(result_list_without_condition)
        except ValueError as e:
            return "Please fill out all the fields!"

        # Process input data in order to feed it into the logistic regression model
        inputs_df_with_X = process_data(X, inputs_df)

        inputs_df_with_X_array = np.array(inputs_df_with_X.iloc[0,]).reshape(1, 22)

        predictions = lr_model.predict(inputs_df_with_X_array)

        # Prints out a message with the results
        if predictions == 1.0:
            predictions = "High risk of non-adherence!"
        elif predictions == 0:
            predictions = "Low risk of non-adherence!"
        else:
            None

        # List of available medical conditions for the bottom drop-down menu, also used to write the result message
        # 'Patients with {} report...'.format(result_list[-1])
        condfile = conditions_list_file
        with open(condfile, 'rb') as f:
            cond_list = pickle.load(f)

        # Select a medical condition is optional. If selected, prints pie charts on results page. This code is a switch.
        if 'Select a condition' not in result_list:
            show_charts = True
        else:
            show_charts = False

        # Testing
        print(result_list)
        print(show_charts)

        # Generates pie charts from plm data
        if show_charts == True:

            costlabels, costvalues = get_condition_cost(plm, result_list[-1])
            cost_vals_labels = list(zip(costvalues, costlabels))
            costcolors = color_cost_chart(cost_vals_labels)
            print(costcolors)

            burdenlabels, burdenvalues = get_condition_burden(plm, result_list[-1])
            burden_vals_labels = list(zip(burdenvalues, burdenlabels))
            burdencolors = color_burden_chart(burden_vals_labels)
            print(burdencolors)

            sideeffectslabels, sideeffectsvalues = get_condition_sideeffects(plm, result_list[-1])
            sideeffects_vals_labels = list(zip(sideeffectsvalues, sideeffectslabels))
            sideeffectscolors = color_side_effects_chart(sideeffects_vals_labels)
            print(sideeffectscolors)

        else:
            costlabels, costvalues, costcolors = [0], [0], [0]
            burdenlabels, burdenvalues, burdencolors = [0], [0], [0]
            sideeffectslabels, sideeffectsvalues, sideeffectscolors = [0], [0], [0]

        # Testing
        # print(show_charts)
        # print(labels, values, colors)

        # Return the results page!
        return render_template("result.html", max=17000,
                               predictions=predictions,  # ML prediction result
                               # For pie charts:
                               show_charts=show_charts,
                               costset=costcolors,
                               burdenset=burdencolors,
                               sideeffectsset=sideeffectscolors,
                               # For inserting python variables into the html files:
                               cond_list=cond_list,
                               result_list=result_list
                               )


if __name__ == '__main__':
    # app.run(debug=True)
    app.run(debug=False, host='0.0.0.0', port=80)


