# Setup
from flask import Flask, render_template, request

import pandas as pd
import sklearn
import numpy as np
import pickle
from _collections import OrderedDict
import requests
from bs4 import BeautifulSoup
import re

from required_files import res_url, conditions_list_file, drugs_list_file

# Initialize application
app = Flask(__name__)


# Home page
@app.route('/', methods=['POST', 'GET'])
def home1():
    # List of available medical conditions for the bottom drop-down menu
    with open(conditions_list_file, 'rb') as f:
        cond_list = pickle.load(f)
    cond_list = sorted(cond_list)

    with open(drugs_list_file, 'rb') as f2:
        drug_list = pickle.load(f2)

    return render_template("index.html", cond_list=cond_list, drug_list=drug_list, res_url=res_url)


@app.route('/slides', methods=['POST', 'GET'])
def pres():
    return render_template("pres.html")


# Load ML Model
filename = r'files/lr_model.sav'
with open(filename, 'rb') as file:
    lr_model = pickle.load(file)

# Import functions for cleaning input data
from functions import make_input_df, process_data
from required_files import X

# Import functions for getting the medical condition cost, burden, side effects (for pie charts)
from functions import get_cost, get_burden, get_sideeffects
from functions import color_cost_chart, color_burden_chart, color_side_effects_chart
from functions import return_max_value
from required_files import plm, drugbank_df


@app.route('/result', methods=['POST', 'GET'])
def result1():
    if request.method == 'POST':
        # Get results after hitting submit
        result_list = request.form.to_dict()
        result_list = list(result_list.values())
        print('result list: ', result_list)
        result_list_without_condition = result_list[0:-2]

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

        # List of available medical conditions and drugs for the bottom drop-down menu, also used to write the result
        # message 'Patients with {} report...'.format(result_list[-1])
        condfile = conditions_list_file
        with open(condfile, 'rb') as f:
            cond_list = pickle.load(f)
        cond_list = sorted(cond_list)

        with open(drugs_list_file, 'rb') as f2:
            drug_list = pickle.load(f2)

        # Select a medical condition is optional. If selected, prints pie charts on results page. This code is a switch.
        print(result_list)

        if 'Select a condition' in result_list and 'Select a drug' in result_list:
            show_charts = False
        else:
            show_charts = True

        # Generates pie charts from PatientsLikeMe data
        if show_charts == True:
            skip_charts = False
            # Return pie charts and text based on medical condition selected
            if 'Select a drug' and not 'Select a condition' in result_list:  # If 'Select a drug' is in the list, this
                drug_selected = False  # is the default value. User must have
                max_results = []  # selected a medical condition instead.
                drug_url, nav_d, nav_s = None, None, None

                costlabels, costvalues = get_cost(plm, result_list[-2], 'condition')
                cost_vals_labels = list(zip(costvalues, costlabels))
                costcolors = color_cost_chart(cost_vals_labels)
                max_results.append(return_max_value('cost', cost_vals_labels))

                burdenlabels, burdenvalues = get_burden(plm, result_list[-2], 'condition')
                burden_vals_labels = list(zip(burdenvalues, burdenlabels))
                burdencolors = color_burden_chart(burden_vals_labels)
                max_results.append(return_max_value('burden', burden_vals_labels))

                sideeffectslabels, sideeffectsvalues = get_sideeffects(plm, result_list[-2], 'condition')
                sideeffects_vals_labels = list(zip(sideeffectsvalues, sideeffectslabels))
                sideeffectscolors = color_side_effects_chart(sideeffects_vals_labels)
                max_results.append(return_max_value('side_effects', sideeffects_vals_labels))

                max_res_df = pd.DataFrame(max_results)
                print(pd.DataFrame(max_results))

                if result_list[-2] == 'acquired immune deficiency syndrome (AIDS)':
                    result_list[-2] = 'HIV infection'
                url = 'https://www.drugs.com/search.php?searchterm={}'.format(result_list[-2])
                with requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}) as page:
                    soup = BeautifulSoup(page.content, features='lxml')
                    condition_url = soup.find('div', {"class": "snippet search-result"}).a.get('href')
                print(condition_url)

            # Return pie charts and text based on medication selected
            elif 'Select a condition' and not 'Select a drug' in result_list:
                drug_selected = True
                max_results = []
                condition_url = None

                costlabels, costvalues = get_cost(plm, result_list[-1], 'drug')
                cost_vals_labels = list(zip(costvalues, costlabels))
                costcolors = color_cost_chart(cost_vals_labels)
                max_results.append(return_max_value('cost', cost_vals_labels))

                burdenlabels, burdenvalues = get_burden(plm, result_list[-1], 'drug')
                burden_vals_labels = list(zip(burdenvalues, burdenlabels))
                burdencolors = color_burden_chart(burden_vals_labels)
                max_results.append(return_max_value('burden', burden_vals_labels))

                sideeffectslabels, sideeffectsvalues = get_sideeffects(plm, result_list[-1], 'drug')
                sideeffects_vals_labels = list(zip(sideeffectsvalues, sideeffectslabels))
                sideeffectscolors = color_side_effects_chart(sideeffects_vals_labels)
                max_results.append(return_max_value('side_effects', sideeffects_vals_labels))

                max_res_df = pd.DataFrame(max_results)
                print(pd.DataFrame(max_results))

                # drug_id = drugbank_df[drugbank_df['name']==result_list[-1]]['drugbank_id'].to_list()
                # print(drug_id)

                url = 'https://www.drugs.com/search.php?searchterm={}'.format(result_list[-1])
                with requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}) as page:
                    soup = BeautifulSoup(page.content, features='lxml')
                    try:
                        drug_url = soup.find('div',
                                             {"class": "snippet search-result search-result-with-secondary"}).a.get(
                            'href')
                    except AttributeError as a:
                        drug_url = soup.find('div', {"class": "snippet search-result"}).a.get('href')
                    except:
                        drug_url = None
                    with requests.get(drug_url) as link:
                        soup = BeautifulSoup(link.content)
                        try:
                            d = soup.find('h2', text=re.compile("How should I [a-zA-Z]+.")).get('id')
                            s = soup.find('h2', text=re.compile(".+[a-zA-Z] side effects")).get('id')
                        except AttributeError as a:
                            d = soup.find('h2', text=re.compile("How should I [a-zA-Z]+.")).parent.get('id')
                            s = soup.find('h2', text=re.compile(".+[a-zA-Z] side effects")).parent.get('id')

                        nav_d = ["HowTake" if d != "directions" else "directions"][0]
                        nav_s = ["SideEffects" if d != "sideEffects" else "sideEffects"][0]

                print(drug_url)
        else:
            drug_selected, drug_url, condition_url, max_res_df, skip_charts = None, None, None, None, True
            nav_d, nav_s = None, None
            costlabels, costvalues, costcolors = [0], [0], [0]
            burdenlabels, burdenvalues, burdencolors = [0], [0], [0]
            sideeffectslabels, sideeffectsvalues, sideeffectscolors = [0], [0], [0]

        # Return the results page!
        return render_template("result.html", max=17000,
                               predictions=predictions,  # ML prediction result
                               # For pie charts:
                               show_charts=show_charts,
                               costset=costcolors,
                               burdenset=burdencolors,
                               sideeffectsset=sideeffectscolors,
                               # For descriptive text below pie charts
                               max_res_df_=max_res_df,
                               # For inserting python variables into the html files:
                               cond_list=cond_list,
                               drug_list=drug_list,
                               drug_url_=drug_url, condition_url_=condition_url, nav_d_=nav_d, nav_s_=nav_s,
                               drug_selected_=drug_selected, skip_charts=skip_charts,
                               result_list=result_list
                               )


if __name__ == '__main__':
    # app.run(debug=True)
    app.run(debug=False, host='0.0.0.0', port=80)
