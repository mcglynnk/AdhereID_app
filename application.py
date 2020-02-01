


from flask import Flask, render_template, request

application = Flask(__name__)

'''
# prediction function 
def ValuePredictor(to_predict_list): 
    to_predict = np.array(to_predict_list).reshape(1, 12) 
    loaded_model = pickle.load(open("model.pkl", "rb")) 
    result = loaded_model.predict(to_predict) 
    return result[0] 
'''
# def chart(numbers):

'''
@app.route('/result', methods = ['POST']) 
def result(): 
    if request.method == 'POST': 
        to_predict_list = request.form.to_dict() 
        to_predict_list = list(to_predict_list.values()) 
        to_predict_list = list(map(int, to_predict_list)) 
        result = ValuePredictor(to_predict_list)         
        if int(result)== 1: 
            prediction ='Income more than 50K'
        else: 
            prediction ='Income less that 50K'            
        return render_template("result.html", prediction = prediction) 

'''

from flask import Flask, Markup, render_template

application = Flask(__name__)


@application.route('/', methods=['POST', 'GET'])
def home1():
    return render_template("index.html")


# ML Model
import pickle

filename = r'nmodel.sav'
with open(filename, 'rb') as file:
    dt_model = pickle.load(file)

from collections import OrderedDict
import pandas as pd
import numpy as np


# X = pd.read_csv(r'C:\Users\Kelly\Documents\Python\streamlit_app\X.csv')

# features = X.loc[range(50)]
# features_display = X.loc[features.index]

labels = [
    'opt1', 'opt2', 'opt3'
]

colors = [
    "#F7464A", "#46BFBD", "#FDB45C"]



@application.route('/result', methods=['POST', 'GET'])
def result1():
    if request.method == 'POST':

        result_list = request.form.to_dict()
        result_list = list(result_list.values())
        print('result list', result_list)

        inputs = OrderedDict()

        # Age
        inputs['age'] = result_list[0]

        # Sex
        sex = result_list[3]
        if sex == 'M':
            sex = 1.0
        elif sex == 'F':
            sex = 2.0
        else:
            None
        inputs['sex'] = sex

        # Num of rx
        inputs['n_rx'] = result_list[2]

        # N provider visits
        inputs['n_provider_visits'] = result_list[1]

        # Gen health
        inputs['general_health'] = result_list[4]
        gen_val = {
            'Excellent': 1,
            'Very good':2,
            'Good':3,
            'Fair':4,
            'Poor':5,
        }
        # inputs['genera_health'].replace(gen_val, inplace=True)

        # Income
        inputs['income'] = result_list[5]


        inputs_df = pd.DataFrame([inputs], columns=inputs.keys())
        inputs_array = np.array(inputs_df)

        # predictions = dt_model.predict(inputs_array)


        # inputs_list = [sex, result_list[0], result_list[2]]
        # X.loc[len(X.index) + 1] = inputs_list
        # import warnings
        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore")
        #     shap_values = explainer.shap_values(features)[1]
        #     shap_interaction_values = explainer.shap_interaction_values(features)
        # if isinstance(shap_interaction_values, list):
        #     shap_interaction_values = shap_interaction_values[1]
        # fig = Figure()

        # t = shap.decision_plot(expected_value, shap_values, features_display)
        # plt.savefig(r'C:\Users\Kelly\Documents\Python\flask_app\static\images\plot.png')

        # Convert plot to PNG image
        # pngImage = io.BytesIO()
        # FigureCanvas(fig).print_png(pngImage)
        #
        # # Encode PNG image to base64 string
        # pngImageB64String = "data:image/png;base64,"
        # pngImageB64String += base64.b64encode(pngImage.getvalue()).decode('utf8')
        #
        # shap_val_pred = shap_values[len(shap_values) - 1]
        #
        # # shap_values = shap.TreeExplainer(dt_model,feature_perturbation='interventional').shap_values(predictions)
        # # print(shap_values)

        # if predictions == 1.0:
        #     predictions = "Low risk of non-adherence"
        # elif predictions == 0:
        #     predictions = "High risk of non-adherence"
        # else:
        #     None

        return render_template("result.html", max=17000,
                               predictions=predictions,

                               )



#
# # bokeh serve bokeh_obj.py --allow-websocket-origin=localhost:5006 --allow-websocket-origin=localhost:5000 --allow-websocket-origin=127.0.0.1:5000
# # flask run
#


# @application.route('/x', methods=['POST', 'GET'])
# def result2():
#     with pull_session(url="http://localhost:5006/bokeh_obj") as session:
#         if request.method == 'POST':
#
#             result_list = request.form.to_dict()
#             result_list = list(result_list.values())
#             print('result list', result_list)
#
#             inputs = OrderedDict()
#
#             sex = result_list[1]
#             if sex == 'M':
#                 sex = 1.0
#             elif sex == 'F':
#                 sex = 2.0
#             else:
#                 None
#             inputs['sex'] = sex
#
#             inputs['age'] = result_list[0]
#
#
#
#             inputs['n_rx'] = result_list[2]
#
#             #print(inputs)
#             inputs_df = pd.DataFrame([inputs], columns=inputs.keys())
#             inputs_array = np.array(inputs_df)
#             print(inputs_array)
#             predictions = dt_model.predict(inputs_array)
#
#             if predictions == 1.0:
#                 predictions = "Low risk of non-adherence"
#             elif predictions == 0:
#                 predictions = "High risk of non-adherence"
#             else:
#                 None
#
#             return render_template("result.html", max=17000,
#                                    predictions=predictions)
#
#         # generate a script to load the customized session
#         script = server_session(None, session.id, url='http://localhost:5006/bokeh_obj')
#
#         return render_template("result2.html", script=script, template="Flask")


if __name__ == '__main__':
    application.run(debug=True)
