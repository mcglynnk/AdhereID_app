import pandas as pd

X = pd.read_csv(r'C:\Users\Kelly\Documents\Python\AdhereID_app\files\X.csv', index_col=0)
plm = pd.read_csv(r'C:\Users\Kelly\Documents\Python\AdhereID_app\files\plm.csv')
conditions_list_file = r'C:\Users\Kelly\Documents\Python\AdhereID_app_charts\cond_list.txt'

res_url = "http://127.0.0.1:5000/result"
