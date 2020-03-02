import pandas as pd

X = pd.read_csv(r'C:\Users\Kelly\Documents\Python\AdhereID_app\files\X2.csv', index_col=0)
plm = pd.read_csv(r'C:\Users\Kelly\Documents\Python\AdhereID_app\files\plm.csv')
conditions_list_file = r'C:\Users\Kelly\Documents\Python\AdhereID_app\files\cond_list.txt'
drugs_list_file = r'C:\Users\Kelly\Documents\Python\AdhereID_app\files\drug_list.txt'


res_url = "http://127.0.0.1:5000/result"

#
# import pandas as pd
#
# X = pd.read_csv('/home/ubuntu/AdhereID_app/files/X2.csv', index_col=0)
# plm = pd.read_csv('/home/ubuntu/AdhereID_app/files/plm.csv')
# conditions_list_file = '/home/ubuntu/AdhereID_app/files/cond_list.txt'
# drugs_list_file = '/home/ubuntu/AdhereID_app/files/drug_list.txt'
#
# res_url = "http://adhere-id.com/result"
