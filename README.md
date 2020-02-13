# AdhereID    
    
This repository holds the code for AdhereID, an app that uses machine learning to help doctors predict patient medication adherence.  Visit this link (http://adhere-id.com/) to view the app!  See this repository (https://github.com/mcglynnk/AdhereID) for the data, processing, and model behind the app.
    
## How to Use
The home page of the app takes inputs for a number of patient characteristics including demographics (age, income, etc.) and basic medical information (number of prescriptions currenly taken, number of doctor visits per year, etc.).  On the results page, high or low risk for medication adherence is reported, along with information on drug cost, burden and side effects for the chosen medical condition (from 'Select a medical condition').
        
## Project Structure
\- files/&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \# Logistic Regression model file    
\- static/&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \# css, js, scss (downloaded bootstrap template sb-admin-2)    
\- templates/&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \# HTML files    
&nbsp;&nbsp;\- index.html &nbsp;&nbsp;&nbsp;&nbsp; \# AdhereID home page     
&nbsp;&nbsp;\- result.html &nbsp;&nbsp;&nbsp;&nbsp; \# AdhereID results page    
\- app.py &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \# \*Flask app 
\- functions.py &nbsp;&nbsp;&nbsp;&nbsp; \# Data processing functions that are used in app.py    

## Deployment    
Built with:    
- [Flask](https://flask.palletsprojects.com/en/1.1.x/)    
- [AWS EC2 instance](https://aws.amazon.com/ec2/)    
- [ChartJS](https://www.chartjs.org/docs/latest/charts/doughnut.html)    
    
## License    
This project is licensed under the MIT License - see the LICENSE.md file for details.
    
Kelly McGlynn, 2020
