import streamlit as st
import pandas as pd
import numpy as np

from streamlit_pandas_profiling import st_profile_report
from pandas_profiling import ProfileReport

import base64
import csv
import time

from PIL import Image

from st_btn_select import st_btn_select
import hydralit_components as hc

import streamlit.components.v1 as components

#https://docs.streamlit.io/library/api-reference/widgets

def structure():
    # -------------------------------------------  Structure  ------------------------------------------
    # Titre page
    st.set_page_config(page_title = "DIABETES",layout = "wide")

    # Masquer menu de reglages
    st.markdown(""" <style>
                #MainMenu {visibility: hidden;}
                header {visibility: hidden;}
                footer {visibility: hidden;}
                </style> """, unsafe_allow_html=True)

    # Style boutons
    m = st.markdown("""
                    <style>
                    div.stButton > button:first-child {
                        background-color: #0099ff;
                        color:#ffffff;
                    }
                    div.stButton > button:hover {
                        background-color: #00ff00;
                        color:#ff0000;
                        }
                    </style>""", unsafe_allow_html=True)
    # --------------------------------------------------------------------------------------------------






def navigationBar():
    # ----------------------------------------  Navigation Bar  ----------------------------------------

    # Navigation Bar V1
    #st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)

    # Navigation Bar V2
    #page = st_btn_select(('Home', 'Dataset', 'DataViz', 'Machine Learning', 'PLUS'), nav = True, index = 0)

    # Navigation Bar V3
    # items NavBar
    menu_data = [
        {'label':'Project', 'id' : 'project'},
        {'label':"Dataset", 'id' : 'dataset'},
        {'label':"Notebook", 'id' : 'notebook'},
        {'label':"Machine Learning", 'id' : 'ml'}]

    over_theme = {'txc_inactive': '#FFFFFF'} # ,'menu_background':'red','txc_active':'yellow','option_active':'blue'}

    # structure NavBar
    global menu_id
    menu_id = hc.nav_bar(menu_definition = menu_data,
        override_theme=over_theme,
        #home_name='Home',
        #login_name='Logout',
        hide_streamlit_markers=False, #will show the st hamburger as well as the navbar now!
        sticky_nav=True, #at the top or not
        sticky_mode='sticky', #jumpy or not-jumpy, but sticky or pinned
    )
    # --------------------------------------------------------------------------------------------------





def app():
    structure()
    navigationBar()
    # ---------------------------------------------  Content  --------------------------------------------   
    
    dataset = load_dataset()
    dataset = dataset[:50]
    #report = load_report(dataset)
    
        
    if menu_id == 'project':
        
        col1, col2, col3 = st.columns([1,6,1])
        
        with col2:
            st.title('Analysis of diabetes dataset')
            st.markdown('\t Hospital readmission is a real-world problem and an on-going topic for improving \
                    health care quality and a patient’s experience, while ensuring cost-effectiveness. \
                    Information of Hospital Readmissions Reduction Program (HRRP) is publicly\
                    available in CMS, Center for Medicare and Medicaid Services, web site.\
                    The dataset, Diabetes 130-US hospitals for years 1999-2008 Data Set, \
                    was downloaded from UCI Machine Learning Repository. It represents 10 years \
                    (1999-2008) of clinical care at 130 US hospitals and integrated delivery networks \
                    with 100,000 observations and 50 features representing patient and hospital outcomes.')
        
        st.title('[GitHub](https://github.com/chlotmpo/python_data_analysis)')
            
            
        
        
        
        
        img = Image.open(r"Resources\diabete.jpg")
        st.image(img)
        
    if menu_id == 'dataset':
        st.title('Diabetes Dataset:')
        st.markdown('\t This dataset represents 10 years (1999-2008) of clinical care at 130 US hospitals and integrated delivery networks. It includes over 50 features representing patient and hospital outcomes. \nInformation was extracted from the database for encounters that satisfied the following criteria.\
                        \n - (1) It is an inpatient encounter (a hospital admission).\
                        \n - (2) It is a diabetic encounter, that is, one during which any kind of diabetes was entered to the system as a diagnosis.\
                        \n - (3) The length of stay was at least 1 day and at most 14 days.\
                        \n - (4) Laboratory tests were performed during the encounter.\
                        \n - (5) Medications were administered during the encounter.')
        st.title('')
        st.dataframe(dataset,1200, 500)
        st.download_button(label = 'Download', data = dataset.to_csv(), file_name='diabetes_sample.csv',mime='text/csv')


    if menu_id == 'notebook':
        st.title('Jupyter Notebook')
        
        HtmlFile = open(r"..\\Notebook_diabetes\\Notebook_diabetes.html", 'r', encoding='utf-8')
        source_code = HtmlFile.read()
        
        col1, col2, col3 = st.columns([6,1,1])
        with col1:
            st.write('You can find here our Jupyter Notebook: ')
        with col3:
            st.download_button(label = 'Download', data = source_code, file_name='notebook.html', mime = 'html')
         
        components.html(source_code, height = 60000)
        
            

    if menu_id == 'ml':
        
        st.title('Machine Learning')
            
        st.write("We use the diabetes dataset to train some Machine Learning algorithms in order\
                    to predict the readmission of a patient\n \
                    Before any modifications, the readmitted features was composed of 3 different values: \
                    \n - No (No readmission)\
                    \n - < 30 (Readmitted under 30 days)\
                    \n - \> 30 (readmitted under or above 30 days)")
        st.write("Based on that, we had to transform this into a binary decision.\
                      Hence some values had to be regrouped.\
                    \n We chose first to regroupe (No) and (>30). This means that the decision is reduced to :\
                    \n > Is the patient going to be readmitted under 30 days ? (Yes or No)")
        st.write("Next, we regrouped (<30) and (>30). This time, the decision is reduced to :\
                      \n > Is the patient going to be readmitted ? (Yes or No)")
        st.title('')
        
        
        #divide page in two columns to compare the two different approaches
        col1, col2 = st.columns([1,1]) 
        with col1:
            st.header("Case 1 :")
            st.header("Predict patient's readmission under 30 days")
            
            
        with col2:
            st.header("Case 2 :")
            st.header("Predict patient's readmission under and above 30 days")


    
        
        
    #components.iframe('https://github.com/chlotmpo/python_data_analysis')
    # --------------------------------------------------------------------------------------------------

    
    
    
    
    
# ----------------------------------------  Fonctions  ----------------------------------------

# Chargement du dataset en cache
@st.cache # We store the dataset in cache so it can be displayed faster
def load_dataset():
    #path = r'C:\Users\mt181547\OneDrive - De Vinci\Bureau\diabetic_data.csv'
    path = r'Resources\diabetic_data.csv'
    dataset = pd.read_csv(path, sep =',', na_values="?", low_memory = False)
    return dataset


#def load_report(dataset):
#    report = ProfileReport(dataset, title = "Diabetes dataset overview", dark_mode = True)
#    return report


#def get_table_download_link(df):
#    csv = df.to_csv(index=False)
#    # some strings <-> bytes conversions necessary here
#    b64 = base64.b64encode(csv.encode()).decode()
#    href = f'<a href="data:file/csv;base64,{b64}">Download csv file</a>'
#    return href

# ---------------------------------------------------------------------------------------------


# Main
if __name__ == '__main__' :
    app()

