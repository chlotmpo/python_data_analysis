import streamlit as st
import pandas as pd
import numpy as np

import base64
import csv
import time

from PIL import Image

from st_btn_select import st_btn_select
import hydralit_components as hc

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
        {'label' : 'Project', 'id' : 'project'},
        {'label':"Dataset", 'id' : 'dataset'},
        {'label':"Data Visualization", 'id' : 'dataviz'},
        {'label':"Machine Learning", 'id' : 'ml'}]
        #{'label':"GitHub", 'id' : 'github'}]

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
    if menu_id == 'project':
        col1, col2, col3 = st.columns([2,6,2])


        with col1:
            st.write(str([x for x in range(50)]))

        with col2:
            st.title('\tAnalysis of diabetes dataset')

        with col3:
            st.write(str([x for x in range(50)]))
            
        
        img = Image.open(r"Resources\diabete.jpg")
        st.image(img)
        st.markdown('\tHospital readmission is a real-world problem and an on-going topic for improving \
                    health care quality and a patient’s experience, while ensuring cost-effectiveness. \
                    Information of Hospital Readmissions Reduction Program (HRRP) is publicly\
                    available in CMS, Center for Medicare and Medicaid Services, web site.\
                    The dataset, Diabetes 130-US hospitals for years 1999-2008 Data Set, \
                    was downloaded from UCI Machine Learning Repository. It represents 10 years \
                    (1999-2008) of clinical care at 130 US hospitals and integrated delivery networks \
                    with 100,000 observations and 50 features representing patient and hospital outcomes.')
        st.title('[GitHub](https://github.com/chlotmpo/python_data_analysis)')
        
    if menu_id == 'dataset':
        st.title('Diabetes Dataset:')
        st.dataframe(load_dataset())
        st.markdown(get_table_download_link(load_dataset()), unsafe_allow_html=True)


    if menu_id == 'dataviz':
        st.title('Data Visualization')
        
        img = Image.open(r"Resources\AgeDistribution.png")
        st.image(img)
        
        st.code('df = pd.read_csv(path, sep = ","')
        st.code('Texte')

    if menu_id == 'ml':
        st.title('Machine Learning')
        st.checkbox()
        
    #if menu_id == 'github':
        #st.title('[GitHub](https://github.com/chlotmpo/python_data_analysis)')
    # --------------------------------------------------------------------------------------------------

    
    
    
    
    
# ----------------------------------------  Fonctions  ----------------------------------------

# Chargement du dataset en cache
@st.cache # We store the dataset in cache so it can be displayed faster
def load_dataset():
    #path = r'C:\Users\mt181547\OneDrive - De Vinci\Bureau\diabetic_data.csv'
    path = r'Resources\diabetic_data.csv'
    dataset = pd.read_csv(path, sep=',')
    dataset2 = dataset[dataset["weight"] != '?']
    return dataset2

def get_table_download_link(df):
    csv = df.to_csv(index=False)
    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}">Download csv file</a>'
    return href

# ---------------------------------------------------------------------------------------------


# Main
if __name__ == '__main__' :
    app()

