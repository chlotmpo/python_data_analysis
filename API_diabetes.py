# ---------------------------------------  Import Libraries  --------------------------------------
import pandas as pd

import streamlit as st
import hydralit_components as hc
import streamlit.components.v1 as components

from PIL import Image

# -------------------------------------------  Structure  ------------------------------------------
def structure():
    
    # Title of the page
    st.set_page_config(page_title = "Diabetes Analysis Project",layout = "wide")

    # Hide settings menu, header and footer
    st.markdown(""" <style>
                #MainMenu {visibility: hidden;}
                header {visibility: hidden;}
                footer {visibility: hidden;}
                </style> """, unsafe_allow_html=True)

    # Style buttons
    m = st.markdown("""
                    <style>
                    div.stButton > button:first-child {
                        background-color: #0099ff;
                        color:#ffffff;
                    }
                    div.stButton > button:hover {
                        background-color: #ffffff;
                        color:#0099ff;
                        }
                    </style>""", unsafe_allow_html=True)
# --------------------------------------------------------------------------------------------------

# ----------------------------------------  Navigation Bar  ----------------------------------------
def navigationBar():
    
    # items NavBar
    menu_data = [
        {'label':'Project', 'id' : 'project'},
        {'label':"Dataset", 'id' : 'dataset'},
        {'label':"Notebook", 'id' : 'notebook'},
        {'label':"Machine Learning", 'id' : 'ml'},
        {'label':"Make Your Own Predictions", 'id' : 'pred'}]
    over_theme = {'txc_inactive': '#FFFFFF'} # ,'menu_background':'red','txc_active':'yellow','option_active':'blue'}

    # structure NavBar
    global menu_id
    menu_id = hc.nav_bar(
        menu_definition = menu_data,
        override_theme=over_theme,
        #home_name='Home',
        #login_name='Logout',
        hide_streamlit_markers=False, 
        sticky_nav=True, 
        sticky_mode='sticky')
    # --------------------------------------------------------------------------------------------------





def app():
    
    structure()
    navigationBar()
    
    # ---------------------------------------------  Content  --------------------------------------------   
    dataset, diabetes_df_ml = load_datasets()
    dataset = dataset[:101]
    diabetes_df_ml = diabetes_df_ml[:10000]
    notebook_html = load_notebook()
    
    if menu_id == 'project': 
        st.title('Analysis of dataset')
        
        st.subheader('PowerPoint:')
        st.markdown('- A presentation explaining the ins and outs of the problem, your thoughts on the asked question,\
                    \n the different variables you created, how the problem fits in the context of the study')
        
        st.subheader('Python:')
        st.markdown('- Data-visualization (use matplotlib, seaborn, bokeh ...): show the link between the variables and the target \
                    \n - Modeling: use the scikit-learn library to try several algorithms, change the hyper parameters, do a grid search, \
                    \n compare the results of your models using graphics')
        
        st.subheader('API:')
        st.markdown('- Transformation of the model into an API of your choice')
        
        st.subheader('Dataset')
        st.markdown('- Diabetes dataset')
        st.markdown('[https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008](https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008)')
        
        st.subheader('Team:')
        st.markdown(' Chloé TEMPO & Matthieu THIBAUT')
        
        st.title('')
        col1, col2 = st.columns([1,2])
        with col1:
            st.header('Link of the repository:')
        with col2:
            st.header('[GitHub](https://github.com/chlotmpo/python_data_analysis)')
        
    if menu_id == 'dataset':
        st.title('Diabetes Dataset:')
        st.markdown('\t This dataset represents 10 years (1999-2008) of clinical care at 130 US hospitals and integrated delivery networks. It includes over 50 features representing patient and hospital outcomes. \nInformation was extracted from the database for encounters that satisfied the following criteria.\
                        \n - (1) It is an inpatient encounter (a hospital admission).\
                        \n - (2) It is a diabetic encounter, that is, one during which any kind of diabetes was entered to the system as a diagnosis.\
                        \n - (3) The length of stay was at least 1 day and at most 14 days.\
                        \n - (4) Laboratory tests were performed during the encounter.\
                        \n - (5) Medications were administered during the encounter.')
        
        col1, col2, col3 = st.columns([6,1,1])
        with col1:
            st.subheader('Dataset sample:')
        with col3:
            st.download_button(label = 'Download', data = dataset.to_csv(), file_name='diabetes_sample.csv',mime='text/csv')
        st.dataframe(dataset,1200, 500)
        
        st.title('')
        st.subheader('Source:')
        st.markdown('\t The data are submitted on behalf of the Center for Clinical and Translational Research, Virginia Commonwealth University, \
                        \n a recipient of NIH CTSA grant UL1 TR00058 and a recipient of the CERNER data. \
                        \n - John Clore (jclore vcu.edu), Krzysztof J. Cios (kcios@vcu.edu), \
                        \n - Jon DeShazo (jpdeshazo@vcu.edu) \
                        \n - Beata Strack (strackb@vcu.edu).')
        st.markdown('This data is a de-identified abstract of the Health Facts database (Cerner Corporation, Kansas City, MO).')
        
        st.title('')
        st.subheader('Citation:')
        st.markdown('Beata Strack, Jonathan P. DeShazo, Chris Gennings, Juan L. Olmo, Sebastian Ventura, Krzysztof J. Cios, and John N. Clore, “Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records,” BioMed Research International, vol. 2014, Article ID 781670, 11 pages, 2014.')        
        st.markdown('[Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records](https://www.hindawi.com/journals/bmri/2014/781670/)')
   
    if menu_id == 'notebook':
        st.title('Jupyter Notebook')
        col1, col2, col3 = st.columns([6,1,1])
        with col1:
            st.write('You can find here our work:')
        with col3:
            st.download_button(label = 'Download', data = notebook_html, file_name='notebook.html', mime = 'html')
         
        components.html(notebook_html, height = 60000)

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
        
        # load the summary dataset (results of our ML models)
        summaryML = pd.read_csv("Dataset/summary_ML.csv", sep =';', header=[1])
        case1 = summaryML[['Model', 'Score', 'Accuracy']]
        case2 = summaryML[['Model.1', 'Score.1', 'Accuracy.1']]
        case2.columns = case1.columns
        
        #divide page in two columns to compare the two different approaches
        col1, col2 = st.columns([1,1]) 
        
        with col1:
            st.header("Case 1 :")
            st.header("Predict patient's readmission under 30 days")
            st.write(case1.style.hide_index().to_html(), unsafe_allow_html=True)         
            
        with col2:
            st.header("Case 2 :")
            st.header("Predict patient's readmission under and above 30 days")
            st.write(case2.style.hide_index().to_html(), unsafe_allow_html=True)  
            
        st.title('')
        st.markdown('We can see that the differents models performed better in case 1 but as mentionned earlier it is not surprising \
                    \n because the two categories to predict were unbalanced. \
                    \n In case 2, we lost some accuracy but it is a much more realistic modelization')  
    
    if menu_id == 'pred':
        diabetes_df_ml.drop(columns = ['patient_nbr'])
        
        st.title('Make Your Own Predictions')
        st.write("Now you can make your own predictions using whatever you want \
                \n Try to have the best accuracy !")
        
        # SELECT READMITTED
        col1, col2 = st.columns([2,1])
        with col1:
            st.subheader('Select the readmitted status that you want to predict: ')
        with col2:
            readmitted = st.radio("", ('Readmission under 30 days', 'Readmission under or above 30 days'))
            
        # SELECT FEATURES
        col1, col2 = st.columns([2,1])
        with col1:
            st.subheader('Select the features you want to use to predict readmission:')
        with col2:
            features = st.multiselect('', diabetes_df_ml.drop(columns = ['patient_nbr','readmitted']).columns.values.tolist()[1:])
        
        
        
        diabetes_df_ml = diabetes_df_ml[features + ['readmitted']]
        
            
        # SELECT ML MODEL
        col1, col2 = st.columns([2,1])
        with col1:
            st.subheader('Select the Machine Learning model:')
        with col2:
            model = st.selectbox('', ('K-Nearest Neighbors', 'Logistic Regression', 'Linear SVC', 'Random Forest', 'Adaptive boosting', 
                                       'Adaptive boosting', 'Decision Tree', 'Extra Trees', 'Naive Bayes'))
        
        # SELECT SPLIT RATIO
        col1, col2 = st.columns([2,1])
        with col1:
            st.subheader('Select the split ratio:')
        with col2:
            split = st.slider('', 0.0, 100.0, (65.0))
            split = split/100
        
        st.title('')
        
        # Readmitted Processing
        if readmitted == 'Readmission under or above 30 days':
            map_readmitted = {"NO" : False, ">30" : True, "<30" : True}
        else:
            map_readmitted = {"NO" : False, ">30" : True, "<30" : False}
        diabetes_df_ml.readmitted = diabetes_df_ml.readmitted.map(map_readmitted)
        
        # Features Processing
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        #diabetes_df_ml["race"] = label_encoder.fit_transform(diabetes_df_ml["race"].astype(str))
        for column in diabetes_df_ml.select_dtypes(include=['object']).columns:
            diabetes_df_ml[column] = label_encoder.fit_transform(diabetes_df_ml[column])
        for column in diabetes_df_ml.select_dtypes(include=['bool']).columns:
            diabetes_df_ml[column] = label_encoder.fit_transform(diabetes_df_ml[column])
    
        if len(features) > 0:
            col1, col2, col3 = st.columns([1, 2, 2])
            #space, col3 = st.columns([4, 1])
            
                
            with col3:
                if st.button('Predict Readmission'):
                    
                    # SHUFFLING ROWS
                    diabetes_df_ml = diabetes_df_ml.sample(n = len(diabetes_df_ml), random_state = 42)
                    
                    # SPLITTING DATA WITH RATIO ENTERED BY USER
                    from sklearn.model_selection import train_test_split
                    x = diabetes_df_ml.loc[:, diabetes_df_ml.columns != 'readmitted'] # every feature except the one that we will try to predict
                    y = diabetes_df_ml.loc[:, 'readmitted'] # readmitted feature
                    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 1 - split)
                    
                    # SCALING DATA
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    scaler.fit(x_train)
                    x_train = scaler.transform(x_train)
                    x_test = scaler.transform(x_test)
                    
                    # COMPUTING PREDICTION DEPENDING ON SELECTED MODEL
                    if model == 'K-Nearest Neighbors':
                        from sklearn.neighbors import KNeighborsClassifier
                        knn = KNeighborsClassifier(n_neighbors = 5)
                        knn.fit(x_train, y_train)
                        y_pred = knn.predict(x_test)
                        
                    if model == 'Logistic Regression':
                        from sklearn.linear_model import LogisticRegression
                        logreg = LogisticRegression()
                        logreg.fit(x_train,y_train)
                        y_pred = logreg.predict(x_test)
                        
                    if model == 'Linear SVC':
                        from sklearn.svm import LinearSVC
                        lsvc  = LinearSVC( dual = False)
                        lsvc.fit(x_train, y_train)
                        y_pred = lsvc.predict(x_test)
                        
                    if model == 'Random Forest':
                        from sklearn.ensemble import RandomForestClassifier
                        clf = RandomForestClassifier(n_estimators = 100, 
                                                    bootstrap = True, 
                                                    max_features = 'log2', 
                                                    min_samples_split = 8)
                        clf = clf.fit(x_train, y_train)
                        y_pred = clf.predict(x_test)
                        
                    if model == 'Adaptive boosting':
                        from sklearn.ensemble import AdaBoostClassifier
                        ab = AdaBoostClassifier(algorithm = 'SAMME',
                                                learning_rate = 0.95,
                                                n_estimators = 100)
                        ab.fit(x_train, y_train)
                        y_pred = ab.predict(x_test)
                    
                    if model == 'Decision Tree':
                        from sklearn.tree import DecisionTreeClassifier
                        clf = DecisionTreeClassifier(max_depth=None,
                                                     min_samples_split=2,
                                                     random_state=41)
                        clf.fit( x_train, y_train)
                        y_pred = clf.predict(x_test)
                        
                    if model == 'Extra Trees':
                        from sklearn.ensemble import ExtraTreesClassifier
                        clf = ExtraTreesClassifier(n_estimators=10,
                                                   max_depth=None,
                                                   min_samples_split=2, 
                                                   random_state=41)
                        clf.fit(x_train, y_train)
                        y_pred = clf.predict(x_test)
                        
                    if model == 'Naive Bayes':
                        from sklearn.naive_bayes import GaussianNB
                        gnb = GaussianNB(var_smoothing=1.0)
                        gnb.fit(x_train, y_train)
                        y_pred = gnb.predict(x_test)
                    
                    from sklearn.metrics import accuracy_score
                    ac = accuracy_score(y_test, y_pred)
                    st.title('')
                    st.metric("Accuracy", str(round(ac*100,2)) + " %")
            
            
            with col2:
                st.title('')
                st.title('')
                st.subheader('Trying to predict ' + str(readmitted) + ' over ' + str(len(features)) + ' features with ' + str(model) + ':')
             
                
            
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
# ----------------------------------------  Fonctions  ----------------------------------------

# Chargement du dataset en cache
@st.cache # We store the dataset in cache so it can be displayed faster
def load_datasets():
    path = "Dataset/"
    dataset = pd.read_csv(path + 'diabetic_data.csv', sep =',', na_values="?", low_memory = False)
    cleaned_dataset = pd.read_csv(path + 'diabetes_df.csv', sep =';')
    
    return dataset, cleaned_dataset

@st.cache
def load_notebook():
    HtmlFile = open("Notebook_diabetes/Notebook_diabetes.html", 'r', encoding='utf-8')
    return HtmlFile.read()

# -------------------------------------------  Main  -------------------------------------------
if __name__ == '__main__' :
    app()

