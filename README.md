# Python for data analysis: Diabetes dataset

## Instructions
As part of the python for data analysis course, the aim of end of the semester's projet is to be able to :
- transform a dataset into usable informations (Data processing)
- show the links between features (Data vizualisation)
- build a machine learning model to predict some features (Modeling)

## Table Of Contents
- [API_diabetes(streamlit)](./API_diabetes(streamlit)/)
This section contains all the files necessary for the operation and management of the Streamlit API. The latter makes it possible to visualize in another way the entire project.     
The link to access it is: [https://share.streamlit.io/chlotmpo/python_data_analysis/main/API_diabetes.py](https://share.streamlit.io/chlotmpo/python_data_analysis/main/API_diabetes.py)
  - [.streamlit](./API_diabetes(streamlit)/.streamlit/)
  This is the folder containing the API configuration file.
  - [API_diabetes.py](./API_diabetes(streamlit)/API_diabetes.py/)
  This file is the one that created the design of the API and also allows it to work. 
  
- [Dataset](./Dataset)
This section contains all files related to and linked to this dataset of our project.
Our analysis focuses on a dataset that containts data on **hospitalized diabetic patients**. 
  - [description_diabetes](./Dataset/description_diabetes)
  This first document is a detailed description of the dataset, which explains the origin of the data, the meaning of the     differents columns and characteristics. It was the starting point for our analysis. 
  - [diabetic_data](./Dataset/diabetic_data)
  All data is stored in this file. This is the one we imported into our notebook in order to retrieve the content and process it.
  - [diabetes_df](./Dataset/diabetes_df)
  Cleaned version of the dataset used in the notebook (generated from the notebook)
  - [id_mapping](./Dataset/id_mapping)
  id_mapping contains the data equivalents needed to map certain columns of the dataset. It was also provided with the dataset.
  - [summary_ML](./Dataset/summary_ML)
  We will find here all the performance results of Machine Learning algorithms that we have trained on our data. It contains the result of the score and accuracy of 8 different models, which we decided to implement. It allows us to compare the different performances to find which model is best suited to our data set. 
  
- [Notebook_diabetes](./Notebook_diabetes/)
This section consists of 2 files, we will find our jupyter notebook and its html version, with all our python code that allowed us to clean, visualize, analyze and train our data. 
  - [Notebook_diabetes.ipynb](./Notebook_diabetes/Notebook_diabetes.ipynb)
  This file with ipynb terminaison corresponds to the notebook created on jupyter. This is where all the code and results are stored. So you can see the blocks of code, visualization graphs and other titles and comments that explain the progress of the project throughout the notebook.    
  This notebook can be open at [https://nbviewer.org/github/chlotmpo/python_data_analysis/blob/main/Notebook_diabetes/Notebook_diabetes.ipynb](https://nbviewer.org/github/chlotmpo/python_data_analysis/blob/main/Notebook_diabetes/Notebook_diabetes.ipynb)
  - [Notebook_diabetes.html](./Notebook_diabetes/Notebook_diabetes.html) 
  This is the html export of the previous file. It can open on any browser and allows you to view the code and results simply. 
  
- [PPT - Full-detailed version](./PPT-Full-detailed-version.pdf)
This presentation is complete and includes all the comments of our choices and steps 

- [PPT - Oral version](./PPT-Oral-version.pdf)
This presentation includes less text and is used for an oral presentation of the project

- [ReadMe.md](./README.md)
You are on the ReadMe.md file. 

## Progress of the project 

### Data loading and cleaning 
First, you must download and recover the data set in our jupyter notebook. In order to be able to treat it afterwards, we must observe it in detail to clean it as well as possible. This involves abandoning columns with too many missing values, or those that have no utilities or values that are too redundant. The same goes for the lines. You should also pay attention to the column types when retrieving the dataset.
This step is very important because it conditions the whole sequence. We can also map some columns so that it is easier to represent real understandable situations during visualization. 

### Data analysis and visualization
The next step is a first analysis of the data and the different characteristics of the dataset. We are going to do visualization to try to find links, behaviors between variables to better understand the hidden meanings behind the data. The graphs make these visual results easier to understand and communicate. 
All graphics are visible in the jupyter notebook. Adding a correlation matrix is of great interest to glimpse the variables that may or may not be correlated.

### Machine Learning - Models Implementation
The aim of this section is to use the diabetes dataset to train some Machine Learning models using the diabete dataset in order to predict the readmission of a patient. 

After preparing a machine learning oriented dataset from the original dataset, we decided to distinguish two cases according to the data:
- Case 1: Predict patient’s readmission under 30 days
- Case 2: Predict patient’s readmission under and above 30 days

In this study, we try to predict a qualitative binary variable. We also have a fairly large set of data, which leads us to use some models more than others.

We therefore tested and implemented the following models to compare their prediction performance :
- K-Nearest Neighbors (KNN) 
- Logistic Regression 
- Linear SVC 
- Random Forest 
- Adaptive Boosting 
- Decision Tree
- Extra Trees 
- Naïve Bayes Classifier

All the results and performance of each model are available in the jupyter notebook.

### Machine Learning - Models Tuning 
With the help of the libraries, we can search for the best possible parameters for the different models and therefore try to improve the performance of each one. This was established in this phase. All the results obtained for the different models are also available in the jupyter notebook.

### API 
To make the project a little more visual and accessible, we created a streamlit project that we linked to our python code to create an API. It presents different tabs with the description of the dataset, the notebook, our results in machine learning and an interactive parts. 

### Conclusion
After observing the results, the following conclusions can be drawn. 
- For case number 1, we can see that the predictions are made with a rather high score but we must qualify this result because the problem is unbalanced. You can't really rely on those results. 
- For case number 2, the prediction performance is lower, less accurate, but this result is closer to reality. 
- We can conclude from this that this dataset is rather little correlated, that the variables seems to have relations between them but without having very large and significant ones. 
- It is therefore difficult to predict whether or not a patient will be readmitted to hospital with the features available in this dataset. We can put out an idea but it will not be very reliable on the subject. 



