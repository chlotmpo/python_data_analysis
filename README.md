# Python for data analysis: Diabetes dataset

## Instructions
As part of the python for data analysis course, the aim of end of the semester's projet is to be able to :
- transform a dataset into usable informations (Data processing)
- show the links between features (Data vizualisation)
- build a machine learning model to predict some features (Modeling)

## Table Of Contents
- [API_diabetes(streamlit)](./API_diabetes(streamlit)/)
This section contains all the files necessary for the operation and management of the Streamlit API. The latter makes it possible to visualize in another way the entire project. 
  - [.streamlit](./API_diabetes(streamlit)/.streamlit/)
  This is the API configuration file.
  - [API_diabetes.py](./API_diabetes(streamlit)/API_diabetes.py/)
  This file is the one that created the design of the API and also allows it to work. 
  
- [Dataset](./Dataset)
This section contains all files related to and linked to this dataset of our project.
Our analysis focuses on a dataset that containts data on **hospitalized diabetic patients**. 
- [description_diabetes](./Dataset/description_diabetes)
  This first document is a detailed description of the dataset, which explains the origin of the data, the meaning of the     differents columns and characteristics. It was the starting point for our analysis. 
   - [diabetic_data](./Dataset/diabetic_data)
  All data is stored in this file. This is the one we imported into our notebook in order to retrieve the content and process it. 
  - [id_mapping](./Dataset/id_mapping)
  id_mapping contains the data equivalents needed to map certain columns of the dataset. It was also provided with the dataset.
  - [summary_ML](./Dataset/summary_ML)
  We will find here all the performance results of Machine Learning algorithms that we have trained on our data. It contains the result of the score and accuracy of 8 different models, which we decided to implement. It allows us to compare the different performances to find which model is best suited to our data set. 
  
- [Notebook_diabetes](./Notebook_diabetes/)
This section consists of 2 files, we will find our jupyter notebook and its html version, with all our python code that allowed us to clean, visualize, analyze and train our data. 
  - [Notebook_diabetes.ipynb](./Notebook_diabetes/Notebook_diabetes.ipynb)
  This file with ipynb terminaison corresponds to the notebook created on jupyter. This is where all the code and results are stored. So you can see the blocks of code, visualization graphs and other titles and comments that explain the progress of the project throughout the notebook. 
  - [Notebook_diabetes.html](./Notebook_diabetes/Notebook_diabetes.html) 
  This is the html export of the previous file. It can open on any browser and allows you to view the code and results simply. 
-[ReadMe.md](./README.md)
You are on the ReadMe.md file. 




