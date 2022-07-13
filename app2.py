# -*- coding: utf-8 -*-
"""
    By Francis
"""
# from optparse import Values
import numpy as np
import pandas as pd
import pickle
import streamlit as st
import streamlit.components.v1 as stc
import seaborn as sns
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
from PIL import Image
from st_card_component import card_component as card


# loading the saved models

diabetes_model = pickle.load(open('trained_model.sav', 'rb'))
df = pd.read_csv("Diabetes.csv")
# sidebar for navigation
with st.sidebar:
    
    selected = option_menu('Diabetes  Prediction System',
                          
                          ['Home','Data Visualiztion','Diabetes Prediction',
                           'About'],
                          icons=['house','kanban','activity','person'],
                          default_index=0)
    
#  Home page
df = pd.read_csv("Diabetes.csv")
if (selected == "Home"):
    HTML_BANNER = """
    <div style="background-color:#464e5f;padding:10px;border-radius:10px">
    <h1 style="color:white;text-align:center;">Diabetes App </h1>
    </div>
    """
    stc.html(HTML_BANNER)
    
    # page title
    st.title("Home")
    st.dataframe(df)
    
    
       
# Diabetes Prediction Page
if (selected == 'Diabetes Prediction'):
    HTML_BANNER = """
    <div style="background-color:#464e5f;padding:10px;border-radius:10px">
    <h1 style="color:white;text-align:center;">Diabetes App </h1>
    </div>
    """
    stc.html(HTML_BANNER)
    # page title
    st.title('Diabetes Prediction using ML')
    # 
    
    # getting the input data from the user
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
        
    with col2:
        Glucose = st.text_input('Glucose Level')
    
    with col3:
        BloodPressure = st.text_input('Blood Pressure value')
    
    with col1:
        SkinThickness = st.text_input('Skin Thickness value')
    
    with col2:
        Insulin = st.text_input('Insulin Level')
    
    with col3:
        BMI = st.text_input('BMI value')
    
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    
    with col2:
        Age = st.text_input('Age of the Person')
    
    
    # code for Prediction
    diab_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        
        if (diab_prediction[0] == 1):
          diab_diagnosis = 'The person is diabetic'
        else:
          diab_diagnosis = 'The person is not diabetic'
        
    st.success(diab_diagnosis)


if (selected == "Data Visualiztion"):
    st.write("""
    ## Plots Showing Relationsip between diabetes outcome and other features
    """)
    def plot():
        page = st.sidebar.selectbox(
            "Select a Page",
            [
                "Count Plot", #New page
            ]
        )

        if page == "Line Plot":
            linePlot()
        
        elif page == "Count Plot":
            countPlot()
        elif page == "Box plot":
            BoxPlot()
        elif page == "Bar plot":
            Barplot()
    # BoxPlot, Barplot,countPlot= st.columns(4)
    st.write("""
    #### Box plot showing Relationship between diabetes outcome and Number of Pregnancies
    """)
    def countPlot():
        fig = plt.figure(figsize=(10, 4))
        sns.countplot(x = "Pregnancies",hue='Outcome', data = df)
        st.pyplot(fig)

    countPlot()
    st.write("""
    #### Box plot showing Relationship between diabetes outcome and Age
    """)
    
    def BoxPlot():
        fig = plt.figure(figsize=(10, 4))
        sns.boxplot(x='Outcome', y='Age', data=df)
        st.pyplot(fig)

    BoxPlot()
    st.write("""
    #### Bar plot showing Relationship between diabetes outcome and BMI
    """)
    def Barplot():
        fig = plt.figure(figsize=(10, 4))
        sns.barplot(x='Outcome', y='BMI', data=df)
        st.pyplot(fig)

    Barplot()
    st.write("""
    #### Bar plot showing Relationship between diabetes outcome and BloodPressure
    """)
    def Barplot():
        fig = plt.figure(figsize=(10, 4))
        sns.barplot(x='Outcome', y='BloodPressure', data=df)
        st.pyplot(fig)

    Barplot()
        
    st.write("""
    #### Bar plot showing Relationship between diabetes outcome and Glucose Level
    """)
    def Barplot():
        fig = plt.figure(figsize=(10, 4))
        sns.barplot(x='Outcome', y='Glucose', data=df)
        st.pyplot(fig)

    Barplot()


# 

# Parkinson's Prediction Page

if (selected == "About"):
    st.title("FRANCIS KIPKOGEI")
    fk_img = Image.open("profile-img.webp")
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    st.markdown("<h3 style='text-align: center; color: black; background: #D3D3D3; margin: 3px'>My Profile</h1>", unsafe_allow_html=True)
    img,  desc =st.columns(2)
    with img:
        st.image(fk_img)
    with desc:
        st.markdown("<h4 style='text-align: center; color: blue; margin: 3px'>Data Scientist|Actuarial Scientist|Data Analyst</h4>", unsafe_allow_html=True)
        st.markdown('<p style="text-align: justify;">I am a Data Scientist with a number of years of experience in data analysis, Statistical analysis, data visualization, data mining, Machine learning, deep learning, Artificial Intelligence and web app deployment. I am highly organized, motivated, and diligent with an advanced understanding of statistical, algebraic, and other analytical techniques. Moreover, I am proficient to an advanced level in using Python, R, SPSS, MS Excel, and SQL. Throughout my career, I contributed to positive business results through effective organization, prioritization, and follow-through of crucial organizational projects. </p>', unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: blue; background: #D3D3D3; margin: 3px'>My Experience</h1>", unsafe_allow_html=True)
    stepwise, Zalego,RSSB,CIDRA= st.columns(4)
   
    with stepwise:
        st.markdown('##### Senior Data Scientist')
        st.write("Company: Zalda, Stepwise Inc")
        st.write('From: 01/01/2022 - todate')
        st.markdown("<div style='text-align: center; color: black;font-weight: bold;'>Responsibilities</div>", unsafe_allow_html=True)
        st.write("Lead Data Scienst")
        st.write("Data Cleaning and Data  Wrangling")
        st.write("Data Analysis and statistical analysis")
        st.write("Machine Learning, deep learning and AI")
        st.write("Data Mining")
        st.write("Data Visualization and Reporting")
        st.write("Innovation")
    with Zalego:
        st.markdown('##### Data Scientist')
        st.write("Company: Zalego Academy, Stepwise Inc")
        st.write('From: 01/04/2021 - todate')
        st.markdown("<div style='text-align: center; color: black;font-weight: bold;'>Responsibilities</div>", unsafe_allow_html=True)
        st.write("Data Cleaning and Data  Wrangling")
        st.write("Data Analysis and statistical analysis")
        st.write("Data Science and Data Analysis Technical Instructor")
        st.write("Develop Data science/Analysis course Content")
        st.write("Machine Learning and Deep learning")
        st.write("Data Mining")
        st.write("Data Visualization and Reporting")
        
    with RSSB:
        st.markdown('##### Data Scientist   Intern')
        st.write("Company: Rwanda Social Security Board, Kigali, Rwanda")
        st.write('From: 01/01/2020 - 01/05/2020')
        st.markdown("<div style='text-align: center; color: black;font-weight: bold;'>Responsibilities</div>", unsafe_allow_html=True)
        # st.write("Data Science/Data Analysis Technical Instructor")
        st.write("Data Cleaning and Data  Wrangling")
        st.write("Data Analysis and statistical analysis")
        st.write("Machine Learning")
        st.write("Data Visualization and Reporting")    
    st.markdown("<h3 style='text-align: center; color: blue; background: #D3D3D3; margin: 3px'>My Skills</h1>", unsafe_allow_html=True)
    with CIDRA:
        st.markdown('##### Data Analyst')
        st.write("Company: CIDRA, Kigali, Rwanda")
        st.write('From: 01/01/2019 - 01/12/2019')
        st.markdown("<div style='text-align: center; color: black;font-weight: bold;'>Responsibilities</div>", unsafe_allow_html=True)
        st.write("Data Cleaning and Data  Wrangling")
        st.write("Data Analysis and statistical analysis")
        st.write("Data Visualization and Reporting")    
     
    s1, s2, s3= st.columns(3)
    with s1:
        st.markdown('##### Programming Languages and Packages')
        x = ['Python', 'R', 'SQL',  "MS Excel", 'SPSS','SAS']
        for i in x:
            st.markdown(
                f'<span style="background-color:#00C4EB;color: #FFFFFF;padding: 0.5em 1em;position: relative;text-decoration: none;font-weight:bold;cursor: pointer;">{i}</span>', unsafe_allow_html=True)
    with s2:
        st.markdown('##### Data Analysis')
        x = ['Data Cleaning','EDA','Data Wrangling', 'Descriptive Analysis', 'Inferential Statistics (A/B Testing', 'Times Series Analysis', 'Predictive Analysis']
        for i in x:
            st.markdown(
                f'<span style="background-color:#00C4EB;color: #FFFFFF;padding: 0.6em 1em;position: relative;text-decoration: none;font-weight:bold;cursor: pointer;">{i}</span>', unsafe_allow_html=True)
    
    with s3:
        st.markdown('##### Visualization')
        x = ['Power BI', 'Plotly', 'Seaborn', 'GGplot 2','Pandas']
        for i in x:
            st.markdown(
                f'<span style="background-color:#00C4EB;color: #FFFFFF;padding: 0.5em 1em;position: relative;text-decoration: none;font-weight:bold;cursor: pointer;">{i}</span>', unsafe_allow_html=True)
    s4, s5, s6= st.columns(3)
    with s4:
        st.markdown('##### ML&DL')
        x = ['Sklearn','Keras','TensorFlow', 'Pytorch', 'NumPy', 'Pandas']
        for i in x:
            st.markdown(
                f'<span style="background-color:#00C4EB;color: #FFFFFF;padding: 0.6em 1em;position: relative;text-decoration: none;font-weight:bold;cursor: pointer;">{i}</span>', unsafe_allow_html=True)
    
    with s5:
        st.markdown('##### AI')
        x = ['NLP', 'Neural Networks']
        for i in x:
            st.markdown(
                f'<span style="background-color:#00C4EB;color: #FFFFFF;padding: 0.5em 1em;position: relative;text-decoration: none;font-weight:bold;cursor: pointer;">{i}</span>', unsafe_allow_html=True)
    with s6:
        st.markdown('##### WEB APP')
        x = ['Flask', 'Streamlit', 'Heroku',"Panel"]
        for i in x:
            st.markdown(
                f'<span style="background-color:#00C4EB;color: #FFFFFF;padding: 0.5em 1em;position: relative;text-decoration: none;font-weight:bold;cursor: pointer;">{i}</span>', unsafe_allow_html=True)
            
    st.markdown("<h3 style='text-align: center; color: blue; background: #D3D3D3; margin: 3px'>Education</h1>", unsafe_allow_html=True)
    st.markdown("<h4 <span style = color:blue;'>Course:</span>Master of Science in Data Science</h4>", unsafe_allow_html=True)
    st.markdown("<h6 <span'>School:</span> ACE-DS, University of Rwanda - Rwanda</h6>", unsafe_allow_html=True)
    st.markdown("<h6 <span'></span>From: Oct-2018 to Dec-2020</h6>", unsafe_allow_html=True)
    st.markdown("<h6 <span'>Dissertation:</span> Tree-based and Logistic Regression Machine Learning Models for Business Success Prediction in Rwanda</h6>", unsafe_allow_html=True)
    st.write(" ")
    

    st.markdown("<h4 <span style = color:blue;'>Course:</span>Master of Science in Financial Engineering</h4>", unsafe_allow_html=True)
    st.markdown("<h6 <span'>School:</span>WorldQuant University, Louisiana, USA</h6>", unsafe_allow_html=True)
    st.markdown("<h6 <span'></span>From: April-2019 to Feb-2021</h6>", unsafe_allow_html=True)
    st.markdown("<h6 <span'></span>Dissertation: Comparison of Traditional Time Series techniques and Machine Learning models in Stock Market Price Prediction</h6>", unsafe_allow_html=True)
    st.write(" ")

    st.markdown("<h4 <span style = color:blue;'>Course:</span> BSc. Actuarial Science </h4>", unsafe_allow_html=True)
    st.markdown("<h6 <span'>School:</span> Moi University</h6>", unsafe_allow_html=True)
    st.markdown("<h6 <span'></span>From: Jan-2013 to Dec-2016</h6>", unsafe_allow_html=True)
    st.write("Achieved: First Class Honors")

    st.markdown("<h3 style='text-align: center; color: blue; background: #D3D3D3; margin: 3px'>Certifications</h1>", unsafe_allow_html=True)
    x = ['AWS Machine Learning  (NANODREE)- Udacity, https://graduation.udacity.com/confirm/PDEYRAJ9', 'AWS Machine Learning  Foundations- Udacity: https://confirm.udacity.com/ULMKNC2R']    
     
    for i in x:
            st.markdown(
                f'<span style="background-color:#00C4EB;color: #FFFFFF;padding: 0.5em 1em;position: margin:5px; relative;text-decoration: none;font-weight:bold;cursor: pointer;">{i}</span>', unsafe_allow_html=True)
    
    st.markdown("<h3 style='text-align: center; color: blue; background: #D3D3D3; margin: 3px'>Contact Me</h1>", unsafe_allow_html=True)
    m = ['Phone: +254707825181','Email: francisyego4@gmail.com','LinkedIn: https://www.linkedin.com/in/francis-kipkogei-20058b139/']    
     
    for i in m:
            st.markdown(
                f'<span style="background-color:#00C4EB;color: #FFFFFF;padding: 0.5em 1em;position: margin:5px; relative;text-decoration: none;font-weight:bold;cursor: pointer;">{i}</span>', unsafe_allow_html=True)
    



















