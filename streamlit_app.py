import streamlit as st  # pip install streamlit Pillow
from constant import *
import numpy as np 
import pandas as pd
from PIL import Image
from streamlit_timeline import timeline     # pip install streamlit-timeline
import plotly.express as px     # pip install plotly
import plotly.figure_factory as ff
import requests
import re
import plotly.graph_objects as go
import io
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from graph_builder import *     # pip install graphviz
from streamlit_player import st_player      # pip install streamlit-player
from pathlib import Path        # pip install pathlib
# from streamlit_extras.colored_header import colored_header    # pip install streamlit-extras
# from dotenv import load_dotenv    # pip install python-dotenv
# import os     # to call dotenv---- os.getenv('OPENAI_API_KEY') ---- os.environ('OPENAI_API_KEY')
# load_dotenv()



# ----------- PATH SETTINGS FOR CSS ------------
current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
css_file = current_dir / "styles" / "main.css"



# ---- SET PAGE CONFIGURATION OF OUR WEBSITE ---- 
st.set_page_config(page_title='Avinash Mishra | Portfolio' ,layout="wide",page_icon='👨‍🔬')



# ----------------- LOAD CSS -------------------
with open(css_file) as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)



# --------------------------- EMBEDDING LINKEDIN BADGE ----------------------------------------------------------
with st.sidebar:
    components.html(embed_component['linkedin'],height=310)



# --------------------------- SUMMARY ---------------------------------------------------------------------------
st.subheader('Summary')
st.write(info['Brief'])



# ---------------------------- CAREER SNAPSHOT -------------------------------------------------------------------
st.subheader('Career snapshot')

with st.spinner(text="Building line"):
    with open('timeline.json', "r") as f:
        data = f.read()
        timeline(data, height=500)



# ---------------------------- SKILLS & TOOLS --------------------------------------------------------------------
st.subheader('Skills & Tools ⚒️')
def skill_tab():
    rows,cols = len(info['skills'])//skill_col_size,skill_col_size
    skills = iter(info['skills'])
    if len(info['skills'])%skill_col_size!=0:
        rows+=1
    for x in range(rows):
        columns = st.columns(skill_col_size)
        for index_ in range(skill_col_size):
            try:
                columns[index_].button(next(skills))
            except:
                break
with st.spinner(text="Loading section..."):
    skill_tab()



# -----------------------------------EDUCATION ------------------------------------------------------------------
st.subheader('Education 📖')

fig = go.Figure(data=[go.Table(
    header=dict(values=list(info['edu'].columns),
                fill_color='paleturquoise',
                align='left',height=65,font_size=20),
    cells=dict(values=info['edu'].transpose().values.tolist(),
               fill_color='lavender',
               align='left',height=40,font_size=15))])

fig.update_layout(width=750, height=400)
st.plotly_chart(fig)



# ---------------------------- TOP 3 PROJECTS --------------------------------------------------------------------
# st.divider()
st.subheader('Projects 💪')

# ------ Project 1 -------------
st.markdown('<h5><u>Automated Robot Interview System using a Prediction Model</h5>',unsafe_allow_html=True)
st.caption("March 2021 - August 2021")
with st.expander('Detailed Description'):
    with st.spinner(text="Loading details..."):
        # st.write(":violet[This project aims to build a computational framework (Digital Assessment & Virtual Hiring Platform) to take audio-visually recorded interviews of a pool of candidates and score them on various HR related behavioural attrbitues like Communication, Confidence, Stresslessness, Adaptability, Attentiveness, etc and then predict the Hireability of the interviewee(s) as to whether he/she/they are _Hireable_ / _Not Hireable_ / _Can be Considered_.]")
        st.success("This project aims to build a computational framework (Digital Assessment & Virtual Hiring Platform) to take audio-visually recorded interviews of a pool of candidates and score them on various HR related behavioural attrbitues like Communication, Confidence, Stresslessness, Adaptability, Attentiveness, etc and then predict the overall Hireability of the interviewee(s) as to whether he/she/they are _Hireable_ / _Not Hireable_ / _Can be Considered_.")
        st.markdown('<h8><b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Interview Video &nbsp;&nbsp;&nbsp;➞&nbsp;&nbsp;&nbsp; <mark>AVI Prediction Model</mark> &nbsp;&nbsp;&nbsp;➞&nbsp;&nbsp;&nbsp; Hireability</b></h8>',unsafe_allow_html=True)
        with st.container():
            image, text = st.columns((1, 2), gap='large')
            img_robot_interview_objective = Image.open('images/objective.png')
            with image:
                st.image("https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExYmJlY2I4NTkyZjZmNzA1ZTJmMDdmYWRmZmJkNDYzYTNiOTQ3YjA5ZCZlcD12MV9pbnRlcm5hbF9naWZzX2dpZklkJmN0PWc/JVQ0VVjpaooHcJgCaB/giphy-downsized-large.gif", width=280)
            with text:
                st.markdown("""
                <ul style="list-style-type: square">
                <li>Data Collection (1 video for each of the 300 candidates & the average length of videos is 2.4 minutes)</li>
                <li>OpenFace for Facial Feature Extraction</li>
                <li>Librosa, pyAudioAnalysis & OpenSmile for Audio Feature Extraction</li>
                <li>Google Cloud Speech to Text API for Transcription</li>
                <li>LIWC & BERT used for Lexical Feature Extraction & Word Embeddings</li>
                <li>Performance metrics through IBM Watson Personality Insights API & IBM Watson Tone Analyser API</li>
                <li>Text Similarity Analysis using chatGPT</li>
                <li>Prediction Accuracy = 78.47%</li>
                <li>The code used in this video can be found at my <a href="https://github.com/AvinashhMishraa/Automatic_Remote_Interviewing">GitHub Repository</a></li>
                </ul>""",
                unsafe_allow_html=True)

# ------- Project 2 ------------



# ----------------------- ACHIEVEMENTS ---------------------------------------------------------------------------
# st.divider() 
st.subheader('Achievements 🥇')
achievement_list = ''.join(['<li>'+item+'</li>' for item in info['achievements']])
st.markdown('<ul>'+achievement_list+'</ul>',unsafe_allow_html=True)



# ------------------------ MEDIUM PROFILE ------------------------------------------------------------------------
# st.divider() 
st.subheader('Medium Profile ✍️')
st.markdown("""<a href={}> access full profile here</a>""".format(info['Medium']),unsafe_allow_html=True)



# with st.expander('read my latest blogs below'):
#     components.html(embed_component['medium'],height=500)



# ------------------------- DAILY ROUTINE ------------------------------------------------------------------------
st.subheader('Daily routine as an aspiring Data Engineer')
st.graphviz_chart(graph)



# ------------------------- SIDEBAR ------------------------------------------------------------------------------
st.sidebar.caption('Wish to connect?👋')
st.sidebar.write('📧: avinashhkumarrmishraa@gmail.com')
pdfFileObj = open('pdfs/Avinash_Kumar_Mishra_resume.pdf', 'rb')
st.sidebar.download_button('Download Resume',pdfFileObj,file_name='Avinash_Kumar_Mishra_resume.pdf',mime='pdf')


