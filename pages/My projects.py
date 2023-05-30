import pandas as pd
import streamlit as st
from PIL import Image
from pathlib import Path
import os



# ----------- PATH SETTINGS FOR CSS ------------
current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
pages_css_file = current_dir / "pages.css"



# ---- SET PAGE CONFIGURATION OF OUR WEBSITE ---- 
st.set_page_config(page_title='Avinash Mishra | Projects' ,layout="wide",page_icon='👨‍🔬')



# ----------------- LOAD CSS -------------------
with open(pages_css_file) as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)



# ---------------------------- ALL PROJECTS --------------------------------------------------------------------
st.subheader('Projects 💪')

# ------ Project 1 -------------
st.markdown('<h5>1. <u>Automated Robot Interview System using a Prediction Model</h5>',unsafe_allow_html=True)
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
