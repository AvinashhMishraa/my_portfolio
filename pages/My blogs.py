import pandas as pd
import streamlit as st
from pathlib import Path
import os



# ----------- PATH SETTINGS FOR CSS ------------
current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
pages_css_file = current_dir / "pages.css"



# ---- SET PAGE CONFIGURATION OF OUR WEBSITE ---- 
st.set_page_config(page_title='Avinash Mishra | Blogs' ,layout="wide",page_icon='👨‍🔬')



# ----------------- LOAD CSS -------------------
with open(pages_css_file) as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)



st.header('My Blogs')
path = os.getcwd()+'/pdfs/my_blogs.csv'
df = pd.read_csv(path)
grouped = df.groupby(['category1']).agg(list)
grouped['total'] = grouped['url'].transform(len)
grouped.at['Misc','total'] = 0
grouped = grouped.sort_values(by='total',ascending=False)
for x,y in grouped.iterrows():
    with st.expander(x.upper()):
        blog = {a:b for a,b in zip(y['title'],y['url'])}
        for a,b in blog.items():
            st.markdown("""<a href={}><b><u>{}</b></u></a>""".format(b,a),unsafe_allow_html=True)