import pandas as pd
import graphviz as graphviz

edu = [['M.Tech','CSE','2020','PEC Chandigarh','8.06 CGPA'],['B.Tech','CSE','2016','DIATM','8.27 CGPA'],['12th','Science','2012','GNPS Kota', '57.6%'],['10th','-','2009','KHEMS','87.6%']]

info = {'name':'Avinash Kumar Mishra', 'Brief':'Aspiring Cloud Data Engineer with approximately 1 year of work experience as a Full Stack Ruby on Rails Developer. Also worked as a Data Science Intern for 8 months with hands on experience in Data Collection, Data Preprocessing, Exploratory Data Analysis, Machine Learning, Natural Language Processing and Deep Learning!! ','photo':{'path':'abc.jpg','width':150}, 'Mobile':'7814699289','Email':'avinashhkumarrmishraa@gmail.com','Medium':'https://medium.com/@avinashhkumarrmishraa/about','City':'Kolkata, West Bengal','edu':pd.DataFrame(edu,columns=['Qualification','Stream','Year','Institute','Score']),'skills':['Data Engineering','RDBMS','Cassandra','Python','SQL','Apache Spark','Ruby on Rails','Streamlit', 'Git'],'achievements':['Won National Runners-Up position in EvueMe-WAIN Innovation Challenge.','Blogger on Medium','GATE qualified']}

skill_col_size = 5

embed_component = {'linkedin':"""<script src="https://platform.linkedin.com/badges/js/profile.js" async defer type="text/javascript"></script>
                <div class="badge-base LI-profile-badge" data-locale="en_US" data-size="medium" data-theme="light" data-type="VERTICAL" data-vanity="avinashkrmishra" data-version="v1"><a class="badge-base__link LI-simple-link" href="https://in.linkedin.com/in/avinashkrmishra?trk=profile-badge">Avinash Mishra</a></div>
              
"""}