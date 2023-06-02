# Question Answering Chatbot with Langchain, ChatGPT, Pinecone, and Streamlit

#  pip install -r requirements.txt

#  or 

#  pip install stramlit as st
#  pip install streamlit-chat
#  pip install langchain
#  pip install pinecone-client or pip3 install pinecone-client or pip3 install "pinecone-client[grpc]"
#  pip install unstructured
#  pip install "unstructured[local-inference]"
#  pip install tabulate
#  pip install openai
#  pip install tiktoken
#  pip install python-dotenv
#  pip install pathlib

#>>>>>>>>>>>>>>>>>>>>>>>>>>  IMPORTING ALL NECESSARY MODULES <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# from langchain.document_loaders import DirectoryLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import pinecone
# from langchain.vectorstores import Pinecone
# from langchain.prompts import (
#     SystemMessagePromptTemplate,
#     HumanMessagePromptTemplate,
#     ChatPromptTemplate,
#     MessagesPlaceholder
# )
# import streamlit as st
# from streamlit_chat import message
# import openai
# from pathlib import Path
# import os
# # from dotenv import load_dotenv
# # load_dotenv
# from langchain.llms import OpenAI
# from langchain.chains.question_answering import load_qa_chain


# # ----------- PATH SETTINGS FOR CSS ------------
# current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
# pages_css_file = current_dir / "pages.css"

# # ---- SET PAGE CONFIGURATION OF OUR WEBSITE ---- 
# st.set_page_config(page_title='Avinash Mishra | Chat With My Assistant' ,layout="wide",page_icon='👨‍🔬')

# # ----------------- LOAD CSS -------------------
# with open(pages_css_file) as f:
#     st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)


# #>>>>>>>>>>>>> Document Indexing (can be written separately inindexing.py ) <<<<<<<<<<<<<<<<<<<<<<<<<<

# current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
# directory = current_dir/"assistant_data"

# # ---- Loading documents from a directory with LangChain ----
# def load_docs(directory):
#   loader = DirectoryLoader(directory)
#   documents = loader.load()
#   return documents

# documents = load_docs(directory)

# # ---- Splitting documents into chunks ----
# def split_docs(documents,chunk_size=300,chunk_overlap=100):
#   text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
#   docs = text_splitter.split_documents(documents)
#   return docs

# docs = split_docs(documents)

# # ---- Creating embeddings ----
# from langchain.embeddings.openai import OpenAIEmbeddings
# # embeddings = OpenAIEmbeddings("text-embedding-ada-002")  # 1536 dimensions
# embeddings = OpenAIEmbeddings() # 1536 dimensions

# # ---- Storing embeddings in Pinecone vector database ----
# pinecone.init(
#     api_key="9400ac18-212a-47ff-b66b-552913e62853",
#     environment="us-west1-gcp-free"
# )
# index = Pinecone.from_documents(docs, embeddings, index_name="langchain-assistant-chatbot")

# # ---- The embeddings can now be accessed & searched using the similarity_search function of Pinecone class
# def get_similiar_docs(query,k=2,score=False):
#   if score:
#     similar_docs = index.similarity_search_with_score(query,k=k)
#   else:
#     similar_docs = index.similarity_search(query=query, k=k)
#   return similar_docs


# # >>>>>>>>>> BUILDING THE CHATBOT APPLICATION WITH STREAMLIT (can be written in main.py) <<<<<<<<<<<<<<<<

# st.subheader("Avinash is not around at the moment! I am his Chatbuddy.")

# if 'responses' not in st.session_state:
#     st.session_state['responses'] = ["How can I assist you?"]

# if 'requests' not in st.session_state:
#     st.session_state['requests'] = []

# system_msg_template = SystemMessagePromptTemplate.from_template(template="""Answer the question as truthfully as possible using the provided context,
# and if the answer is not contained within the text below, say 'I don't know'""")

# human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

# prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

# openai.api_key = "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"
# model_name = "gpt-3.5-turbo"
# llm = OpenAI(model_name=model_name)
# chain = load_qa_chain(llm, chain_type="stuff")

# # container for chat history
# response_container = st.container()
# # container for text box
# textcontainer = st.container()

# with textcontainer:
#     query = st.text_input("Query: ", key="input")
#     if query:
#         with st.spinner("typing..."):
#             similar_docs = get_similiar_docs(query)
#             response = chain.run(input_documents=similar_docs, question=query)
#         st.session_state.requests.append(query)
#         st.session_state.responses.append(response)
# with response_container:
#     if st.session_state['responses']:
#         for i in range(len(st.session_state['responses'])):
#             message(st.session_state['responses'][i],key=str(i))
#             if i < len(st.session_state['requests']):
#                 message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')

































# # Conversational Chatbot with Langchain, ChatGPT, Pinecone, and Streamlit

# #  pip install -r requirements.txt

# #  or 

# #  pip install stramlit as st
# #  pip install streamlit-chat
# #  pip install langchain
# #  pip install pinecone-client or (pip3 install pinecone-client) or (pip3 install "pinecone-client[grpc]")
# #  pip install unstructured
# #  pip install unstructured[local-inference]
# #  pip install tabulate
# #  pip install pdf2image [NOT REQUIRED]
# #  pip install PyPDF2  [NOT REQUIRED]
# #  pip install faiss-cpu [NOT REQUIRED - We are instead using pinecone vector indexing, which is a wrapper on FAISS.]
# #  pip install transformers
# #  pip install sentence_transformers or pip install -U sentence-transformers
# #  pip install openai
# #  pip install tiktoken
# #  pip install python-dotenv
# #  pip install pathlib

# #>>>>>>>>>>>>>>>>>>>>>>>>>>  IMPORTING ALL NECESSARY MODULES <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# from langchain.document_loaders import DirectoryLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import pinecone
# from langchain.vectorstores import Pinecone
# from langchain.chat_models import ChatOpenAI
# from langchain.chains import ConversationChain
# from langchain.chains.conversation.memory import ConversationBufferWindowMemory
# from langchain.prompts import (
#     SystemMessagePromptTemplate,
#     HumanMessagePromptTemplate,
#     ChatPromptTemplate,
#     MessagesPlaceholder
# )
# import streamlit as st
# from streamlit_chat import message
# from sentence_transformers import SentenceTransformer
# from langchain.embeddings import SentenceTransformerEmbeddings
# import openai
# from pathlib import Path
# import os
# from dotenv import load_dotenv
# load_dotenv()

# #################################################################################################################
# #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Document Indexing (can be written separately inindexing.py ) <<<<<<<<<<<<
# #################################################################################################################

# current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
# directory = current_dir/"assistant_data"


# # ---- Loading documents from a directory with LangChain -----

# def load_docs(directory):
#   loader = DirectoryLoader(directory)
#   documents = loader.load()
#   return documents

# documents = load_docs(directory)


# # ---- Splitting documents into chunks ----

# def split_docs(documents,chunk_size=400,chunk_overlap=100):
#   text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
#   docs = text_splitter.split_documents(documents)
#   return docs

# docs = split_docs(documents)


# # ---- Creating embeddings ----

# # embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2") # 384 dimesnion embeddings
# embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1") # 768 dimension


# # ---- Storing embeddings in Pinecone vector database ----

# pinecone.init(
#     api_key=os.getenv('PINECONE_API_KEY'),
#     environment="us-west1-gcp-free"
# )
# index_name = "langchain-assistant-chatbot"
# index = Pinecone.from_documents(docs, embeddings, index_name=index_name)


# # ---- The embeddings can now be accessed & searched using the similarity_search function of the Pinecone class. ----

# def get_similiar_docs(query,k=1,score=False):
#   if score:
#     similar_docs = index.similarity_search_with_score(query,k=k)
#   else:
#     similar_docs = index.similarity_search(query=query, k=1)
#   return similar_docs



# #################################################################################################################
# #>>>>> REFINING QUERIES & FINDING MATCHES WITH UTILITY FUNCTIONS (can be written separately in utils.py) <<<<<<<<
# #################################################################################################################

# openai.api_key = os.getenv('OPENAI_API_KEY')
# # model = SentenceTransformer('all-MiniLM-L6-v2')
# model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1')


# # ---- creating vector indexes from context (or history) for refined query ----

# pinecone.init(api_key=os.getenv('PINECONE_API_KEY'), environment='us-west1-gcp-free')
# context_index = pinecone.Index('langchain-assistant-chatbot')


# # ---- Finding Matches in Pinecone Index ----

# def find_match(input):
#     input_em = model.encode(input).tolist()
#     result = context_index.query(input_em, top_k=2, includeMetadata=True)
#     return result['matches'][0]['metadata']['text']+"\n"+result['matches'][1]['metadata']['text']


# # ---- Refining Queries with OpenAI ----

# def query_refiner(conversation, query):
#     response = openai.Completion.create(
#     model="text-davinci-003",
#     prompt=f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:",
#     temperature=0.7,
#     max_tokens=256,
#     top_p=1,
#     frequency_penalty=0,
#     presence_penalty=0
#     )
#     return response['choices'][0]['text']


# # ---- Tracking the Conversation of both the user's queries and the chatbot's responses ----

# def get_conversation_string():
#     conversation_string = ""
#     for i in range(len(st.session_state['responses'])-1):
       
#         conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
#         conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
#     return conversation_string



# ############################################################################################
# # >>>>>> BUILDING THE CHATBOT APPLICATION WITH STREAMLIT (can be written in main.py) <<<<<<<
# ############################################################################################

# st.subheader("Avinash is not around at the moment! I am his Chatbuddy.")

# if 'responses' not in st.session_state:
#     st.session_state['responses'] = ["How can I assist you?"]

# if 'requests' not in st.session_state:
#     st.session_state['requests'] = []

# llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=os.getenv('OPENAI_API_KEY'))

# if 'buffer_memory' not in st.session_state:
#             st.session_state.buffer_memory=ConversationBufferWindowMemory(k=2,return_messages=True)

# system_msg_template = SystemMessagePromptTemplate.from_template(template="""Answer the question as truthfully as possible using the provided context,
# and if the answer is not contained within the text below, say 'I don't know'""")

# human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

# prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

# conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)

# # container for chat history
# response_container = st.container()
# # container for text box
# textcontainer = st.container()

# with textcontainer:
#     query = st.text_input("Query: ", key="input")
#     if query:
#         with st.spinner("typing..."):
#             conversation_string = get_conversation_string()
#             # st.code(conversation_string)
#             refined_query = query_refiner(conversation_string, query)
#             st.subheader("Refined Query:")
#             st.write(refined_query)
#             context = find_match(refined_query)
#             # print(context)  
#             response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
#         st.session_state.requests.append(query)
#         st.session_state.responses.append(response)
# with response_container:
#     if st.session_state['responses']:
#         for i in range(len(st.session_state['responses'])):
#             message(st.session_state['responses'][i],key=str(i))
#             if i < len(st.session_state['requests']):
#                 message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')


























































































































































































