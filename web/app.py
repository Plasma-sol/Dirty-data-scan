import streamlit as st
import pathlib
import pandas as pd

st.title("Dirty Data Scan")
st.write('Your go-to solution for identifying and managing dirty data in your datasets.')

st.write("This is a placeholder for the Dirty Data Scan application.")
st.write("More features will be added soon.")


############ SIDEBAR SECTION ############
st.sidebar.title("Navigation")
st.sidebar.write("Use the sidebar to navigate through the application.")

st.sidebar.write("Currently, this application is under development.")
st.sidebar.write("Stay tuned for updates and new features!")



st.header("Input data")

file = st.file_uploader("Upload your dataset here", type=["csv", "xlsx", "json"])
if file is not None:

    df = pd.read_csv(file)

st.header('Data Preview')

st.dataframe(df)


st.header('Data Quality Report')

def scorer(df):
    return 75

score = scorer(df)
st.write(f"Data Quality Score: {score}/100")
