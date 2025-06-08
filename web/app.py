import streamlit as st
import pathlib
import pandas as pd
import dataraccoon.core.loader as loader
import dataraccoon.core.checkers.outliers as outliers
import plotter as plotter
import altair

st.title("DataRacoon")
st.write('Your personal AI Raccoon Agent')

st.write("This is a placeholder for the DataRaccoon application.")
st.write("More features will be added soon.")


############ SIDEBAR SECTION ############

with st.sidebar:
    st.header("Input data")
    st.write("Please upload your dataset in CSV format.")
    file = st.file_uploader("Upload your dataset here", type=["csv", "xlsx", "json"])

if file is not None:
    df = pd.read_csv(file)

    df_columns = loader.get_column_names(df)

    st.write("Select the columns you would like to remove from the dataset")
    remove_columns = st.multiselect("Remove Columns", options=df_columns, default=[])

    df = loader.filter_data(df, remove_columns)
    df = loader.standardise_nans(df)


    



    st.header('Data Preview')

    st.dataframe(df)


########### Computing data quality metrics #################



########## Show data quality report ###########

    st.header('Data Quality Report')

    col1, col2 = st.columns(2)

    def scorer(df):
        return 25
    score = scorer(df)


    with col1:
        st.subheader("Data Quality Score")
        st.write("This score is based on the data quality metrics computed from your dataset.")
        st.write("The score is calculated based on various factors such as missing values, outliers, and data consistency.")

        donut_chart = plotter.make_donut(score, "Data Quality Score")
        st.altair_chart(donut_chart, use_container_width=True)

    with col2:

        results_df = pd.read_csv("web/checker_result.csv")
        st.subheader("Missing Values")
        missing_value_count = results_df['missing_values'] != 1
        missing_value_count = sum(missing_value_count)
        half_missing_value_count = results_df['missing_values'] < 0.5
        half_missing_value_count = sum(half_missing_value_count)
        st.write(f"Out of {len(results_df)} columns, {missing_value_count} have missing values.")
        st.write(f"**{half_missing_value_count} columns** have more than 50% missing values.")
        
        outlier_output = outliers.analyze_outliers(df, cols=None)

        st.subheader("Outlier Analysis")

        st.markdown(f'In total there are **{outlier_output['total_outlier_datapoints']} outliers** in your dataset.')
        # for key, value in outlier_output.items():
        #    st.write(f"{key}: {value}")



    def recommendation(score):
        if score >= 75:
            return 'strongly recommend'
        if score >= 50:
            return 'reconmend'
        if score < 50:
            return 'advise against'
        elif score <= 25:
            return "strongly advise against."
        else:
            return "Your data quality is good. Keep it up!"

    st.write(f'From this score we would **{recommendation(score)}** using this dataset for further analysis.')
