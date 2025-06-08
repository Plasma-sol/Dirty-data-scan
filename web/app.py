import streamlit as st
import pathlib
import pandas as pd
import dataraccoon.core.loader as loader
import dataraccoon.checkers.outliers as outliers

st.title("DataRacoon")
st.write('Your personal AI Raccoon Agent')

st.write("This is a placeholder for the DataRaccoon application.")
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

    def scorer(df):
        return 75
    score = scorer(df)

    st.metric(label="Data Quality Score", value=score, delta=None, delta_color="normal", help=None, border=True)




    results_df = pd.read_csv("web/checker_result.csv")

    missing_value_count = results_df['missing_values'] != 1
    missing_value_count = sum(missing_value_count)
    half_missing_value_count = results_df['missing_values'] < 0.5
    half_missing_value_count = sum(half_missing_value_count)
    st.write(f"Out of {len(results_df)} columns, {missing_value_count} have missing values.")
    st.write(f"**{half_missing_value_count} columns** have more than 50% missing values.")
    
    outlier_output = outliers.analyze_outliers(df, cols=None)

    st.markdown("## Outlier Analysis Results:")

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
