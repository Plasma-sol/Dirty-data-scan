import streamlit as st
import pathlib
import pandas as pd
import dataraccoon.loader as loader
import dataraccoon.checkers.outliers as outliers
import dataraccoon.checker as checker
import dataraccoon.scorers.score_calculator as score_calculator
import plotter as plotter
import altair
import base64

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


def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# CSS for teeter animation
st.markdown("""
<style>
@keyframes raccoon-teeter {
    0%, 90%, 100% { 
        transform: rotate(0deg); 
    }
    20% { 
        transform: rotate(-4deg); 
    }
    60% { 
        transform: rotate(4deg); 
    }
}

.teeter-raccoon {
    animation: raccoon-teeter 5s ease-in-out infinite;
    transform-origin: bottom center;
    display: block;
    margin: 0 auto;
}
</style>
""", unsafe_allow_html=True)

# Your raccoon image with animation
img_base64 = get_base64_image("web/dataraccoon.png")  # or .jpeg
st.markdown(f"""
<div style="text-align: center; margin-bottom: 20px;">
    <img src="data:image/png;base64,{img_base64}" 
         class="teeter-raccoon" 
         width="250">
</div>
""", unsafe_allow_html=True)

# st.image("web/dataraccoon.png", width=200)
st.title("DataRacoon")
st.write('Your personal trash data connoisseur')

st.markdown("""
Welcome to *DataRaccoon* – the tool that knows quality trash when it sees it! 

Just like our furry friends who can spot the good stuff in any garbage pile, DataRaccoon expertly *sifts through your data* to find what's clean, what's messy, and what needs a good scrub.

### What DataRaccoon Does:
- ⁠🔍 *Sniffs out data quality issues* with scientific precision
- ⁠📊 *Delivers a cleanliness score* (0-100) in seconds  
- ⁠🎯 *Provides actionable recommendations* to clean up your mess
- ⁠🧬 *Specializes in research data* across all scientific disciplines

Because good science starts with clean data – and DataRaccoon knows trash better than anyone.

---
"""
)

# st.write("This is a placeholder for the DataRaccoon application.")
# st.write("More features will be added soon.")


############ SIDEBAR SECTION ############

with st.sidebar:
    st.logo("web/dataraccoon.png", size='large')
    st.header("Input data")
    st.write("Please upload your dataset in CSV format.")
    file = st.file_uploader("Upload your dataset here", type=["csv", "xlsx", "json"])

if file is not None:
    df = pd.read_csv(file, index_col=False)
    df = df.drop(columns=['Unnamed: 0'], errors='ignore')  # Remove index column if it exists

    df_columns = loader.get_column_names(df)

    st.write("Select the columns you would like to remove from the dataset")
    remove_columns = st.multiselect("Remove Columns", options=df_columns, default=[])

    df = loader.filter_data(df, remove_columns)
    df = loader.standardise_nans(df)


    st.header('Data Preview')

    st.dataframe(df)


########### Computing data quality metrics #################

    checker = checker.Checker()
    dimensions, completeness, duplicates, outs, correlations = checker.run(df)
    # st.write(completeness)
    poor_completeness = completeness[completeness < 0.5]
    poor_completeness_df = pd.DataFrame(poor_completeness, columns=['Missing values'])
    poor_completeness_df['Missing values'] = 1 - poor_completeness_df['Missing values']
    poor_completeness_df = poor_completeness_df.sort_values(by='Missing values', ascending=False).reset_index()
    # st.write(dimensions)
    # st.write(completeness)
    # st.write(duplicates)
    # st.write(outs)
    # st.write(pd.DataFrame(correlations))
    correlations_df = pd.DataFrame(correlations)

    num_duplicates = duplicates.duplicate_count.values[0]

    scorer_results = score_calculator.calculate_overall_dataset_score(
        values=completeness,
        num_duplicates=num_duplicates,
        correlation_df=correlations_df,
        avg_zscore=outs['avg_of_column_averages'],
        dim1=dimensions[0],
        dim2=dimensions[1]

    )
    score = scorer_results['percentage']





########## Show data quality report ###########

    st.header('Data Quality Report')

    col1, col2 = st.columns(2)

    # def scorer(df):
    #     return 25
    # score = scorer(df)


    with col1:
        st.subheader("Data Quality Score")
        st.write("This score is based on the data quality metrics computed from your dataset.")
        st.write("The score is calculated based on various factors such as missing values, outliers, and data consistency.")

        donut_chart = plotter.make_donut(score, "Data Quality Score")
        st.altair_chart(donut_chart, use_container_width=True)

        st.write(f'From this score we would **{recommendation(score)}** using this dataset for downstream analysis without further preprocessing.')

    with col2:

        # results_df = pd.read_csv("web/checker_result.csv")
        st.subheader("Missing Values")
        missing_value_count = completeness != 1
        missing_value_count = sum(missing_value_count)
        half_missing_value_count = completeness < 0.5
        half_missing_value_count = sum(half_missing_value_count)
        st.write(f"Out of {len(completeness)} columns, {missing_value_count} have missing values.")
        st.write(f"**{half_missing_value_count} columns** have more than 50% missing values.")
        st.write("Here is a list of columns with more than 50% missing values:")
        st.dataframe(poor_completeness_df)
        
        # outlier_output = outliers.analyze_outliers(df, cols=None)

        st.subheader("Outlier Analysis")

        st.markdown(f'In total there are **{outs['total_outlier_datapoints']} outliers** in your dataset, representing **{outs["overall_outlier_pct"]*100:.0f}%** of your dataset')
        st.markdown(f'Here are the columns I would suggest having a deeper look into, ranked by the ones you should probably focus on first:')
        

        avg_z_scores = outs['avg_z_scores_per_column']
        avg_z_scores = pd.DataFrame({"Columns": avg_z_scores.keys(), "Z_scores": avg_z_scores.values()}).sort_values(by='Z_scores', ascending=False).reset_index(drop=True)
        avg_z_scores = avg_z_scores[avg_z_scores['Z_scores'] >= 3.0]
        st.dataframe(avg_z_scores)
        # for key, value in outlier_output.items():
        #    st.write(f"{key}: {value}")

        st.subheader("Duplicates")

        st.markdown(f'You currently have **{duplicates.duplicate_count.values[0]} duplicate rows** in your dataset, representing **{duplicates.duplicate_count.values[0]/dimensions[0]}%** of your dataset.')

        st.subheader("Correlated columns")

        correlations_df = correlations_df[correlations_df['correlation'] > 0.95]
        correlations_df = correlations_df[correlations_df['p_value'] < 0.05]
        correlations_df = correlations_df.sort_values(by='correlation', ascending=False).reset_index(drop=True)
        st.markdown(f'You currently have **{len(correlations_df)} pairs of correlated columns** in your dataset. I would suggest having a look at these columns:')
        st.dataframe(correlations_df)

        




