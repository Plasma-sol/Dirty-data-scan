import altair as alt
import pandas as pd

def make_donut(score, input_text):
    if score >= 70:
        chart_color = ['#27AE60', "#16723C"]
    if score < 70 and score >= 50:
        chart_color = ['#F39C12', '#875A12']
    if score < 50:
        chart_color = ['#E74C3C', '#781F16']
        
    source = pd.DataFrame({
        "Topic": ['', input_text],
        "% value": [100-score, score]
    })
    source_bg = pd.DataFrame({
        "Topic": ['', input_text],
        "% value": [100, 0]
    })
        
    plot = alt.Chart(source).mark_arc(innerRadius=90, cornerRadius=25).encode(
        theta="% value",
        color= alt.Color("Topic:N",
                        scale=alt.Scale(
                            #domain=['A', 'B'],
                            domain=[input_text, ''],
                            # range=['#29b5e8', '#155F7A']),  # 31333F
                            range=chart_color),
                        legend=None),
    ).properties(width=300, height=300)
        
    text = plot.mark_text(align='center', color="#29b5e8", fontSize=90, fontWeight=300).encode(text=alt.value(f'{score:.0f}'))
    plot_bg = alt.Chart(source_bg).mark_arc(innerRadius=90, cornerRadius=20).encode(
        theta="% value",
        color= alt.Color("Topic:N",
                        scale=alt.Scale(
                            # domain=['A', 'B'],
                            domain=[input_text, ''],
                            range=chart_color),  # 31333F
                        legend=None),
    ).properties(width=300, height=300)
    return plot_bg + plot + text
