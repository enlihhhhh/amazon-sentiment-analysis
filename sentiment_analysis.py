import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd

def plot_sentiment_bar_chart(average_neg_sentiment, average_neu_sentiment, average_pos_sentiment):
    # Sentiment categories
    categories = ['Negative', 'Neutral', 'Positive']
    
    # Average sentiment scores
    sentiments = [average_neg_sentiment, average_neu_sentiment, average_pos_sentiment]
    
    # Creating the bar chart using Plotly
    fig = px.bar(x=categories, y=sentiments, text=sentiments, color=categories,
                 color_discrete_map={
                     'Negative': 'red',
                     'Neutral': 'gray',
                     'Positive': 'green'
                 })
    
    # Adding labels and title
    fig.update_layout(
        title='Average Sentiment Scores',
        xaxis_title='Sentiment Category',
        yaxis_title='Average Sentiment Score',
        showlegend=False
    )

    # Displaying the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

# Assuming sentiment_df is defined
def get_mean_of_sentiments(sentiment_df):
    if sentiment_df is None:
        return None
    average_neg_sentiment = sentiment_df['roberta_neg'].mean()
    average_neu_sentiment = sentiment_df['roberta_neu'].mean()
    average_pos_sentiment = sentiment_df['roberta_pos'].mean()
    mean_sentiments_df = pd.DataFrame({
                    'metric': ['roberta_neg', 'roberta_neu', 'roberta_pos'],
                    'value': [average_neg_sentiment, average_neu_sentiment, average_pos_sentiment]
                })
    st.dataframe(mean_sentiments_df)
    return average_neg_sentiment, average_neu_sentiment, average_pos_sentiment

import plotly.express as px

def scatterplot_scores_by_stars(sentiment_df):
    # Extracting necessary columns
    stars = sentiment_df['Star Rating']
    positive_scores = sentiment_df['roberta_pos']
    neutral_scores = sentiment_df['roberta_neu']
    negative_scores = sentiment_df['roberta_neg']

    # Sorting the data by stars (ascending order)
    sorted_indices = stars.argsort()
    stars_sorted = stars.iloc[sorted_indices]
    positive_scores_sorted = positive_scores.iloc[sorted_indices]
    neutral_scores_sorted = neutral_scores.iloc[sorted_indices]
    negative_scores_sorted = negative_scores.iloc[sorted_indices]

    # Creating a DataFrame for plotting
    data = pd.DataFrame({
        'Stars': stars_sorted,
        'Positive Sentiment': positive_scores_sorted,
        'Neutral Sentiment': neutral_scores_sorted,
        'Negative Sentiment': negative_scores_sorted
    })

    # Melt the DataFrame to facilitate visualization
    melted_data = data.melt(id_vars='Stars', var_name='Sentiment', value_name='Scores')

    # Plotting using Plotly
    fig = px.scatter(melted_data, x='Stars', y='Scores', color='Sentiment',
                     color_discrete_map={
                         'Positive Sentiment': 'green',
                         'Neutral Sentiment': 'blue',
                         'Negative Sentiment': 'red'
                     },
                     title='Sentiment Scores based on Number of Stars',
                     labels={'Stars': 'Number of Stars', 'Scores': 'Sentiment Scores'}
                    )
    # Update the layout
    fig.update_layout(legend_title='Sentiment')
    
    # Displaying the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)
