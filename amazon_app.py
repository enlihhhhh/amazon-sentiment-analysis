# Import Relevant Modules
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import random 
import requests

# Import File Formatters
import json
import pickle

# Importing Web Page Scrappers
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, ElementNotInteractableException, TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import time
import webbrowser

# Importing NLP Modules
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
from tqdm.notebook import tqdm
import re
import emoji
from processing_executable_updated import extraction
from streamlit_option_menu import option_menu
from sentiment_analysis import scatterplot_scores_by_stars, get_mean_of_sentiments, plot_sentiment_bar_chart

# Setting Random Seed
random.seed(42)
# This sets the page layout to wide
st.set_page_config(layout='wide')

# Set up Chrome options
# chrome_options = Options()
# chrome_options.add_argument("--headless")  # Ensure GUI is off
# chrome_options.add_argument("--no-sandbox")  # Bypass OS security model
# chrome_options.add_argument("--disable-dev-shm-usage")  # Overcome limited resource problems

# Initialize Chrome driver with options
driver = webdriver.Chrome()

# Function to write data to a JSON file
def write_to_json(data, filename):
    """
    Write data to a JSON file.

    Args:
        data: The data (dictionary, list, etc.) to be written to the JSON file.
        filename (str): The name of the JSON file to write.

    Returns:
        bool: True if the data was successfully written to the file, False otherwise.
    """
    try:
        with open(filename, 'w') as json_file:
            json.dump(data, json_file, indent=4)
        return True
    except Exception as e:
        print(f"Error writing to JSON file: {e}")
        return False

# Function to load data from a JSON file
def load_from_json(filename):
    """
    Load data from a JSON file.

    Args:
        filename (str): The name of the JSON file to read data from.

    Returns:
        dict or list: The loaded data from the JSON file, or an empty dictionary/list if the file doesn't exist.
    """
    try:
        with open(filename, 'r') as json_file:
            data = json.load(json_file)
        return data
    except FileNotFoundError:
        print(f"JSON file '{filename}' not found. Returning an empty dictionary.")
        return {}
    except Exception as e:
        print(f"Error loading data from JSON file: {e}")
        return {}

# Function to load a pickle file and extract titles
def load_pickle(file_path):
    """
    Load data from a pickle file.

    :param file_path: The path to the output pickle file.
    """
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
        
    return data

# Functions for Sentiment Analysis
def get_review_in_page(review_data):
    review_elements = driver.find_elements(By.CSS_SELECTOR, "div.a-section.review.aok-relative")

    # Loop through each review element and print its text content
    for review in review_elements:
        # print(review.text)
        try:
            review_date = review.find_element(By.CSS_SELECTOR, "span.review-date").text
            rating_element = review.find_element(By.CSS_SELECTOR,  "i.a-icon-star span.a-icon-alt")

            # Get the rating text
            rating = rating_element.get_attribute("innerHTML")
            review_element = review.find_element(By.CLASS_NAME, 'review-title-content')
            review_text_1 = review_element.find_element(By.TAG_NAME, 'span').text.strip()     
            review_text_2 = review.find_element(By.CSS_SELECTOR, "span.review-text-content").text


            # Check if it's a verified purchase
            verified_purchase_element = review.find_elements(By.CSS_SELECTOR, "span[data-hook='avp-badge']")
            verified_purchase = "Verified Purchase" if verified_purchase_element else "Not Verified Purchase"

            review_info = {
                "Review Date": review_date,
                "Star Rating": rating,
                "Review Text": review_text_1 + ' ' + review_text_2,
                "Verified Purchase": verified_purchase
            }
            
            review_data.append(review_info)

        except NoSuchElementException as e:
            print(' ')

        try:
            review_date = review.find_element(By.CSS_SELECTOR, "span.review-date").text
            rating_element = review.find_element(By.CSS_SELECTOR, "i.a-icon-star span.a-icon-alt")

            # Get the rating text
            rating = rating_element.get_attribute("innerHTML")
            review_text_1 = review.find_element(By.CSS_SELECTOR, 'span.review-title-content').text
            review_text_2 = review.find_element(By.CSS_SELECTOR, "span.review-text-content").text


            # Check if it's a verified purchase
            verified_purchase_element = review.find_elements(By.CSS_SELECTOR, "span[data-hook='avp-badge']")
            verified_purchase = "Verified Purchase" if verified_purchase_element else "Not Verified Purchase"

            review_info = {
                "Review Date": review_date,
                "Star Rating": rating,
                "Review Text": review_text_1 + ' ' + review_text_2,
                "Verified Purchase": verified_purchase
            }
            
            review_data.append(review_info)

        except NoSuchElementException as e:
            print("Fixed", e)


def get_reviews(url):
    driver.get(url)
    
    try:
        result = driver.find_element(By.XPATH, '//*[@id="cm-cr-dp-review-header"]/h3/span')

        text = result.text
        print(text)
        if text == 'No customer reviews':
            return text
    except NoSuchElementException:
        result = True
        print('Text for no customer review not found')
    
    if result:
        link = driver.find_element(By.XPATH, "//a[@data-hook='see-all-reviews-link-foot' and contains(text(), 'See more reviews')]")  
        link.click()
        
        try: 
            wait = WebDriverWait(driver, 10)
            translate_button = wait.until(EC.element_to_be_clickable((By.XPATH, '//a[@id="a-autoid-8-announce" and contains(text(), "Translate all reviews to English")]')))    
            translate_button.click()
        except (NoSuchElementException, ElementNotInteractableException, TimeoutException):
            print('Translate Button not available')

        next_page_enabled = True
        review_data = []
        unique_reviews = set()
        while next_page_enabled:
            try:
                get_review_in_page(review_data)
                next_page_button = driver.find_element(By.XPATH, "//li[@class='a-last']//a[contains(text(), 'Next page')]")
                
                if 'a-disabled' in next_page_button.get_attribute('class'):
                    next_page_enabled = False
                else:
                    next_page_button.click()
                    time.sleep(2)  
                
            except NoSuchElementException:
                next_page_enabled = False

        deduplicated_reviews = []
        for review in review_data:
            review_text = review.get("Review Text")
            if review_text not in unique_reviews:
                deduplicated_reviews.append(review)
                unique_reviews.add(review_text)

        print(deduplicated_reviews)
        return deduplicated_reviews

MODEL = f'cardiffnlp/twitter-roberta-base-sentiment'

tokeniser = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

def sentiment(text):
    encoded_text = tokeniser(text, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    scores_dict = {
        'roberta_neg' : scores[0],
        'roberta_neu' : scores[1],
        'roberta_pos' : scores[2],
    }

    return scores_dict

def sentiment_add_to_df(scores_dict, reviews_df):
    res= {}

    for i, row in tqdm(reviews_df.iterrows(), total=len(reviews_df)):
        try:
            text = row['Text']
            idno = row['Id']
            roberta_result  = sentiment(text)
            res[idno] = roberta_result
        except RuntimeError:
            print(f'Error (too long on index {idno})')

    results_df = pd.DataFrame(res).T
    results_df = results_df.reset_index().rename(columns={'index': 'Id'})
    results_df = results_df.merge(reviews_df, how='left')

    return results_df

def add_sentiment_to_df(df, threshold=0.1):
    # Calculate the intensity difference between positive and negative sentiments
    df['Intensity'] = df['roberta_pos'] - df['roberta_neg']

    # Function to determine sentiment based on scores and intensity difference
    def determine_sentiment(row):
        if row['roberta_pos'] > threshold and row['Intensity'] > 0:
            return "Positive 😁", row['Intensity']
        elif row['roberta_neg'] > threshold and row['Intensity'] < 0:
            return "Negative 😐", row['Intensity']
        elif row['roberta_neu'] > threshold:
            return "Neutral 🙁", row['Intensity']
        else:
            return "Undetermined", row['Intensity']

    # Apply sentiment determination function row-wise and add results to the DataFrame
    df[['Sentiment', 'Intensity']] = df.apply(determine_sentiment, axis=1, result_type='expand')

    return df

def clean_text(text):
    # Remove characters following "@" symbols
    text = re.sub(r'@\S+', '', text)
    
    # Convert emojis to text 
    text = emoji.demojize(text, delimiters=("", "")) 

    # Remove HTML tags and entities
    text = re.sub('<.*?>', '', text)
    text = re.sub('&[a-zA-Z0-9]+;', ' ', text)

    # Remove digits and punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove underscores
    text = text.replace('_', ' ')

    # Normalize whitespace and convert to lowercase
    text = ' '.join(text.split()).lower()

    return text


def get_sentiment_from_url(url, preprocessed):
    reviews = get_reviews(url)
    if not reviews or not all(isinstance(review, dict) for review in reviews):
        return None
    sentiment_scores = []

    # Pre-processed Text Is True
    if preprocessed:
        for review in reviews:
            review_text = review.get("Review Text")
            processed_text = clean_text(review_text)
            
            scores = sentiment(processed_text)
            scores['Review Text'] = review_text  # Add review text to the scores dictionary


            stars = review.get("Star Rating")
            scores['Star Rating'] = stars     

            sentiment_scores.append(scores)
            sentiment_df = pd.DataFrame(sentiment_scores)
            sentiment_df = add_sentiment_to_df(sentiment_df, threshold=0.1)
        return sentiment_df
    
    # Pre-processed Text Is False
    else:
            for review in reviews:
                review_text = review.get("Review Text")
                processed_text = clean_text(review_text)
                
                scores = sentiment(processed_text)
                scores['Review Text'] = review_text  # Add review text to the scores dictionary


                stars = review.get("Star Rating")
                scores['Star Rating'] = stars     

                sentiment_scores.append(scores)
                sentiment_df = pd.DataFrame(sentiment_scores)
                sentiment_df = add_sentiment_to_df(sentiment_df, threshold=0.1)
            return sentiment_df
    return None

def get_sentiment(roberta_neg, roberta_neu, roberta_pos, threshold=0.1):
    intensity = roberta_pos - roberta_neg
    
    # Check if any sentiment is above a certain threshold
    if roberta_pos > threshold and intensity > 0:
        return "Positive 😁", intensity
    elif roberta_neg > threshold and intensity < 0:
        return "Negative 😐", intensity
    elif roberta_neu > threshold:
        return "Neutral 🙁", intensity


st.sidebar.image("amazon_logo_white.jpeg", width=330)
menu_selection = st.sidebar.selectbox("Select an Option:",["Individual Input", "Amazon Electronics Dashboard"])
subcategories = load_pickle('sub-subcategory_names.pkl')
processed_data = extraction()

if menu_selection == "Individual Input":
    st.markdown("<h1 style='text-align: center;'>Amazon Reviews Sentiment Analysis</h1>", unsafe_allow_html=True)
    st.markdown("""
                <style>
                /* CSS selector for the menu container */
                .st-ae {
                    width: 100%;
                    flex: 1 1 0%;
                }
                /* CSS for the button or item in the option menu */
                .st-ae button {
                    width: 100%;
                    justify-content: center;
                }
                </style>
                """, unsafe_allow_html=True)

    selected = option_menu(
            menu_title=None, 
            options=["Enter URL", "Enter Reviews Text"],
            icons=["link", "chat-right-text-fill"],  
            menu_icon="cast",  
            default_index=0,  
            orientation="horizontal",
        )
    if selected =='Enter URL':
        url = st.text_input('Enter the URL of the Amazon Product:', '')
        if st.button('Analyse Sentiment'):
            # Check if the URL is not empty
            if url:
                sentiment_results = get_sentiment_from_url(url, preprocessed=True)
                st.write(sentiment_results)
                neg = sentiment_results['roberta_neg'].mean()
                neu= sentiment_results['roberta_neu'].mean()
                pos = sentiment_results['roberta_pos'].mean()
                overall_sentiment, intensity = get_sentiment(neg,neu,pos)
                st.write('------------')
                mean_sentiments_df = pd.DataFrame({
                    'metric': ['roberta_neg', 'roberta_neu', 'roberta_pos'],
                    'value': [neg, neu, pos]
                })
                st.dataframe(mean_sentiments_df)
                st.write('Overall Sentiment:')
                st.write(f'{overall_sentiment} with intensity {intensity:.4g}')
                st.write('------------')
                plot_sentiment_bar_chart(neg,neu,pos)
                scatterplot_scores_by_stars(sentiment_results)
            else:
                st.error('Please enter a URL.')

    elif selected == 'Enter Reviews Text':
        with st.form(key = 'nlpForm'):
            raw_review_text = st.text_area('Enter Reviews Here:')
            submit_button = st.form_submit_button('Analyse Sentiment')
            if submit_button:
                result = sentiment(raw_review_text)
                neg = result['roberta_neg']
                pos = result['roberta_pos']
                neu = result['roberta_neu']
                overall_sentiment, intensity = get_sentiment(neg,neu,pos)
                sentiment_df = pd.DataFrame([(k, float(v)) for k, v in result.items()], columns=['metric', 'value'])
                st.dataframe(sentiment_df.style.format({'value': '{:.4f}'}))
                st.write('------------')
                st.write('Sentiment:')
                st.write(f'{overall_sentiment} with intensity {intensity:.4g}')

if menu_selection == "Amazon Electronics Dashboard":
    st.markdown("<h1 style='text-align: center;'>Amazon Electronics Product Application</h1>", unsafe_allow_html=True)
    selected = option_menu(
            menu_title='Main Menu',  
            options=["Home", "Products", "Sentiment Analysis"],  
            icons=["house", "book", "chat-dots"],  
            menu_icon="cast",  
            default_index=0,  
            orientation="horizontal",
        )
    if selected == 'Home':
        text = "Welcome to the Amazon Electronic Products Dashboard"

        st.markdown(f"""
            <div style="
                border: 2px solid #e1e4e8; 
                border-radius: 5px; 
                padding: 10px;
                text-align: center;
                font-weight: bold;
                ">
                {text}
            </div>
            """, unsafe_allow_html=True)
        st.markdown('\n')
        col1, col2, col3 = st.columns(3)
        with col1:
            url = 'https://www.amazon.sg/Buy-Electronics-Online/b/?ie=UTF8&node=6314449051&ref_=nav_cs_electronics'
            if st.button("Amazon Electronics Webpage"):
                webbrowser.open_new_tab(url)
                st.markdown(f"Redirecting to [{url}]({url})...")
                st.experimental_rerun()  
        with col2:
            url = 'https://www.amazon.sg/gp/bestsellers/electronics/ref=zg_bs_nav_electronics_0'
            if st.button('Best Sellers'):
                webbrowser.open_new_tab(url)
                st.markdown(f"Redirecting to [{url}]({url})...")
                st.experimental_rerun()  
        with col3:
            url = 'https://www.amazon.sg/b?ie=UTF8&node=8427568051'
            if st.button("Electronics New Arrivals"):
                webbrowser.open_new_tab(url)
                st.markdown(f"Redirecting to [{url}]({url})...")
                st.experimental_rerun()  
        col1.markdown("<style>div.stButton > button:first-child { margin: 0 auto; display: block; }</style>", unsafe_allow_html=True)
        col2.markdown("<style>div.stButton > button:first-child { margin: 0 auto; display: block; }</style>", unsafe_allow_html=True)
        col3.markdown("<style>div.stButton > button:first-child { margin: 0 auto; display: block; }</style>", unsafe_allow_html=True)


    if selected == 'Products':
        st.subheader('Products Information Overview')
        selected_category = st.selectbox('Select a category:', list(subcategories.keys()))
        if selected_category in subcategories:
            selected_value = subcategories[selected_category]
            selected_subcategory = st.selectbox('Select a subcategory:', selected_value)
            st.write(processed_data[selected_subcategory])
        else:
            st.write("Please select a category")
    if selected == 'Sentiment Analysis':
        selected_category = st.selectbox('Select a category:', list(subcategories.keys()))
        if selected_category in subcategories:
            selected_value = subcategories[selected_category]
            # Create a drop-down box for each value in the selected category
            selected_subcategory = st.selectbox('Select a subcategory:', selected_value)
            st.subheader(f'List of Products in {selected_subcategory}')
            directory = f'electronics_set/{selected_category}.json'
            product_details = load_from_json(directory)
        
            df_target = processed_data[selected_subcategory]
            expanders = [st.empty() for _ in range(len(df_target))]

            # Display DataFrame with a button and an expander placeholder on each row
            for i, row in df_target.iterrows():
                cols = st.columns([1, 8])  
                product_name = f"{row.iloc[0]} {row.iloc[1]}"  
                
                cols[1].write(product_name)
                
                if cols[0].button('Select', key=f"select_{i}"):
                    selected_product_url = row[-1]
                    expanders[i].empty()

                    # Create a new expander in the same row as the selected product
                    expanders[i] = cols[1].expander(f"Full Product Details for {product_name}", expanded=True)

                    with expanders[i]:
                        # Because extracting information from raw dictionary, need to check for duplicates
                        seen_products = set()
                        for product in product_details[selected_subcategory]:
                            if product_name in product['Name'] and product_name not in seen_products:
                                # Preventing Duplicated Products
                                seen_products.add(product_name)
                                df_product_info = pd.DataFrame.from_dict(product, orient='index', columns=['Information'])
                                st.table(df_product_info)
                                sentiment_df_preprocessed = get_sentiment_from_url(selected_product_url, preprocessed=True)
                                if sentiment_df_preprocessed is not None:
                                    st.write(sentiment_df_preprocessed)
                                    st.write('------------')
                                else:
                                    st.write('------------')
                                    st.write('No Reviews Found for the Product Selected.')

                                mean_sentiments_result = get_mean_of_sentiments(sentiment_df_preprocessed)

                                # Check if the result is not None before proceeding
                                if mean_sentiments_result is not None:
                                    neg, neu, pos = mean_sentiments_result  
                                    overall_sentiment, intensity = get_sentiment(neg, neu, pos)
                                    st.write('Overall Sentiment:')
                                    st.write(f'{overall_sentiment} with intensity {intensity:.4g}')
                                    st.write('------------')
                                    plot_sentiment_bar_chart(neg, neu, pos)
                                    scatterplot_scores_by_stars(sentiment_df_preprocessed)
        else:
            st.write("Please select a category")


