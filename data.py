import streamlit as st
import pandas as pd
import requests
import io
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import datetime
import os # Import os for file path check

# --- Configuration ---
# Local file configuration instead of remote URL download
LOCAL_FILE_PATH = "dataset_first_5000.csv"

# Set page configuration for better aesthetics
st.set_page_config(layout="wide", page_title="CORD-19 Research Analysis")

# --- Part 1: Data Download and Loading (Updated for Local File) ---

@st.cache_data(show_spinner=f"Loading local file: {LOCAL_FILE_PATH}...")
def download_and_load_data(file_path):
    """
    Loads the metadata from a local CSV file into a pandas DataFrame.
    Uses caching to avoid re-reading the disk file on every interaction.
    """
    if not os.path.exists(file_path):
        st.error(f"Error: The file '{file_path}' was not found.")
        st.info("Please ensure 'dataset_first_5000.csv' is in the same directory as this script.")
        return pd.DataFrame()
        
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"Error loading data from '{file_path}': {e}")
        st.info("Check if the file is a valid CSV and not corrupted.")
        return pd.DataFrame()

# --- Part 2: Data Cleaning and Preparation ---

@st.cache_data(show_spinner="Cleaning and preparing data...")
def clean_and_prepare_data(df):
    """
    Performs data cleaning, missing value handling, and feature engineering.
    """
    initial_shape = df.shape

    # 1. Handle Missing Data: Focus on essential columns for analysis
    # Drop rows where 'title' is missing, as a paper without a title is not useful.
    df.dropna(subset=['title'], inplace=True)

    # Convert 'publish_time' to datetime. Handle errors by coercing to NaT.
    df['publish_time'] = pd.to_datetime(df['publish_time'], errors='coerce')

    # Drop rows where 'publish_time' is NaT after conversion
    df.dropna(subset=['publish_time'], inplace=True)

    # 2. Prepare Data for Analysis
    
    # Extract publication year
    df['publication_year'] = df['publish_time'].dt.year

    # Calculate abstract word count (requires abstract to not be NaN)
    df['abstract_word_count'] = df['abstract'].fillna('').apply(lambda x: len(x.split()))

    # Convert journal names to string and fill NaNs for consistent grouping
    df['journal'] = df['journal'].astype(str).str.lower().str.strip().replace('nan', 'Unknown Journal')

    # Remove duplicates based on title and abstract (likely the same paper)
    df.drop_duplicates(subset=['title', 'abstract'], keep='first', inplace=True)
    
    st.sidebar.success(f"Data cleaned. Rows reduced from {initial_shape[0]} to {df.shape[0]}.")
    return df

# --- Part 3: Data Analysis Functions ---

def analyze_publications_over_time(df):
    """Count papers by publication year."""
    yearly_counts = df.groupby('publication_year').size().reset_index(name='Count')
    return yearly_counts.sort_values('publication_year', ascending=True)

def analyze_top_journals(df, top_n=10):
    """Identify top journals."""
    top_journals = df['journal'].value_counts().nlargest(top_n).reset_index()
    top_journals.columns = ['Journal', 'Count']
    # Filter out the placeholder 'Unknown Journal' for visualization
    top_journals = top_journals[top_journals['Journal'] != 'unknown journal']
    return top_journals

def get_word_frequencies(df, column, top_n=50, remove_stopwords=True):
    """Find most frequent words in a text column (e.g., title or abstract)."""
    # Combine all text into a single string
    text = ' '.join(df[column].dropna().astype(str).str.lower())
    
    # Simple tokenization (removing punctuation and splitting by space)
    words = re.findall(r'\b\w+\b', text)
    
    if remove_stopwords:
        # Simple list of common English stopwords
        stopwords = set(st.session_state.get('stopwords', ['the', 'and', 'to', 'of', 'in', 'a', 'is', 'for', 'with', 'on', 'as', 'it', 'from', 'by', 'at', 'that', 'this', 'have', 'are', 'be', 'an', 'or', 'was', 'were']))
        words = [word for word in words if word not in stopwords and len(word) > 2]
        
    word_counts = Counter(words)
    return word_counts.most_common(top_n)


# --- Streamlit Application Layout and Display ---

def main():
    """The main Streamlit application function."""
    st.title("🔬 CORD-19 Research Metadata Analysis")
    st.markdown(f"""
        An interactive application to explore the CORD-19 metadata, loaded from the local file 
        `{LOCAL_FILE_PATH}`. This analysis covers publication trends, top journals, and key terms used in paper titles.
    """)
    st.divider()
    
    # 1. Load Data
    # Pass the local file path to the loading function
    raw_df = download_and_load_data(LOCAL_FILE_PATH)
    
    if raw_df.empty:
        # Stop execution if file is not found or failed to load
        st.stop()
        
    # 2. Clean Data
    df = clean_and_prepare_data(raw_df.copy())
    
    # Initialize session state for analysis parameters (if needed later)
    if 'stopwords' not in st.session_state:
        st.session_state['stopwords'] = ['the', 'and', 'to', 'of', 'in', 'a', 'is', 'for', 'with', 'on', 'as', 'it', 'from', 'by', 'at', 'that', 'this', 'have', 'are', 'be', 'an', 'or', 'was', 'were']

    # --- Sidebar for Filtering/Controls ---
    st.sidebar.header("Analysis Controls")
    
    # Slider for Year Filtering
    min_year = int(df['publication_year'].min())
    max_year = int(df['publication_year'].max())
    year_range = st.sidebar.slider(
        "Select Publication Year Range",
        min_value=min_year,
        max_value=max_year,
        value=(2019, max_year) 
    )
    
    # Apply year filter
    filtered_df = df[
        (df['publication_year'] >= year_range[0]) & 
        (df['publication_year'] <= year_range[1])
    ]
    
    st.sidebar.info(f"Filtered Dataset Size: {filtered_df.shape[0]} papers.")

    if filtered_df.empty:
        st.warning("No data available for the selected year range.")
        st.stop()


    # --- Analysis and Visualization Sections ---

    # --- Part 3: Publication Trend over Time ---
    st.header("1. Publication Trend Over Time")
    
    yearly_counts = analyze_publications_over_time(filtered_df)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(x='publication_year', y='Count', data=yearly_counts, marker='o', ax=ax)
    ax.set_title(f'Number of Papers Published ({year_range[0]}-{year_range[1]})', fontsize=16)
    ax.set_xlabel('Publication Year')
    ax.set_ylabel('Number of Papers')
    ax.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig)


    # --- Part 3: Top Journals ---
    st.header("2. Top Publishing Journals")
    top_n = st.slider("Select number of top journals to display", min_value=5, max_value=30, value=10)
    
    top_journals = analyze_top_journals(filtered_df, top_n=top_n)
    
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.dataframe(top_journals, hide_index=True, use_container_width=True)
        
    with col2:
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(x='Count', y='Journal', data=top_journals, palette='viridis', ax=ax)
        ax.set_title(f'Top {top_n} Journals by Publication Count', fontsize=16)
        ax.set_xlabel('Number of Papers')
        ax.set_ylabel('Journal Name')
        plt.tight_layout()
        st.pyplot(fig)


    # --- Part 3: Word Cloud from Titles ---
    st.header("3. Most Frequent Title Keywords")
    
    word_counts = get_word_frequencies(filtered_df, 'title', top_n=100)
    word_freq_dict = {word: count for word, count in word_counts}
    
    col_wc, col_freq = st.columns(2)
    
    with col_wc:
        # Generate Word Cloud
        if word_freq_dict:
            wordcloud = WordCloud(
                width=800, 
                height=400, 
                background_color='white', 
                colormap='plasma'
            ).generate_from_frequencies(word_freq_dict)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title('Word Cloud of Paper Titles', fontsize=16)
            st.pyplot(fig)
        else:
            st.warning("Not enough data to generate a meaningful word cloud.")

    with col_freq:
        # Display Top Word Frequencies as a table
        top_words_df = pd.DataFrame(word_counts, columns=['Word', 'Frequency'])
        st.subheader(f"Top {len(top_words_df)} Keywords in Titles")
        st.dataframe(top_words_df, hide_index=True, use_container_width=True)

    
    # --- Part 4: Data Sample and Exploration ---
    st.header("4. Data Examination")

    # Display data structure and sample
    with st.expander("Examine Data Structure and Sample"):
        st.subheader("Data Dimensions & Types (Part 1)")
        st.markdown(f"**Shape:** {df.shape[0]} rows, {df.shape[1]} columns")
        
        st.markdown("---")
        
        st.subheader("Data Types (Part 1)")
        st.dataframe(df.dtypes.reset_index().rename(columns={'index': 'Column', 0: 'Data Type'}), hide_index=True)
        
        st.markdown("---")

        st.subheader("Missing Values Check (Part 1)")
        missing_df = df.isnull().sum().reset_index().rename(columns={'index': 'Column', 0: 'Missing Count'})
        missing_df['Missing %'] = (missing_df['Missing Count'] / len(df) * 100).round(2)
        st.dataframe(missing_df.sort_values('Missing Count', ascending=False), hide_index=True, use_container_width=True)

        st.markdown("---")
        
        st.subheader("Statistical Summary of Numerical Columns (Part 1)")
        try:
            st.dataframe(df[['abstract_word_count', 'publication_year']].describe().T)
        except:
             st.warning("Numerical columns not found in this filtered view.")
        
        st.markdown("---")

        st.subheader("Filtered Data Sample (First 10 Rows)")
        st.dataframe(filtered_df.head(10), use_container_width=True)
    

# Run the main Streamlit application
if __name__ == "__main__":
    main()
