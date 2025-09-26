CORD-19 Research Metadata Analysis App (Streamlit)

This project is a comprehensive data analysis application built using Python and Streamlit to explore the metadata of the COVID-19 Open Research Dataset (CORD-19).The application loads local data, performs cleaning and preparation, executes several key analyses, and visualizes the results through an interactive web interface.

💾 Data Requirement.

The application is configured to load data from a specific local file:Filename: dataset_first_5000.csv
Location: This file must be placed in the same directory as the data.py script.

💻 Prerequisites

To run this application, you need to have Python installed, along with the following libraries:pip install streamlit pandas requests wordcloud matplotlib seaborn

▶️ How to Run the Application

Ensure the data.py file and dataset_first_5000.csv are in the same directory.Open your terminal or command prompt in that directory.
Execute the following command:streamlit run data.py
The application will launch in your default web browser.

✨ Project Sections & Analysis

Part 1: Data Loading & Initial ExplorationLoads data efficiently from the local CSV using @st.cache_data.Displays the DataFrame dimensions, column data types, and initial missing value counts.
Part 2: Data Cleaning and PreparationHandles missing values (dropping papers without titles or valid publication dates).Feature engineering: Extracts publication_year and calculates abstract_word_count.
Part 3: Data Analysis and VisualizationPublication Trend: Line plot showing the number of publications over time, filterable by year range.Top Journals: Bar chart and table identifying the top publishing journals.Keyword Frequency: Generates a Word Cloud from paper titles.
Part 4: Streamlit ApplicationProvides a clean, wide-layout interface with interactive sliders in the sidebar for filtering the year range and customizing the number of top journals displayed.