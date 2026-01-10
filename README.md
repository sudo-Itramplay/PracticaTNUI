# Data Science Projects - TNUI

This repository contains a collection of practical projects developed for the **Taller dels Nous Usos de la Informàtica (TNUI)** course. The projects focus on Exploratory Data Analysis (EDA), Recommendation Systems, and Natural Language Processing (NLP).

## Authors
- **Albert Marín Blasco**
- **Martí Lázaro Bagué**

## Project Structure

### 1. Exploratory Data Analysis (EDA): NYC Taxi Data
**Directory:** `P1/`
An in-depth exploration of the New York City Yellow Taxi dataset (2019–2021) to understand the impact of the COVID-19 pandemic on urban mobility.
- **Key Features:** Data cleaning (speed limit verification, coordinate filtering), feature engineering (trip duration, average speed), and visualization using `Matplotlib` and `Seaborn`.
- **Main Analysis:** Comparison of annual trip volumes and hourly distributions before and during the pandemic.

### 2. Heuristic Recommender System
**Directory:** `P2/`
Development of a movie recommendation engine using the MovieLens 1M dataset.
- **Methodology:** Implementation of user-based collaborative filtering.
- **Similarity Metrics:** Usage of similarity matrices to find comparable user tastes.
- **Evaluation:** Performance assessment using metrics like **MAPE** (Mean Absolute Percentage Error).

### 3. Open n-Grams and Naive Bayes Classifier
**Directory:** `P3/`
A project focused on text representation and language identification.
- **Techniques:** Feature extraction using **Open n-Grams** (non-contiguous character sequences), demonstrating robustness against spelling errors.
- **Modeling:** Implementation of a **Multinomial Naive Bayes** classifier from scratch to detect languages.
- **Performance:** Achievement of over 95% accuracy in language classification tasks.

## Requirements
The project is developed in **Python 3.13** using the following libraries:
- `Pandas` & `NumPy` for data manipulation.
- `Matplotlib` & `Seaborn` for visualization.
- `PyArrow` for efficient Parquet file handling.
- `Scikit-learn` for machine learning utilities.
- `Tqdm` for progress tracking.

## Setup
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install pandas numpy tqdm pyarrow matplotlib seaborn scikit-learn
3. Run the Jupyter Notebooks (.ipynb) located in each practice folder.
