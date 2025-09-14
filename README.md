# House Price Prediction 

## Data Cleaning and Exploratory Data Analysis
This repository contains the initial data cleaning, preprocessing, and exploratory data analysis (EDA) for a house prices dataset. The primary goal of this Jupyter notebook (01_Data_Cleaning_and_EDA.ipynb) is to prepare the raw dataset for machine learning modeling by handling missing values, correcting data types, engineering new features, and encoding categorical variables.

## 📝 Project Overview
The project aims to build a model to predict house prices. This notebook covers the foundational step of data preparation. The raw dataset contains inconsistencies, missing values, and columns that require significant transformation before they can be used for training a predictive model.

## 💾 Dataset
The dataset used is Data/house_prices.csv. It contains various features describing residential properties, such as location, size, number of rooms, price, and other amenities.

##  🧹 Data Cleaning and Preprocessing Pipeline
The notebook follows a systematic approach to clean and transform the data. Here are the key steps undertaken:

### Initial Setup & Data Loading:

- Imported necessary libraries: pandas, numpy, matplotlib, seaborn, and re.

- Loaded the house_prices.csv dataset into a pandas DataFrame.

### Column Removal:

Dropped columns with a high percentage of null values or redundant information. The following columns were removed:
Index, Society, Car Parking, Status, Super Area, Dimensions, Plot Area, overlooking, Title, Description, Price (in rupees).

### Feature Engineering & Extraction:

- BHK: Extracted the number of bedrooms (BHK) from the Title column using regular expressions and created a new BHK column.

- Area: Created a new Area column from Carpet Area by:

- Standardizing different units (sqft, sqm, sqyrd) into a consistent sqft format.

- The original Carpet Area column was dropped after this step.

### Data Transformation and Cleaning:

- Amount (Target Variable):
    Converted the Amount(in rupees) column from a string format (e.g., "42 Lac", "1.40 Cr") to a numerical integer format representing the price in INR.

- Handled extreme outliers by filtering out properties with prices above 50 Cr.

- Applied a log transformation (np.log1p) to normalize the distribution and reduce its high skewness (from ~270 to ~0.27).

- Floor:
    - Extracted the specific floor number from string values like "10 out of 11".

    - Handled special text values like "Ground" (converted to 0), "Upper" basement (converted to -1), and "Lower"   basement (converted to -2).

    - Converted the column to a nullable integer type (Int64).

- Bathroom & Balcony:

    - Cleaned the columns to handle non-numeric entries (e.g., ">10").

    - Handled outliers by capping the maximum values at the 99.9th percentile to reduce skewness.

- Handling Missing Values (Imputation):

    Missing values for the following columns were imputed using the mode (most frequent value):

        - Balcony

        - BHK

        - Floor

        - Bathroom

        - Transaction

        - facing

        - Furnishing

        - Ownership

For the Area column, nulls were imputed using the median area grouped by the number of BHK. Any remaining nulls were filled with the overall median area.

## Categorical Data Encoding:

Categorical columns with multiple text-based classes were identified: location, Transaction, Furnishing, facing, and Ownership.

One-Hot Encoding was applied to these columns using pandas.get_dummies() to convert them into a numerical format suitable for machine learning algorithms.

## 📤 Final Output
The cleaning process generates two new CSV files:

CleanedData.csv: Contains the cleaned data before one-hot encoding. This is useful for analyses where categorical features are preferred in their original form.

CleanedData_OHE.csv: Contains the final, fully preprocessed dataset with one-hot encoded categorical variables, ready for model training.

## 🛠️ Libraries Used
pandas

numpy

matplotlib

seaborn

re (Regular Expressions)

🚀 How to Use
Clone the repository to your local machine.

Ensure you have Python and the required libraries installed. You can install them using pip:

```bash

pip install pandas numpy matplotlib seaborn jupyter
```
Place the house_prices.csv file inside a Data/ directory in the same folder as the notebook.

Run the Jupyter Notebook 01_Data_Cleaning_and_EDA.ipynb to execute the data cleaning process and generate the output files.