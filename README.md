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

lightgbm

🚀 How to Use
Clone the repository to your local machine.

Ensure you have Python and the required libraries installed. You can install them using pip:

```bash

pip install pandas numpy matplotlib seaborn jupyter lightgbm
```
Place the house_prices.csv file inside a Data/ directory in the same folder as the notebook.

Run the Jupyter Notebook 01_Data_Cleaning_and_EDA.ipynb to execute the data cleaning process and generate the output files.

## Model Training
### Model Selection
After thorough data preparation, a LightGBM Regressor was chosen for this task. This model is highly efficient and particularly well-suited for this dataset because of its excellent built-in capabilities for handling categorical features. This allowed us to use all 81 unique locations without needing to group them, preserving the granular detail in the data.

### Training Process
The modeling workflow was executed as follows:

- Data Preparation: The cleaned dataset was loaded, and all categorical columns (location, Transaction, Furnishing, etc.) were converted to the category data type, as recommended for LightGBM.

- Data Splitting: The dataset was split into features (X) and the log-transformed target variable (y). A standard 80/20 split was used to create training and testing sets.

- Model Training: An instance of lgb.LGBMRegressor was trained on the training data (X_train, y_train). The model training was straightforward, as LightGBM automatically handles the categorical features without requiring manual one-hot encoding.

## Model Performance & Limitations
The model's performance was evaluated on the test set after converting the log-transformed predictions back to their original price scale using np.expm1().

- Key Metrics:
    R-squared (R²): 0.80

- Root Mean Squared Error (RMSE): ₹63,35,519.30

## Analysis of Results
An R² score of 0.80 indicates that the model successfully explains 80% of the variance in house prices, which is a strong result for a complex real estate market.

However, the RMSE of approximately 63 Lakhs seemed high. A deeper analysis revealed that this was not a flaw in the model's overall logic but was caused by its performance on a few outlier properties.

As shown in the analysis, the model predicts prices for typical properties (e.g., in the 30 Lakh to 1.6 Crore range) with high accuracy. The large errors that inflate the RMSE are concentrated on a handful of ultra-luxury properties valued at over 25 Crores.

## Conclusion
The model is robust and reliable for the vast majority of properties in the Jaipur market. Its primary limitation is predicting prices for the extreme high end of the market, likely due to the scarcity of such data points in the training set.

## 🚀 Interactive Web Application (Streamlit)
To provide a user-friendly way to interact with the model, an interactive web application was built using Streamlit.

Key Features:
    - Real-time Predictions: Get instant house price estimates based on your inputs.

    - User-Friendly Interface: A clean layout with a sidebar for inputting all necessary property features.

    - Dynamic Dropdowns: Categorical inputs like Location, Furnishing, and Ownership are dynamically populated from the dataset, ensuring valid selections.

    - Interactive Controls: Sliders and number inputs for features like Area, BHK, and Bathrooms.

The app loads the trained LightGBM_model.pkl and provides an intuitive interface for anyone to get a price estimate without needing to run any code.

## 🛠️ Libraries Used
- pandas
- numpy
- matplotlib
- seaborn
- re (Regular Expressions)
- lightgbm
- streamlit
- joblib

## Exporting the Model
The final trained LightGBM model was exported as LightGBM_model.pkl using joblib for future use in a web application or API.

## ⚙️ How to Use
1. Data Cleaning and Model Training
Clone the repository to your local machine.

Place the house_prices.csv file inside a Data/ directory.

Run the Jupyter Notebook 01_Data_Cleaning_and_EDA.ipynb to execute the data cleaning process and train the model. This will generate the necessary CleanedData.csv and LightGBM_model.pkl files.

2. Running the Streamlit Web App
Ensure you have Python and the required libraries installed.

```Bash

pip install pandas numpy lightgbm scikit-learn streamlit joblib
```
Make sure the following files are in your project directory:

- app.py (the Streamlit script)

- LightGBM_model.pkl

- Data/CleanedData.csv

Open your terminal, navigate to the project directory, and run the following command:

```Bash

streamlit run app.py
```
Your web browser will automatically open a new tab with the Jaipur House Price Predictor application running locally.