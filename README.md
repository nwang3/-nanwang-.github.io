# nwang3.github.io
## Overview
### This project has been an exciting journey, providing opportunities for hands-on learning, problem-solving, and continuous improvement. The primary goal was to develop a web application that consolidates real estate data, predicts property prices, and presents the results in a user-friendly manner. From the initial idea to the final implementation, I've encountered various hurdles, learned new concepts, and honed my problem-solving skills.

## Progress

## 1. Data Consolidation
### I successfully implemented a feature to consolidate real estate data from multiple Excel files. The Python Flask framework facilitated the creation of a robust backend that handles file uploads, data processing, and Excel file generation. This was a crucial step toward creating a centralized dataset for analysis.

## 2. Machine Learning Model
### The incorporation of a machine learning model for price prediction was a significant achievement. The process was not east as I have encountered many different error code along the process. For example: "KeyError: "['status', 'state'] not in index" is one of the frequent errors I got, and after spending ours debugging among app.py, html, and the regression_analysis.py, I still could not resolve the problem. Professor Li's office hour has been extremely hepful in guiding me to the right direction and sitting down with me and helping me finding out where the error was and how to debug. 
### Leveraging the scikit-learn library, I trained a linear regression model using relevant features such as the number of bedrooms and bathrooms. However, at first, my model included all attributes including house_size, status, city, state, zip code, etc. I had to pivot to using only number of bedrooms and bathrooms to predict the average housing price for the entire USA instead of user's input for location. 

## 3. User Interface
### The user interface was designed to be user-friendly and visually appealing. The front-end, powered by HTML, CSS, and Flask templates, offers a seamless experience for users uploading files and receiving predictions. The interface is responsive, making it accessible across different devices.

## Problem Solving

## 1. Feature Selection

### I encountered challenges related to feature selection when users input data for predictions. Through careful consideration, I modified the preprocessor to handle missing features. This ensured a smooth user experience while maintaining the integrity of the machine learning model.

## 2. Preprocessor Expectations

### Addressing issues related to the preprocessor expecting specific columns, I refined the data processing workflow to align with the model's expectations. I adjusted the input data to include only the necessary features, preventing unnecessary errors and enhancing the application's robustness.

## 3. Visual Styling

### Enhancing the visual appeal of the user interface required attention to styling details. The implementation of CSS styles and the selection of appropriate color schemes contributed to a more professional and engaging appearance.

## Learning Curve

### This project has been a valuable learning experience, allowing me to:

### - Deepen Machine Learning Skills: I strengthened my understanding of machine learning concepts, particularly in the context of regression models and feature preprocessing.
  
### - Improve Web Development Proficiency: Working with Flask, HTML, and CSS has enhanced my web development skills, enabling me to build a functional and aesthetically pleasing application.

### - Refine Problem-Solving Skills: The various challenges faced during feature handling and preprocessor expectations provided opportunities to hone my problem-solving abilities.

## Next Steps
### Moving forward, I plan to:
### - Explore Additional Features: Consider incorporating more features for prediction, such as location-based data or property amenities, to enhance the accuracy of predictions.
### Implement User Authentication: Introduce user authentication to provide a personalized experience and secure user data.

### - Optimize Model Performance: Continuously refine the machine learning model by exploring alternative algorithms and fine-tuning hyperparameters.

## Attribution
### The dataset realtor-data.csv is from Kaggle. I used this dataset for the machine learning regression model for housing price prediction. Here is the link: https://www.kaggle.com/datasets/ahmedshahriarsakib/usa-real-estate-dataset. It contains Real Estate listings in the US broken by State and zip code.The dataset has 1 CSV file with 10 columns: realtor-data.csv (1.1 Million+ entries)
#### - status 
#### - bed (# of beds)
#### - bath (# of bathrooms)
#### - acre_lot (Property / Land size in acres)
#### - city (city name)
#### - state (state name)
#### - zip_code
#### - postal code of the area 
#### - house_size (house area/size/living space in square feet)
#### - prev_sold_date (Previously sold date)#### - price (Housing price, it is either the current listing price or recently sold price if the house is sold recently)
#### - NB: acre_lot means the total land area, and house_size denotes the living space/building area

## Conclusion

### This project has been a rewarding experience, blending machine learning and web development to create a practical application. The process of overcoming challenges, refining the application, and continuous learning has contributed to my growth as a developer.