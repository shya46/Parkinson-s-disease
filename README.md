Parkinson's Disease Detection Using Speech Analysis

Overview : 

This project aims to detect Parkinson’s disease using machine learning and speech analysis. By analyzing vocal features such as MFCCs, Jitter, Shimmer, and HNR, the system can identify speech impairments commonly associated with Parkinson’s. A Random Forest Classifier is trained on extracted features to classify whether a person has Parkinson’s or not.

Features : 

Voice-Based Detection – No invasive tests required, just a simple speech recording.

Machine Learning (Random Forest Classifier) – Trained to classify Parkinson’s presence.

Feature Extraction – Uses MFCCs, Jitter, Shimmer, and HNR to analyze voice irregularities.

Automated Workflow – Records, extracts features, trains a model, and predicts in real-time.

Scalable & Cost-Effective – Can be integrated into mobile apps or healthcare devices.

Installation & Requirements : 

1. Install Dependencies

Make sure you have Python installed, then install the required libraries:

pip install numpy pandas librosa sounddevice soundfile scikit-learn seaborn matplotlib scipy

2. Clone the Repository

git clone https://github.com/shya46/Parkinson-s-disease.git
cd Parkinson-s-disease

Usage :

1. Run the Script

python parkinsons_detection.py

2. Steps Performed

Records a 5-second audio sample

Extracts voice features

Saves extracted features into features.csv

Prepares data and trains a Random Forest model

Predicts Parkinson’s disease status and evaluates model performance

Displays feature importance visualization

How It Works : 

Recording Audio: Captures speech using sounddevice and saves it as a .wav file.

Feature Extraction: Extracts MFCCs, Jitter, Shimmer, and HNR to analyze voice characteristics.

Data Processing: Loads previous data from features.csv, preprocesses it, and scales the features.

Model Training & Prediction: Trains a Random Forest Classifier and evaluates it on test data.

Feature Importance Analysis: Identifies the most relevant vocal features contributing to the prediction.

Expected Output :

Model Accuracy (~80-90%)

Mean Absolute Error (MAE)

Root Mean Squared Error (RMSE)

Top 5 Most Important Features

Feature Importance Graph

Future Improvements :

Deep Learning Models (CNNs, RNNs) for better accuracy.

Mobile App Integration for remote monitoring.

Multilingual Support to analyze speech in different languages.

Integration with Smart Healthcare Devices for real-time diagnosis.

License : 

This project is open-source under the MIT License.

Contributing : 

Pull requests are welcome! If you find any issues, feel free to open an issue or contribute by improving the model.

Collaborators:

Shriya Jadhav - https://github.com/shya46

Arnav Jain - https://github.com/ARJ1510

Contact :

For questions or collaboration, reach out to shriyajadhav46@gmail.com, arnav151003@gmail.com or open an issue in the GitHub repository.

Early detection of Parkinson’s can help improve treatment outcomes. Let’s make AI-powered healthcare accessible to everyone!
