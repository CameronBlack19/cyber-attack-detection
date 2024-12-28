# Cyber Attack Detection Using Data Analytics and Machine Learning  

This repository contains resources for detecting cyber-attacks using data analytics and machine learning techniques, specifically with Support Vector Machines (SVM) and K-Nearest Neighbors (KNN).  

## Files in this Repository  

1. **`CyberAttackDetection-svm-knn(2) (1).ipynb`**  
   - Jupyter Notebook implementing SVM and KNN models for cyber-attack detection.  
   - Includes data preprocessing, model training, evaluation, and visualization of results.  

2. **`cybersecurity_attacks.csv`**  
   - Dataset containing labeled records of network activity for training and testing the models.  
   - Features include network parameters and attack labels.  

3. **`test_res.csv`**  
   - Output file with test results, including metrics (precision, recall, F1-score, support) for categories (High, Low, Medium) and overall performance (accuracy, macro avg, weighted avg).

## Features  

- **Machine Learning Models**: Implementation of SVM and KNN for robust detection of cybersecurity threats.  
- **Data-Driven Insights**: Utilizes real-world network traffic data to train and evaluate the models.  
- **Evaluation and Results**: Includes test outputs and metrics like accuracy and confusion matrix.  

## Requirements  

Ensure the following are installed before running the code:  
- Python 3.8 or higher  
- Required Python libraries:  
  ```bash
  pip install numpy pandas matplotlib scikit-learn jupyter
  ```

## How to Use

1. Clone the repository
  ```bash
  git clone https://github.com/yourusername/Cyber-Attack-Detection
  ```

2. Navigate to the project directory
  ```bash
  cd Cyber-Attack-Detection
  ```

3. Open Jupiter Notebook
  ```bash
  jupyter notebook CyberAttackDetection-svm-knn(2)\ \(1\).ipynb
  ```
4. Follow these steps in the notebook:
- Load the dataset cybersecurity_attacks.csv.
- Perform data preprocessing, including normalization and splitting the data into training and testing sets.
- Train SVM and KNN models using the provided scripts.
- Evaluate the models using metrics such as accuracy and confusion matrix.
- Review the output predictions in test_res.csv.

5. To test real-time detection, modify the notebook to accept new data inputs and run the pre-trained models.

## Results

The project demonstrates the successful detection of cyber-attacks using SVM and KNN. Key outcomes include:

- High accuracy in classifying normal and malicious network traffic.
- Insights from confusion matrices and evaluation metrics for both models.
- Predictions and test results stored in **`test_res.csv`** for analysis.

## Acknowledgements

We would like to thank the contributors of the dataset and the open-source community for providing tools and libraries that made this project possible. Special appreciation to researchers in cybersecurity and machine learning for inspiring this work.
