# DS303_Project
Early Detection of Parkinson's Disease

How to run: 
1. Download the "projectcode2.ipynb" file.
2. Download the  "DS303 Project - ABSOLUTELY_FINAL_DATASET.csv" and "DS303 Project - LABEL.csv" into the same directory as the code.
3. Run the Jupyter notebook file.

**Introduction:**

Parkinson's disease is a neurodegenerative disorder that affects the central nervous system, causing movement-related symptoms such as tremors, rigidity, and impaired balance. Early detection of Parkinson's disease is crucial for timely intervention and effective management. In this project, we aim to replicate the results of the research paper titled "Early Detection of Parkinson’s Disease Using Deep Learning and Machine Learning" by implementing the provided code.

**Data:**

The project utilizes two input datasets: "DS303 Project - ABSOLUTELY_FINAL_DATASET.csv" and "DS303 Project - LABEL.csv". The first dataset contains various features related to Parkinson's disease, such as RBDSQ, UPSIT, asyn, abeta, pTau, tTau, and others. The second dataset provides the corresponding labels indicating the presence or absence of Parkinson's disease.

**Methodology:**

The project employs a neural network model for the classification task of predicting Parkinson's disease. The code implements the MLPClassifier from scikit-learn library for training and testing the neural network. The key steps involved are as follows:
- Data Preprocessing:
    - The relevant features are selected from the dataset, including RBDSQ, UPSIT, asyn, abeta, pTau, tTau, pTau/abeta, pTau/tTau, tTau/abeta, caudate_l, caudate_r, putamen_l, and putamen_r.
    - The data is then standardized using the StandardScaler to ensure all features have zero mean and unit variance.
- Model Training:
    - The dataset is split into training and testing sets using a 70:30 ratio.
    - The MLPClassifier is initialized with specific parameters such as hidden_layer_sizes, activation function, solver, regularization alpha, batch size, learning rate, and maximum iterations.
    - The model is trained on the training data using the fit() function.
- Model Evaluation:
    - The trained model is used to predict the target labels for the testing data.
    - Various performance metrics are computed, including accuracy, F1 score, and precision score.
    - Additionally, a confusion matrix is generated to assess the classification results.
- Hyperparameter Tuning:
    - The code incorporates a hyperparameter tuning loop that iterates 100 times.
    - Random values are generated for the parameters alpha, learning rate init, and power_t.
    - The model is trained and evaluated for each set of hyperparameters, and the best-performing set is recorded.

**Results:**

Based on the provided code, the results obtained from running the neural network model for the given Parkinson's disease dataset are as follows:
- Best Accuracy: 0.9841269841269841
- F1 Score: 0.9598214285714286
- Precision Score: 0.9911504424778761

Please note that the specific values of accuracy, F1 score, and precision score will depend on the dataset used and the randomization of hyperparameters.

**Conclusion:**

In this project, we replicated the research paper's approach for early detection of Parkinson's disease using deep learning and machine learning techniques. By training a neural network model on the provided dataset, we obtained results for accuracy, F1 score, and precision score. These metrics serve as indicators of the model's performance in predicting Parkinson's disease.

