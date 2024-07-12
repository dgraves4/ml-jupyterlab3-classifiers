# ml-jupyterlab3-classifiers

## Project Overview

In this lab, we use the Howell data set to train models that predict the gender of a person based on their height, weight, and age. The project involves training multiple classifiers, comparing their performance, and analyzing the results.

---

## Objectives

1. **Prepare the Data**:
   - Load and preprocess the Howell data set.
   - Split the data into training and test sets.

2. **Train Multiple Models**:
   - Decision Tree
   - Support Vector Classifier (SVC)
   - Neural Network (Multi-Layer Perceptron)

3. **Evaluate Model Performance**:
   - Calculate performance metrics: Accuracy, Precision, Recall, F1 Score.
   - Compare performance across different models and feature sets.

4. **Analyze Results**:
   - Determine if the models overfit the data.
   - Identify the best feature set for training.
   - Compare model performances and propose explanations for differences.
   - Visualize support vectors and decision boundaries.

---

## Data Preparation

- **Data Source**: Howell data set.
- **Features**: Height, Weight, Age.
- **Target**: Gender (0 for Female, 1 for Male).

### Steps:

1. **Data Loading**: Load the data set into the notebook.
2. **Data Cleaning**: Handle any missing values (if applicable).
3. **Feature Engineering**: Add or transform features as needed.
4. **Train/Test Split**: Split the data into training and test sets using a stratified split to maintain the gender distribution.

---

## Models Trained

### 1. Decision Tree

- **Features Used**: Height only, Weight only, Height and Weight.
- **Implementation**: `DecisionTreeClassifier` from `sklearn.tree`.
- **Performance Metrics**:
  - **Height Only**: 
    - Training Accuracy: 89.85%
    - Test Accuracy: 81.43%
  - **Weight Only**:
    - Training Accuracy: 96.01%
    - Test Accuracy: 62.86%
  - **Height and Weight**:
    - Training Accuracy: 100.00%
    - Test Accuracy: 71.43%

### 2. Support Vector Classifier (SVC)

- **Features Used**: Height and Weight.
- **Implementation**: `SVC` from `sklearn.svm`.
- **Performance Metrics**:
  - Training Accuracy: 84.78%
  - Test Accuracy: 77.14%

### 3. Neural Network (Multi-Layer Perceptron)

- **Features Used**: Height and Weight.
- **Implementation**: `MLPClassifier` from `sklearn.neural_network`.
- **Model Parameters**: Hidden layers = (50, 25, 10), Solver = 'lbfgs'.
- **Performance Metrics**:
  - Training Accuracy: 77.17%
  - Test Accuracy: 78.57%

---

## Performance Analysis

### Overfitting Analysis

- **Decision Tree**:
  - The model using weight only showed significant overfitting with high training performance and low test performance.
  - The model using height and weight also showed overfitting with perfect training performance but lower test performance.

### Best Feature Set

- **Height Only**:
  - Demonstrated the best balance between training and test performance metrics, making it the most reliable feature set for this data.

### Model Comparison

- **Decision Tree (Height Only)**:
  - Best overall performance with balanced metrics.
- **SVC (Height and Weight)**:
  - Good generalization with consistent performance across training and test sets.
- **Neural Network (Height and Weight, (50,25,10) lbfgs)**:
  - Slightly lower test performance than Decision Tree (Height Only) but demonstrated balanced performance across training and test sets.

### Support Vectors Analysis

- **Learning from Support Vectors**:
  - Determined the decision boundary and margins.
  - Highlighted areas of uncertainty and overlap between classes.

---

## Conclusion

- **Best Performing Model**: Decision Tree using Height only, with the highest test performance and generalization.
- **Balanced Model**: Neural Network, showing good consistency between training and test sets and suitable for real-world applications involving more features.

---

## Future Work

- **Feature Expansion**:
  - Investigate the impact of adding the Age feature.
- **Hyperparameter Tuning**:
  - Optimize the hyperparameters for better performance.
- **Scaling and Normalization**:
  - Implement scaling to improve neural network convergence.

---

## How to Run the Project

1. **Environment Setup**:
   - Ensure Python and Jupyter Lab are installed.
   - Install necessary libraries using:
     ```sh
     pip install -r requirements.txt
     ```
2. **Load the Notebook**:
   - Open the Jupyter Notebook and rename it to include your name.
3. **Execute the Cells**:
   - Follow the checkpoints and execute the cells sequentially.

---

## Acknowledgments

- **Data Source**: Howell data set.
- **Tools Used**: Python, Jupyter Lab, Scikit-learn, Matplotlib.