# Boston-House-Prediction# **Boston Housing Price Prediction**

## **Introduction**
This project aims to predict housing prices using machine learning models. The dataset used is the California Housing dataset, which contains various socio-economic and geographical features. The project implements **Linear Regression** and **Ridge Regression** models to analyze and predict house prices.

## **Project Overview**
- **Objective**: To build a predictive model for estimating house prices based on available features.
- **Dataset**: California Housing dataset from `sklearn.datasets`.
- **Models Used**:
  - Linear Regression
  - Ridge Regression
- **Evaluation Metrics**:
  - Mean Squared Error (MSE)
  - R² Score

## **Technologies Used**
- **Python**
- **Libraries**:
  - `scikit-learn` (Machine Learning)
  - `pandas` (Data Handling)
  - `numpy` (Numerical Computation)
  - `matplotlib` (Data Visualization)

## **Installation & Setup**
1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/Boston-Housing-Prediction.git
   cd Boston-Housing-Prediction
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the script:
   ```sh
   python boston.py
   ```

## **Methodology**
1. **Data Preprocessing**:
   - Load the dataset using `fetch_california_housing()`.
   - Split data into training and testing sets.
2. **Model Training**:
   - Train **Linear Regression** and **Ridge Regression** models.
3. **Prediction & Evaluation**:
   - Evaluate models using MSE and R² score.
   - Visualize actual vs. predicted prices.

## **Results**
| Model               | MSE  | R² Score |
|--------------------|------|----------|
| Linear Regression  | 0.56 | 0.58     |
| Ridge Regression   | 0.56 | 0.58     |

## **Project Outcome**
- Both models performed similarly, with Ridge Regression slightly reducing overfitting.
- Scatter plots were generated for actual vs. predicted prices.
- The project provides insights into housing price predictions and can be extended using advanced models.

## **Future Enhancements**
- Implement advanced models like Decision Trees, Random Forest, or Gradient Boosting.
- Fine-tune hyperparameters to improve accuracy.
- Integrate additional real-world features for better predictions.

## **License**
This project is open-source and available under the MIT License.

---

### 📌 **Contributions & Feedback**
Feel free to contribute by raising issues or submitting pull requests. If you have any suggestions, reach out via GitHub!

