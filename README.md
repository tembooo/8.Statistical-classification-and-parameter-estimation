# 8.Statistical-classification-and-parameter-estimation
This project explores key concepts in statistical classification and parameter estimation using Gaussian models.
📊 Bayesian and ML Estimation for Classification
This project explores key concepts in statistical classification and parameter estimation using Gaussian models. It demonstrates the derivation and application of discriminant functions for binary classification in a 2D feature space using known means, covariance matrices, and class priors. Additionally, it includes maximum likelihood (ML) estimation for univariate normal distributions under both known and unknown mean scenarios. Ideal for students and researchers in pattern recognition and machine learning, the project offers hands-on insight into how theoretical models translate into precise decision boundaries and distribution estimations.
![image](https://github.com/user-attachments/assets/3d94f85e-54ad-4692-94a9-35dc95ec931b)

Let us assume a binary classification problem in a 2D feature space. Bivariate (two-dimensional) normal distribution is used to model each class ω₁, ω₂, and the relevant parameters estimated from the data are as follows:

μ₁ = [−1; 1], μ₂ = [3; −1]

Σᵢ = Σ = σ²I, σ² = 2,

P(ω₁) = 9/10, P(ω₂) = 1/10,
where I is the identity matrix.

For such covariance, we know that the discriminant functions can be defined as follows:

gᵢ(x) = wᵢᵀx + wᵢ₀

wᵢ = Σ⁻¹μᵢ                      where ωᵢ is a class

wᵢ₀ = -½ μᵢᵀΣ⁻¹μᵢ + ln P(ωᵢ)

Determine the decision surface for the given classification problem. The decision surface intersects the line going through the mean of class ω₁ and the mean of class ω₂ at the point:

(1.88, -0.44)

Let us assume the univariate normal distribution as the model for the data. Determine the relevant distribution parameters for the given data by using maximum likelihood (ML) estimation as needed and give the parameter values using two decimals.

The mean is unknown. As the result of (unbiased) estimation, the distribution parameter values are as follows:
[2.33, 8.76]

The mean is known to be 1. As the result of (biased) estimation, the distribution parameter values are as follows:
[1.00, 10.09]

Additional files: CSV, MAT

```python
import pandas as pd
import numpy as np
# Load the data from the CSV file
file_path = 't102.csv'
data = pd.read_csv(file_path)
# Convert the column names to float values (assuming data is stored in column headers)
data_values = data.columns.astype(float)
# Calculate the number of observations
N = len(data_values)
#Unbiased Estimation (Unknown Mean) ----
sum_values = sum(data_values)
mean_unbiased = sum_values / N
# Calculate the unbiased variance manually (ddof = 1)
sum_squared_diff = sum((x - mean_unbiased) ** 2 for x in data_values)
variance_unbiased = sum_squared_diff / (N - 1)
case_1_result = [[round(mean_unbiased, 2), round(variance_unbiased, 2)]]
#Biased Estimation (Known Mean = 1) ----
known_mean = 1.0
sum_squared_diff_known_mean = sum((x - known_mean) ** 2 for x in data_values)
biased_variance = sum_squared_diff_known_mean / N
case_2_result = [[known_mean, round(biased_variance, 2)]]
print("Case 1 (Unknown Mean, Unbiased Estimation):", case_1_result)
print("Case 2 (Known Mean = 1, Biased Estimation):", case_2_result)




```

