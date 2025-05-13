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


