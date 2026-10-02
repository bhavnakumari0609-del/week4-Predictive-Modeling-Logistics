Week 4 – Predictive Modeling and Optimization in Logistics Systems

Project Overview

This project applies predictive modeling and optimization techniques to a simulated last-mile logistics delivery system. The main objective is to predict delivery time and use the prediction results to support better logistics planning and resource allocation.

The project uses Python and a synthetic dataset containing 5,000 logistics delivery records.

Objectives

- Predict delivery time using machine learning.
- Compare multiple regression models.
- Evaluate models using RMSE, MAE, and R².
- Validate model performance using 5-fold cross-validation.
- Identify the most important factors affecting delivery time.
- Apply optimization for driver and route assignment.
- Generate business insights and practical recommendations.

Dataset

The dataset contains 5,000 simulated delivery records.

Features

- "distance_km" – Delivery distance in kilometers
- "traffic_index" – Traffic severity index
- "package_weight_kg" – Package weight in kilograms
- "driver_experience_yr" – Driver experience in years
- "weather_severity" – Weather severity level
- "is_peak_hour" – Indicates whether delivery occurs during peak hours
- "num_stops_enroute" – Number of stops during the route

Target Variable

- "delivery_time_min" – Predicted delivery time in minutes

Machine Learning Models

Three regression models were implemented:

1. Ridge Regression
2. Random Forest Regressor
3. Gradient Boosting Regressor

Gradient Boosting was selected based on the lowest test-set RMSE among the implemented models.

Model Evaluation

Test Set Results

Metric| Gradient Boosting
RMSE| 4.919 minutes
MAE| 3.881 minutes
R²| 0.9863

5-Fold Cross-Validation

Model| Mean CV RMSE
Ridge Regression| 9.508 minutes
Random Forest| 5.987 minutes
Gradient Boosting| 4.864 minutes

Feature Importance

The most influential features in the Gradient Boosting model were:

- Distance: 75.83%
- Traffic Index: 19.12%
- Number of Stops: 2.63%
- Package Weight: 0.96%
- Peak Hour: 0.77%
- Weather Severity: 0.68%
- Driver Experience: approximately 0.00%

Distance and traffic were the strongest predictive factors in the simulated dataset.

Optimization

A route-assignment optimization model was implemented using linear programming.

The optimization objective was to minimize total predicted delivery time while ensuring:

- Every route is assigned.
- Each driver handles at most two routes.
- Driver-route assignments remain within the defined constraints.

The optimized assignment is saved in:

"optimized_driver_assignment.csv"

Project Files

create_dataset.py
logistics_delivery_data.csv
week4_predictive_model.py
model_comparison_results.csv
feature_importance.csv
week4_optimization.py
optimized_driver_assignment.csv
actual_vs_predicted.png
model_performance_comparison.png
feature_importance.png
distance_vs_delivery_time.png
traffic_vs_delivery_time.png
README.md

Visualizations

The project includes visualizations for:

- Actual vs Predicted Delivery Time
- Model Performance Comparison
- Feature Importance
- Distance vs Delivery Time
- Traffic Index vs Delivery Time

Business Insights

The analysis indicates that delivery distance and traffic conditions are major factors affecting delivery time in the simulated logistics environment.

The predictive model can support:

- Dynamic delivery-time estimation
- Route planning
- Driver workload balancing
- Delivery scheduling
- Operational resource allocation

Optimization Recommendations

1. Use predicted delivery times to improve route planning.
2. Allocate drivers based on route requirements and workload.
3. Consider traffic conditions when scheduling deliveries.
4. Use dynamic time buffers for routes with higher predicted delays.
5. Regularly retrain the model when new real-world delivery data becomes available.

Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- SciPy
- VS Code
- GitHub

Conclusion

This project demonstrates how predictive modeling and optimization can be combined to address logistics planning problems. Gradient Boosting provided strong predictive performance on the simulated dataset, while the optimization model demonstrated how predicted delivery times can be incorporated into driver-route assignment decisions.

The project provides a practical foundation for data-driven logistics decision-making and can be extended with real-world delivery, GPS, traffic, and operational cost data.