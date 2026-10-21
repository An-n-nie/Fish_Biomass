# Fish Biomass Prediction Project

## Course Information

Course: Data Problems (DATA 4380) at UTA

Group Members:
1. Isabela Alaniz
2. Tsion Sehene
3. Clarissa Terrones
4. Annie Tran
5. Manya Vikal

## Overview

This project predicts fish biomass (estimated mass/weight) from biological and reef characteristics using machine learning. The goal is to answer: How accurately can we predict fish biomass, and which factors matter most?

## Problem Statement

This is a supervised regression problem. We use fish biological data (species, size, quantity) and reef characteristics (location, habitat, protection status) to predict biomass values.

## Dataset

Source: Alacranes Reef and Bajos del Norte Reef Ecological Assessment
- Kaggle: https://www.kaggle.com/datasets/jocelyndumlao/reef-ecological-assessment-del-norte
- Original Data Source: https://data.mendeley.com/datasets/v6ksv75gk3/1
- Researcher: Fabio Favoretto (Universidad Autonoma de Baja California Sur)

Data Collection:
- Field surveys using SCUBA diving techniques
- Location: Del Norte County (Alacranes Reef and Bajos del Norte Reef)
- File used: fish_data.csv

Dataset Details:
- Total observations: 2,431 fish records
- Features after cleaning: 17 (removed constant features and leaking variables)
- Target variable: biomass (kg)

Key features include:
- Biological: species, size (cm), quantity, trophic level
- Reef characteristics: region, island, reef name, protection status, latitude, longitude, transect
- Taxonomic: family, taxonomic group, functional group

## Data Preparation

1. Removed constant features (year, month, habitat, phylum, taxa1, taxa2) with only 1 unique value
2. Removed data leakage: dropped a_ord and b_pen (parameters used in biomass formula)
3. Train-test split: 80% training (1,944 samples) and 20% testing (487 samples)
4. Preprocessing:
   - StandardScaler for numerical features
   - OneHotEncoder for categorical features
   - Fit preprocessing on training data only to prevent data leakage

## Models Tested

Eight model variants were trained and evaluated on the same test set:

1. Linear Regression (with and without PCA)
2. Random Forest Regressor (with and without PCA)
3. Gradient Boosting Regressor (with and without PCA)
4. XGBoost Regressor (with and without PCA)

Models were also evaluated using 5-fold cross-validation on training data.

## Key Findings

### Best Model Performance

XGBoost without PCA achieved the best results:
- R² Score: 0.7613 (explains 76% of biomass variation)
- MAE: 0.0066 kg (80% better than baseline)
- RMSE: 0.0359 kg (51% better than baseline)

### Cross-Validation Results

5-fold cross-validation mean scores on training data:
- XGBoost: 0.8082
- Gradient Boosting: 0.7776
- Random Forest: 0.6905

### Feature Importance

Top 5 most important features for XGBoost:
1. Family (43%)
2. Quantity (18%)
3. Taxa group (11%)
4. Species (10%)
5. Size (8%)

Location features (reef, latitude, longitude) had near-zero importance.

### Hyperparameter Tuning

GridSearchCV found optimal parameters (n_estimators=250, max_depth=2) that achieved CV R²=0.865 but test R²=0.702. This demonstrates overfitting to training data patterns that don't generalize. Original settings (100 estimators, default depth) performed better on the test set.

### PCA Results

PCA reduced features from 190 to 34 components while maintaining 95% explained variance. However, PCA reduced performance of all tree-based models:
- XGBoost: 0.7613 > 0.2894
- Gradient Boosting: 0.7395 > 0.4958
- Random Forest: 0.7385 > 0.6777

Tree-based models benefit from the original feature structure.

### Feature Subset Test

With only 3 features (species + size + quantity):
- R²: 0.7404 (97% of full model's power)

With only 2 features (size + quantity):
- R²: 0.6198 (significant drop)

Reef/location features alone:
- R²: -0.0424 (worse than baseline)

## How to Run the Code

1. Install required libraries:
   ```
   pip install pandas scikit-learn xgboost numpy matplotlib
   ```

2. Place fish_data.csv in the working directory

3. Run the notebook:
   ```
   jupyter notebook DataProblems_Project1_Final.ipynb
   ```

4. The notebook will:
   - Load and explore the data
   - Perform train-test split
   - Preprocess features
   - Train all 8 models
   - Compare performance on test set
   - Conduct 5-fold cross-validation
   - Analyze feature importance
   - Test feature subsets

## Model Evaluation Metrics

- R² Score: Proportion of variance explained (0-1, higher is better)
- MAE (Mean Absolute Error): Average absolute prediction error in kg
- RMSE (Root Mean Squared Error): Square root of average squared error in kg

Baseline: Dummy regressor predicting mean biomass for all samples (R²=0, MAE=0.0334)

## Limitations

1. Biomass is highly right-skewed, with many small values and few very large ones
2. Model performance varied across 5-fold CV folds, suggesting sensitivity to training subset
3. Feature importance shows correlation, not causation
4. After one-hot encoding, categorical variables (especially species with 110 categories) increase feature dimensionality to 190

## Why Overfitting Occurred

The high R² values (0.95+) in early analysis were due to data leakage. The biomass formula is:

biomass = a_ord * (size^b_pen) * quantity / (1000 * 60 * 10)

We kept the features that directly appear in this formula (size, quantity, and species with their a_ord and b_pen coefficients). The model could reverse-engineer the formula rather than learning actual predictive patterns. This explains the overfitting.

When we removed a_ord and b_pen but kept size, quantity, and species, the model still achieved R²=0.74, confirming it was reconstructing the known formula.

## Conclusions

1. Fish biomass can be predicted with R²=0.7613 using biological and reef characteristics
2. Biological features (species, size, quantity) are dominant predictors because they are components of the biomass calculation formula
3. Reef location characteristics alone provide zero predictive value
4. Tree-based models (XGBoost, Gradient Boosting) substantially outperform linear models
5. PCA dimensionality reduction reduces performance for tree-based models
6. Hyperparameter tuning can hurt test performance if validated only on training data
7. Overfitting occurs when features directly used in the target calculation are included as predictors

## Future Work

- Apply log transformation to target variable
- Investigate species-specific patterns
- Compare models on data with transformed targets
