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

## Repository Structure

/tabular/
  - DataProblems_Project1_Final__3_.ipynb (Primary Analysis)
  - Fish_Biomass_Investigation.ipynb (Investigation Analysis)
  - fish_data.csv (Dataset)

## How to Run the Code

1. Install required libraries:
   ```
   pip install pandas scikit-learn xgboost numpy matplotlib seaborn
   ```

2. Place fish_data.csv in the /tabular/ directory

3. Run the primary notebook:
   ```
   jupyter notebook DataProblems_Project1_Final__3_.ipynb
   ```
   This notebook will:
   - Load and explore the data
   - Perform EDA (Exploratory Data Analysis)
   - Perform train-test split (80/20)
   - Preprocess features (StandardScaler, OneHotEncoder)
   - Train 8 model variants (4 models, with/without PCA)
   - Compare performance on held-out test set
   - Conduct 5-fold cross-validation on training data
   - Analyze feature importance
   - Test feature subsets to identify key predictors
   - Demonstrate hyperparameter tuning and its effects

4. Run the investigation notebook (optional):
   ```
   jupyter notebook Fish_Biomass_Investigation.ipynb
   ```
   This notebook explores model performance when all biomass formula components are removed, showing the impact of data leakage on predictive power.

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

## Two Analysis Approaches

To understand the impact of data leakage, we conducted two separate analyses:

### Analysis 1: Primary Analysis (DataProblems_Project1_Final__3_.ipynb)

This notebook kept biological features that are components of the biomass formula:
- Kept: size, quantity, species (with their a_ord and b_pen coefficients)
- Removed: a_ord and b_pen only

Results:
- Best Model: XGBoost
- Test R²: 0.7613
- Test MAE: 0.0066 kg
- Test RMSE: 0.0359 kg

The model achieved strong performance because it could use the formula components to predict biomass.

### Analysis 2: Investigation Analysis (Fish_Biomass_Investigation.ipynb)

This notebook investigated whether models have predictive power when ALL formula components are removed:
- Removed: a_ord, b_pen, size, quantity
- Kept: only reef characteristics and species taxonomy

Results (Best Model: Linear Regression):
- Test R²: 0.2469
- Test MAE: 0.0244 kg
- Test RMSE: 0.0638 kg

With only environmental features, model performance dropped significantly.

### Comparison: Impact of Data Leakage

| Feature Set | R² Score | Interpretation |
|------------|----------|-----------------|
| All data with leakage | 0.95+ | Model reconstructs biomass formula |
| Keep size + quantity + species | 0.7613 | Strong predictions from formula components |
| Only reef characteristics | 0.2469 | Weak predictions from environment alone |

This progression clearly shows that the biomass formula components (size, quantity, species) drive most of the predictive power. Environmental location features alone provide minimal predictive signal.

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
