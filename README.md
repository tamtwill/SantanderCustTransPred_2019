# SantanderCustTransPred_2019
# Work in process
## 2019 - Kaggle competition to predict customer transactions for Santander
https://www.kaggle.com/c/santander-customer-transaction-prediction

###Random Notes to self
+Dataset too big to check into Github, even in the zipped form
+The competition is couched as a classification problem, but scoring is based on AUC, so returning probability, rather than a class 
+Based on the model evaluation script, gradient boosted trees seemed the most promising, so I focused on XGB and am trying LGBM for the first time
+Most likely, the dataset is too small to make ANN a good choice, but I may try one anyway
+PCA didn't show any obvious "elbows" to use it feature reduction, and tests using the smaller datasets (fewer features) scored poorly, so didn't continue to invest in this after the first few experiments
