# Linear-classification-algorithms
Implementation of linear classification algorithms on banknote authentication dataset. The models implemented include Fisher's linear discriminant, Probabilistic generative model and logistic regression.

Goal : Implement different linear models for binary classification. 

Dataset: uci repository’s ’banknote authentication Data Set’
https://archive.ics.uci.edu/ml/datasets/banknote+authentication

-------------------------------------------------------------------------------------------
I. Fisher’s linear discriminant

confusion matrix :

threshold: 0.966963
tp : 184
fp : 3
tn : 224
fn : 1

correct predictions: 408

incorrect predictions: 4

precision: 0.983957

recall: 0.994595

accuracy: 99.0291%


Confusion Matrix:

n = 412     Predicted: NO       Predicted: YES

Actual: NO    TN =224                FP =3 

Actual: YES   FN = 1                 TP = 184
-------------------------------------------------------------------------------------------------

II. Probabilistic generative model

tp : 185
fp : 11
tn : 216
fn : 0

correct predictions: 401

incorrect predictions: 11

precision: 0.943878

recall: 1

accuracy: 97.3301%


Confusion Matrix:

n = 412     Predicted: NO       Predicted: YES

Actual: NO    TN =216                FP =11 

Actual: YES   FN = 0                 TP = 185

-----------------------------------------------------------------------------------------------------------------------
III. Logistic Regression Model 

true_positive: 185
true_negative: 224
false_positive: 3
false_negative: 0

correctly predicted: 409

incorrectly predicted: 3

precision : 0.984043

recall : 1

accuracy: 99.2718%


Confusion Matrix:

n = 412     Predicted: NO       Predicted: YES

Actual: NO    TN =224                FP =3 

Actual: YES   FN = 0                 TP = 185
--------------------------------------------------------------------------------------------------------------------
