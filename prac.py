import pandas

file =  pandas.read_csv("/home/tarun/Downloads/final.csv")

# Here null values are filled with median
file["TOTCHOL"] = file["TOTCHOL"].fillna(file["TOTCHOL"].median())
file["SYSBP"] = file["SYSBP"].fillna(file["SYSBP"].median())
file["DIABP"] = file["DIABP"].fillna(file["DIABP"].median())
file["BMI"] = file["BMI"].fillna(file["BMI"].median())
 #file.loc[file["SEX"]=="M","SEX"]= 0 If we have categorical column we need to translate into numerical
 #file.loc[file["SEX"]== "F", "SEX"] = 1

# we are importing logistic Regression from sklearn kit
from sklearn import  cross_validation
from  sklearn import linear_model
from sklearn.linear_model import  LogisticRegression

# algorithm instantiation
algorithm = LogisticRegression(random_state=1)
Predictors = ["TOTCHOL", "SYSBP","DIABP","BMI"]
#Predicting values using 10 fold cross validation for train and test
Predictedhypertension = cross_validation.cross_val_predict(algorithm,file[Predictors],file["HYPERTEN"],cv=10)
score = cross_validation.cross_val_score(algorithm,file[Predictors],file["HYPERTEN"],cv=10)
file2 = pandas.DataFrame({"chol": file["TOTCHOL"],"sysbp" : file["SYSBP"],"diabp" : file["DIABP"],"bmi": file["BMI"],"givenhypertn": file["HYPERTEN"],"predhyptnsn": Predictedhypertension })
# Generation of Predicted values
file2.to_csv("Preditedfile", index=False)
print (score.mean())