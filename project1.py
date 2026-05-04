import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
sns.set(style = "whitegrid")
import joblib
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv("AIML Dataset.csv")
print(df.head())
print(df.info())
print(df.columns)
print(df["isFraud"].value_counts())
print(df["isFlaggedFraud"].value_counts())
print(df.shape[0])
print(round(df["isFraud"].value_counts()[1]/df.shape[0]*100, 2))
print(df["amount"].describe().astype(int))



df["type"].value_counts().plot(kind="bar",title="Transaction Types",color="skyblue")
plt.xlabel("Transaction Type")
plt.ylabel("Count")
plt.show()

fraud_by_type = df.groupby("type")["isFraud"].mean().sort_values(ascending = False)
fraud_by_type.plot(kind="bar",title="Fraud Rate by Type",color="salmon")
plt.xlabel("Transaction Type")
plt.ylabel("Fraud Rate")
plt.show()

sns.histplot(np.log1p(df["amount"]),bins=100,kde=True,color="purple")
plt.title("Transaction Amount Distribution(log scaled)")
plt.xlabel("Log(Amount + 1)")
plt.show()

sns.boxplot(data=df[df["amount"]<50000],x = "isFraud",y = "amount")
plt.title("Amount vs isFraud(Filtered under 50k)")
plt.show()

 


df["balanceDiffOrig"]=df["oldbalanceOrg"]-df["newbalanceOrig"]
df["balanceDiffDest"]=df["newbalanceDest"]-df["oldbalanceDest"]
print(df["balanceDiffOrig"])
print(df["balanceDiffDest"])
print((df["balanceDiffOrig"] < 0).sum())
print((df["balanceDiffDest"] < 0).sum())

frauds_per_step = df[df["isFraud"] == 1]["step"].value_counts().sort_index()
plt.plot(frauds_per_step.index, frauds_per_step.values, label="Frauds per step")
plt.xlabel("step(Time)")
plt.ylabel("Numbers of Frauds")
plt.title("Fraud Over Time")
plt.grid(True)
plt.show()


(df.drop(columns = "step",inplace=True))
print(df.head())



top_recivers = df["nameDest"].value_counts().head(10)
print(top_recivers)
fraud_users= df[df["isFraud"] ==1]["nameOrig"].value_counts().head(10)
print(fraud_users)

fraud_type = df[df["type"].isin(("TRANSFER","CASH_OUT"))]
print(fraud_type.head(10))
print(fraud_type["type"].value_counts())

sns.countplot(data= fraud_type, x = "type",hue = "isFraud")
plt.title("Fraud Distribution in Transfer and Cash_Out")
plt.show()

corr = df[["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "isFraud"]].corr()
print(corr)

sns.heatmap(corr,annot= True,cmap = "coolwarm",fmt= ".2f")
plt.title("correlation Matrix")
plt.show()

zero_after_transfer = df[
    (df["oldbalanceOrg"]>0)&
    (df["newbalanceOrig"]==0)&
    (df["type"].isin(["TRANSFER","CASH_OUT"]))
]
len(zero_after_transfer)
print(zero_after_transfer.head(10))

df_model  =df.drop(["nameOrig","nameDest","isFlaggedFraud"], axis=1)

categorical = ["type"]
numeric = ["amount","oldbalanceOrg","newbalanceOrig","oldbalanceDest","newbalanceDest"]

y = df_model["isFraud"]
x = df_model.drop("isFraud",axis = 1)
x_train,x_test, y_train,y_test = train_test_split(x,y, test_size=0.2,stratify = y)


preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(drop='first', handle_unknown='ignore'), categorical)
    ]
)

pipeline = Pipeline([
    ("prep", preprocessor),
    ("clf", RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

pipeline.fit(x_train, y_train)
y_pred = pipeline.predict(x_test)

print(classification_report(y_test, y_pred))
print("Accuracy:", pipeline.score(x_test, y_test) * 100)

joblib.dump(pipeline, "fraud_detection_pipeline.pkl")