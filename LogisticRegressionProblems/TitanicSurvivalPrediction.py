import kagglehub
import pandas as pd
import os

# path = kagglehub.dataset_download("yasserh/titanic-dataset", 
#                                   output_dir = "D:/MachineLearning/LogisticRegressionProblems/data")

# print(path)

df = pd.read_csv("data/Titanic-Dataset.csv")
# print(df.head())
#print(os.listdir(path)) # show what files it contains

#before preprocessing lets inspect what data it contains
# print("shape :" , df.shape)
# print("columns :", df.columns.tolist())
# print("datatypes :", df.dtypes)

# our target is passenger survided or not 
y = df["Survived"]
# print(y)

x = df.drop("Survived", axis = 1)
# print(x.shape)

# pid
# print(x["PassengerId"]) #we can observe , nothing we can do with passenger id
x = df.drop("PassengerId", axis = 1)

print(x.isnull().sum())

# there are 177 missing values in age
# we can use median to fill the values
# median is better than mean beacuse mean pulls the average heavily if the highest age is above 80

df["Age"] = df["Age"].fillna(df["Age"].median())

# cabin has 687 null values among 890

# print(df["Cabin"])

# does canbin contains usefull information , yes it conatins beacuse cabin location contains useful info
# for survival

# but there is another problem 77% values are null
# so investigating the cabin values
print("==========")
print(df["Cabin"].nunique())

# comparing the survived information with cabin data
print(pd.crosstab(df["Cabin"].notna(), df["Survived"], normalize="index"))

#so cabin carries huge information beacuse passengers who has cabin info survived 66%
# passengers who doesnt have cabin info survived 30% only.

print(df["Cabin"].str[0].value_counts())

# so finally we have observed that the cabin has simpler representation of first letter str[0] 
# and also it provides meaning info , person who has cabin , survival rate 66% who doesnt have only 30%

print(df[df["Embarked"].isnull()])
# we can observe that the these two NAN passengers are travelling together

print(df[df["Ticket"] == "113572"][["PassengerId", "Name", "Ticket", "Fare", "Cabin", "Embarked"]])
# again these two are only passengers with same ticket

print(df["Embarked"].value_counts())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
print(df.isnull().sum())

#now handle cabin value
# craeting a new column called deck

df["Deck"] = df["Cabin"].str[0]
print(df[["Cabin", "Deck"]].head(20))

# cabin know 
df["CabinKnown"] = df["Deck"].notna().astype(int)
print(df[["Cabin", "Deck", "CabinKnown"]].head(10))

df["Deck"] = df["Deck"].fillna("Unknown")
# print(df["Deck"].value_counts())

# print(df.info())
#age is object 

# print(df["Age"].head(20))
# print(df["Age"].dtype)

# print(df[["PassengerId", "Survived"]].head(10))
# print(df["Name"].head(10))

# print(df["Name"].str.extract(r",\s*([^.]*)\.")[0].value_counts())
# print(
#     df.groupby(
#         df["Name"].str.extract(r",\s*([^.]*)\.")[0]
#     )["Survived"].mean().sort_values(ascending=False)
# )

#we can observe that there are common ones and rare ones , in rare one lady who survived but 
# lady is only one passenger among the all categories 
# one passenger surviving doesnt establish a reliable pattern 

#thats why we eventually grouping uncommon titles into group rare

df["Title"] = df["Name"].str.extract(r",\s*([^.]*)\.")[0] 
# print(df[["Name", "Title"]].head(10))

common = ["Mr", "Miss", "Mrs", "Master"]

df["Title"] = df["Title"].apply(
    lambda x : x if x in common else "Rare"
)

print(df["Title"].value_counts())

print(df["Sex"].value_counts())

print(df["Embarked"].value_counts()) #no order we can perform one hot encodding 

print(df["Pclass"].value_counts().sort_index()) # has order

print(df.groupby("Pclass")["Survived"].mean()) #the class 1 has more survival rate

print(df.groupby("SibSp")["Survived"].mean()) #siblings has no clean linear relationship

print(df.groupby("Parch")["Survived"].mean()) #same as sibilings

print(df.groupby("Survived")["Fare"].mean()) #higher fare higher survival rate can relate with pclass

print(df["Ticket"].nunique())

# print(df["Ticket"].value_counts().head(10)) 
# there is no useful information with the ticket no , its just a identifier 
# each ticket has some group of numbers so we can craete groups

df["TicketGroupSize"] = df["Ticket"].map(df["Ticket"].value_counts())
# print(df.groupby("Ticket")["Survived"].mean().head(10))

print("====================")
ticket_counts = df["Ticket"].value_counts()
df["TicketGroupSize"] = df["Ticket"].map(ticket_counts)

# print(df.groupby("TicketGroupSize")["Survived"].mean())

# print(df.groupby(pd.cut(df["PassengerId"], 10))["Survived"].mean()) #no meaningful feature from passenger id

# print(df.groupby("Deck")["Survived"].mean())

##done understanding columns , now we have to build features

print(df.columns.tolist())

features = ["Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked",
    "Deck",
    "CabinKnown",
    "Title",
    "TicketGroupSize"]

X = df[features]
y = df["Survived"]

print(X.shape)
print(y.shape)

categorical = ["Embarked", "Deck", "Title"]

X["Sex"] = X["Sex"].map({
    "male" : 0,
    "female": 1
})

X = pd.get_dummies(
    X,
    columns=["Embarked", "Deck", "Title"],
    drop_first=True,
    dtype = int
)

print(X.head())
print(X.dtypes)

# preprocessing is completed

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=y)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

x_train_scaler = scaler.fit_transform(X_train)
x_test_scaler = scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(x_train_scaler, y_train)

print(model.coef_)
print(model.intercept_)

y_pred = model.predict(x_test_scaler)

print(y_pred[:20])
print(y_test.values[:20])

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

y_prob = model.predict_proba(x_test_scaler)
print(y_prob[:10])

passenger = X_test.iloc[2]

print(passenger)

print("Train Accuracy:", model.score(x_train_scaler, y_train))
print("Test Accuracy:", model.score(x_test_scaler, y_test))

from sklearn.metrics import log_loss

test_loss = log_loss(y_test, y_prob[:, 1])

train_prob = model.predict_proba(x_train_scaler)

train_loss = log_loss(y_train, train_prob[:, 1])

print("Train Log Loss:", train_loss)
print("Test Log Loss:", test_loss)

for c in [0.01, 0.1, 0.5, 1, 2, 5, 10]:
    
    model = LogisticRegression(C=c, max_iter=1000)
    model.fit(x_train_scaler, y_train)
    
    accuracy = model.score(x_test_scaler, y_test)
    
    print("C =", c, "Accuracy =", accuracy)

# "I trained a Logistic Regression model on the Titanic dataset. "
# "The baseline model achieved about 83.8% test accuracy. "
# "I then tuned the regularization parameter C and achieved 84.92% accuracy with C=10."