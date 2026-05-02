from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,f1_score
import numpy as np
import pickle
import pandas as pd

df = pd.read_csv('cleaned_data.csv')


# print(df.head())
proteins = df['uniprot_id'].unique()
train_proteins,test_proteins = train_test_split(proteins,test_size=0.25,random_state=42)

train_df = df[df['uniprot_id'].isin(train_proteins)]
test_df = df[df['uniprot_id'].isin(test_proteins)]


# print(train_df.shape,test_df.shape)

ohe = OneHotEncoder(sparse_output=False)
scaler = StandardScaler()

train_df_ohe = ohe.fit_transform(train_df[['wt','mut']])


test_df_ohe = ohe.transform(test_df[['wt','mut']])

train_df_ohe = pd.DataFrame(train_df_ohe,index=train_df.index)
# print(train_df_ohe)

test_df_ohe = pd.DataFrame(test_df_ohe,index=test_df.index)

# print(test_df_ohe.info())

train_df_scaled = scaler.fit_transform(train_df[['position']])
test_df_scaled = scaler.transform(test_df[['position']])

# print(train_df_scaled)

train_df_scaled = pd.DataFrame(train_df_scaled, index=train_df.index)
test_df_scaled  = pd.DataFrame(test_df_scaled, index=test_df.index)

x_train = pd.concat([train_df_ohe, train_df_scaled], axis=1)
x_test  = pd.concat([test_df_ohe, test_df_scaled], axis=1)

y_train = train_df['pathogenicity']
y_test = test_df['pathogenicity']

with open("split.pkl",'wb') as f:
    pickle.dump((train_df,test_df),f)

model = RandomForestClassifier(n_estimators=200,n_jobs=-1,class_weight='balanced',random_state=42)

model.fit(x_train,y_train)

y_predicted = model.predict(x_test)

Cr = classification_report(y_predicted,y_test)
F1 = f1_score(y_predicted,y_test)

print("Classification Report : ",Cr)
print("========================================================")
print("F1 Score : ",F1)
print("========================================================")