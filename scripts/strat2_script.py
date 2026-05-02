import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import StandardScaler

y_train = np.load('y_train.npy')[:55000]
y_test = np.load('y_test.npy')

embedding_train = np.load('X_strat2_from0_at55000.npy')
embedding_test = np.load('X_strat2_test_FINAL.npy')

mask = ~np.all(embedding_train == 0, axis=1)
X_train_clean = embedding_train[mask]
y_train_clean  = y_train[mask]


mask = ~np.all(embedding_test == 0, axis=1)
X_test_clean = embedding_test[mask]
y_test_clean  = y_test[mask]




print("embedding_train:", embedding_train.shape)
print("embedding_test: ", embedding_test.shape)
print("y_train:        ", y_train.shape)
print("y_test:         ", y_test.shape)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_clean)
X_test_scaled  = scaler.transform(X_test_clean)

lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train_clean)

y_pred = lr.predict(X_test_scaled)

print(classification_report(y_test_clean, y_pred))
print("F1:", f1_score(y_test_clean, y_pred))



