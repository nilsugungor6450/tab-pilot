from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from tabpfn import TabPFNRegressor

X, y = load_diabetes(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)

model = TabPFNRegressor()
model.fit(X_tr, y_tr)
print("MAE:", mean_absolute_error(y_te, model.predict(X_te)))