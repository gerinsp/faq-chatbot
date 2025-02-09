from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

from src.config import FAQ_CSV_PATH
from src.rag_pipeline import get_answer

df_test = pd.read_csv(FAQ_CSV_PATH)

y_true = df_test["Answer"].tolist()
y_pred = [get_answer(q) for q in df_test["Question"].tolist()]

report = classification_report(y_true, y_pred, zero_division=0)
conf_matrix = confusion_matrix(y_true, y_pred)

print("Classification Report:\n", report)
print("Confusion Matrix:\n", conf_matrix)
