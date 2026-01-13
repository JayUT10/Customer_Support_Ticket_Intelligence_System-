import joblib
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay)

X=joblib.load('models/X.pkl')
y=joblib.load('models/y_category.pkl')

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2, random_state=42, stratify=y)

model=LinearSVC()
model.fit(X_train,y_train)

y_pred=model.predict(X_test)

print("SVM Accuracy: ",accuracy_score(y_test,y_pred))
print("\nClassification Report:\n", classification_report(y_test,y_pred))

cofm=confusion_matrix(y_test,y_pred)
disp=ConfusionMatrixDisplay(confusion_matrix=cofm)
disp.plot()
plt.title("SSVM Category Confusion Matrix")
plt.savefig('preprocessing/plots/SVM_category_confusion_matrix.png')
plt.show()
joblib.dump(model,'models_SVM/svm_category_model.pkl')
print("training completed")