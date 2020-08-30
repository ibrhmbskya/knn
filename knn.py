import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt

#satır isimleri
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
#pandas kullanılarak veri okunur
df = pd.read_csv("pima-indians-diabetes.csv", header=None, names=col_names)

#Verilerin doğru okunduğunu kontrol ediyor
df.head()
#veri kümesindeki satır ve sütun sayısını kontrol ediyor
df.shape
#Hedef sütun hariç tüm eğitim verilerinin yer aldığı bir veri çerçevesi oluşturulur
X = df.drop(columns=['label'])
#Hedef değişkenin kaldırıldığından emin olunur
X.head()
#ayrı hedef değerler
y = df['label'].values

#veri setini eğitim ve test verisine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

# KNN classifier oluşturulur
knn = KNeighborsClassifier(n_neighbors = 3)

# Fit the classifier to the data
knn.fit(X_train,y_train)

#cTest verilerimizdeki modelimizin doğruluğunu kontrol eder
knn.score(X_test, y_test)

print(knn.score(X_test, y_test))

#yeni bir KNN model oluşturur
knn_cv = KNeighborsClassifier(n_neighbors=3)

#5 cv ile eğitim modeli
cv_scores = cross_val_score(knn_cv, X, y, cv=5)

#doğruluk oranı yazdırılır.
print("Accuracy:",format(np.mean(cv_scores)))

#hata matrisi oluşturuluyor
pred=knn.predict(X_test)
cmat = confusion_matrix(y_test, pred)
print(cmat)

#hata matrisi grafiği çizdirilir.
plt.matshow(cmat)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

