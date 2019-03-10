from sklearn.datasets import fetch_openml as fm
import matplotlib as pl
import matplotlib.pyplot as ppl
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score as cvs
import warnings
from sklearn.model_selection import cross_val_predict as cvp
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import precision_score, recall_score, f1_score

warnings.filterwarnings("ignore")

mnist_data=fm('mnist_784', version=1)
print("Keys:",mnist_data.keys())
X,Y=mnist_data["data"], mnist_data["target"]
print("Size of data:",X.shape)
print("Size of labels:",Y.shape)
#sd=X[0]
#sd_image=sd.reshape(28,28)
#ppl.imshow(sd_image, cmap=pl.cm.binary, interpolation="nearest")
#ppl.axis('off')
#ppl.show()
Y=Y.astype(np.uint8)
#print(Y[0])
trainx, testx, trainy, testy=X[:60000], X[60000:], Y[:60000], Y[60000:]
trainy_5=(trainy==5)
testy_5=(testy==5)
sgd_clf=SGDClassifier()
sgd_clf.fit(trainx, trainy_5)
#sd=X[3]
#print(sgd_clf.predict([sd]))
print("Accuracy on cross validation three times:",cvs(sgd_clf, trainx, trainy_5, cv=3, scoring="accuracy"))
trainy_pred=cvp(sgd_clf, trainx, trainy_5, cv=3)
confm=cm(trainy_5, trainy_pred)
print("Confusion Matrix:",confm)
p=precision_score(trainy_5, trainy_pred)
r=recall_score(trainy_5, trainy_pred)
f1=f1_score(trainy_5, trainy_pred)
print("Precision, Recall and f1 score on cross validation three times:",p,r,f1)
