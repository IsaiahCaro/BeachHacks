import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

user_data_dict = pickle.load(open('user_data.pickle', 'rb'))

data = np.asarray(user_data_dict['user_data'])
labels = np.asarray(user_data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size = 0.2, shuffle = True, stratify=labels)

user_model = RandomForestClassifier()

user_model.fit(x_train, y_train)

y_predict = user_model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly! '.format(score * 100))

f = open('user_model.p', 'wb')
pickle.dump({'user_model': user_model}, f)
f.close()
