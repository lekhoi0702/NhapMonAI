import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

class NaiveBayes:
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        self.mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self.var = np.zeros((n_classes, n_features), dtype=np.float64)
        self.priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0)
            self.priors[idx] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        posteriors = []

        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            posterior = np.sum(np.log(self._pdf(idx, x)))
            posterior = posterior + prior
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator


iris = load_iris()
X = iris.data
y = iris.target

#
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


nb = NaiveBayes()
nb.fit(X_train, y_train)


predictions = nb.predict(X_test)


class_names = iris.target_names



for i in range(len(X_test)):
    print("Mẫu test:", X_test[i])
    print("Dự đoán:", class_names[predictions[i]])
    print()

# Calculate accuracy
accuracy = np.sum(y_test == predictions) / len(y_test)
print("Độ chính xác:", accuracy)
