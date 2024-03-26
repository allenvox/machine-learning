import numpy as np
import matplotlib.pyplot as plt

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        num_samples, num_features = X.shape
        unique_classes = np.unique(y)

        # Stopping criteria
        if len(unique_classes) == 1 or (self.max_depth and depth == self.max_depth):
            return {'class': unique_classes[0]}

        # Find best split
        best_split = self._find_best_split(X, y, num_samples, num_features)

        # Split recursively
        left_indices = np.where(X[:, best_split['feature']] <= best_split['threshold'])[0]
        right_indices = np.where(X[:, best_split['feature']] > best_split['threshold'])[0]

        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return {'feature': best_split['feature'],
                'threshold': best_split['threshold'],
                'left': left_subtree,
                'right': right_subtree}

    def _find_best_split(self, X, y, num_samples, num_features):
        best_split = {}
        best_gini = float('inf')

        for feature_idx in range(num_features):
            thresholds = np.unique(X[:, feature_idx])

            for threshold in thresholds:
                left_indices = np.where(X[:, feature_idx] <= threshold)[0]
                right_indices = np.where(X[:, feature_idx] > threshold)[0]

                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue

                gini = self._gini_index(y[left_indices], y[right_indices])

                if gini < best_gini:
                    best_gini = gini
                    best_split = {'feature': feature_idx, 'threshold': threshold}

        return best_split

    def _gini_index(self, left_y, right_y):
        num_left = len(left_y)
        num_right = len(right_y)
        total = num_left + num_right

        gini_left = 1.0 - sum([(np.sum(left_y == c) / num_left) ** 2 for c in np.unique(left_y)])
        gini_right = 1.0 - sum([(np.sum(right_y == c) / num_right) ** 2 for c in np.unique(right_y)])

        gini = (num_left / total) * gini_left + (num_right / total) * gini_right
        return gini

    def predict(self, X):
        return np.array([self._predict_tree(x, self.tree) for x in X])

    def _predict_tree(self, x, tree):
        if 'class' in tree:
            return tree['class']

        if x[tree['feature']] <= tree['threshold']:
            return self._predict_tree(x, tree['left'])
        else:
            return self._predict_tree(x, tree['right'])

    def print_tree(self, tree=None, indent=" "):
        if tree is None:
            tree = self.tree
        if 'class' in tree:
            print(str(tree['class']))
        else:
            print("X_" + str(tree['feature']) + " <= " + str(tree['threshold']) + "?")
            print("%sleft:" % (indent), end="")
            self.print_tree(tree['left'], indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree['right'], indent + indent)


class RandomForest:
    def __init__(self, n_estimators=100, max_depth=None, bootstrap_ratio=0.8):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.bootstrap_ratio = bootstrap_ratio
        self.trees = []

    def fit(self, X, y):
        num_samples = X.shape[0]
        num_bootstrap_samples = int(self.bootstrap_ratio * num_samples)
        for _ in range(self.n_estimators):
            bootstrap_indices = np.random.choice(num_samples, num_bootstrap_samples, replace=True)
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X[bootstrap_indices], y[bootstrap_indices])
            self.trees.append(tree)
            for i, tree in enumerate(rf.trees):
                print("Tree", i + 1)
                tree.print_tree()

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.array([np.bincount(predictions[:, i]).argmax() for i in range(X.shape[0])])

# Example usage:
# Generate some synthetic data for classification
np.random.seed(42)
X = np.random.rand(100, 2) * 10
y = np.array([int(x[0] + x[1] > 10) for x in X])

# Train Random Forest model
rf = RandomForest(n_estimators=5, max_depth=3)
rf.fit(X, y)

# Visualize decision boundary
def plot_decision_boundary(X, y, model):
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')
    plt.show()

plot_decision_boundary(X, y, rf)
