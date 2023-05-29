import numpy as np
import scipy as sp
from sklearn.linear_model import Ridge, lars_path, ElasticNet, LinearRegression, Lasso
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.utils import check_random_state

class MyLime(object):

    def __init__(self, kernel_fn, random_state=None):
        self.kernel_fn = kernel_fn
        self.random_state = check_random_state(random_state)

    def generate_lars_path(weighted_data, weighted_labels):
        """Generates the lars path for weighted data.

        Args:
            weighted_data: data that has been weighted by kernel
            weighted_label: labels, weighted by kernel

        Returns:
            (alphas, coefs), both are arrays corresponding to the
            regularization parameter and coefficients, respectively
        """
        x_vector = weighted_data
        alphas, _, coefs = lars_path(x_vector,
                                     weighted_labels,
                                     method='lasso',
                                     verbose=False)
        return alphas, coefs

    def forward_selection(self, data, labels, weights, num_features):
        """Iteratively adds features to the model"""
        # clf = Ridge(alpha = 0, fit_intercept=True, random_state=self.random_state)
        clf = LinearRegression(fit_intercept=True)
        # clf = Lasso(alpha = 0.02, fit_intercept=True, random_state=self.random_state)

        used_features = []
        for _ in range(min(num_features, data.shape[1])):
            max_ = -100000000
            best = 0
            for feature in range(data.shape[1]):
                if feature in used_features:
                    continue
                clf.fit(data[:, used_features + [feature]], labels,
                        sample_weight=weights)
                score = clf.score(data[:, used_features + [feature]],
                                  labels,
                                  sample_weight=weights)
                if score > max_:
                    best = feature
                    max_ = score
            used_features.append(best)
        return np.array(used_features)

    def feature_selection(self, data, labels, weights, num_features, method):
        """Selects features for the model. see explain_instance_with_data to
           understand the parameters."""
        if method == 'none':
            return np.array(range(data.shape[1]))
        elif method == 'forward_selection':
            return self.forward_selection(data, labels, weights, num_features)
        elif method == 'highest_weights':
            # clf = Ridge(alpha=0.01, fit_intercept=True,
            #             random_state=self.random_state)
            clf = LinearRegression(fit_intercept=True)
            # clf = Lasso(alpha = 0.02, fit_intercept=True, random_state=self.random_state)

            clf.fit(data, labels, sample_weight=weights)


            coef = clf.coef_
            if sp.sparse.issparse(data):
                coef = sp.sparse.csr_matrix(clf.coef_)
                weighted_data = coef.multiply(data[0])
                # Note: most efficient to slice the data before reversing
                sdata = len(weighted_data.data)
                argsort_data = np.abs(weighted_data.data).argsort()
                # Edge case where data is more sparse than requested number of feature importances
                # In that case, we just pad with zero-valued features
                if sdata < num_features:
                    nnz_indexes = argsort_data[::-1]
                    indices = weighted_data.indices[nnz_indexes]
                    num_to_pad = num_features - sdata
                    indices = np.concatenate((indices, np.zeros(num_to_pad, dtype=indices.dtype)))
                    indices_set = set(indices)
                    pad_counter = 0
                    for i in range(data.shape[1]):
                        if i not in indices_set:
                            indices[pad_counter + sdata] = i
                            pad_counter += 1
                            if pad_counter >= num_to_pad:
                                break
                else:
                    nnz_indexes = argsort_data[sdata - num_features:sdata][::-1]
                    indices = weighted_data.indices[nnz_indexes]
                return indices
            else:
                weighted_data = coef * data[0]
                feature_weights = sorted(
                    zip(range(data.shape[1]), weighted_data),
                    key=lambda x: np.abs(x[1]),
                    reverse=True)
                return np.array([x[0] for x in feature_weights[:num_features]])
        elif method == 'lasso_path':
            weighted_data = ((data - np.average(data, axis=0, weights=weights))
                             * np.sqrt(weights[:, np.newaxis]))
            weighted_labels = ((labels - np.average(labels, weights=weights))
                               * np.sqrt(weights))
            nonzero = range(weighted_data.shape[1])
            _, coefs = self.generate_lars_path(weighted_data,
                                               weighted_labels)
            for i in range(len(coefs.T) - 1, 0, -1):
                nonzero = coefs.T[i].nonzero()[0]
                if len(nonzero) <= num_features:
                    break
            used_features = nonzero
            return used_features
        elif method == 'auto':
            if num_features <= 6:
                n_method = 'forward_selection'
            else:
                n_method = 'highest_weights'
            return self.feature_selection(data, labels, weights,
                                          num_features, n_method)


    def explain_instance_linear(self, neighbor_wordbags, neighbor_labels, distances, label = 1, num_features = 5, feature_selection='auto'):
        weights = self.kernel_fn(distances)
        labels_column = neighbor_labels[:, label]
        used_features = self.feature_selection(neighbor_wordbags,
                                               labels_column,
                                               weights,
                                               num_features,
                                               feature_selection)

        easy_model = LinearRegression(fit_intercept=True)
        # easy_model = Ridge(alpha=1, fit_intercept=True, random_state=self.random_state)
        # easy_model = Lasso(alpha=0.02, fit_intercept=True, random_state=self.random_state)

        # easy_model = DecisionTreeRegressor(random_state=self.random_state)

        easy_model.fit(neighbor_wordbags[:, used_features],
                       labels_column, sample_weight=weights)
        prediction_score = easy_model.score(
            neighbor_wordbags[:, used_features],
            labels_column, sample_weight=weights)

        local_pred = easy_model.predict(neighbor_wordbags[0, used_features].reshape(1, -1))

        return (sorted(zip(used_features, easy_model.coef_),
                       key=lambda x: np.abs(x[1]), reverse=True),
                prediction_score, local_pred)

    def explain_instance_tree(self, neighbor_wordbags, neighbor_labels, distances):
        """
        explain a text with decision tree
        """

        weights = self.kernel_fn(distances)

        self.binary_labels = []
        for label in neighbor_labels:
            self.binary_labels.append(1 if label[1] > label[0] else 0)

        tree_model = DecisionTreeClassifier(random_state=0, max_depth=20)
        tree_model.fit(neighbor_wordbags, self.binary_labels, sample_weight=weights)

        return tree_model







