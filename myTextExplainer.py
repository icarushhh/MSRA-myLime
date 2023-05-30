import re
import random

import graphviz
import numpy as np
import scipy as sp
import sklearn
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree

import my_lime


class SplitedString(object):
    """1. split raw string into array of words
    2. create a vocabulary and each words' position in the string
    3. offer a method to remove certain word in the string
    """

    def __init__(self, raw_string):
        self.raw_string = raw_string

        split_expression = r'\W+'
        splitter = re.compile(r'(%s)|$' % split_expression)
        self.word_list = [s for s in splitter.split(self.raw_string) if s]
        # word_list  是一个list，保存分割后的word，包括标点和空格

        self.word_list_without_splitter = list(filter(None, re.split(split_expression, self.raw_string)))
        self.vocab = []
        self.position = []
        # self.vocab 是单词表，nonredundant
        # self.position 保存vocab 中对应单词在word_list 的位置

        for i, word in enumerate(self.word_list):
            if word in self.word_list_without_splitter:
                if word not in self.vocab:
                    self.vocab.append(word)
                    self.position.append([i])
                else:
                    index = self.vocab.index(word)
                    self.position[index].append(i)

    def vocab_size(self):
        return len(self.vocab)

    def raw_string(self):
        return self.raw_string

    def remove_words(self, index_to_remove):
        """input：一个1 x num_of_vocab 的list，其中0表示remove vocab中相应单词
        output：删去单词后的str
        """
        position_to_remove = []
        result_string = ""
        for i, flag in enumerate(index_to_remove):
            if not flag:
                for p in self.position[i]:
                    position_to_remove.append(p)

        for i, word in enumerate(self.word_list):
            if i in position_to_remove:
                continue
            else:
                result_string += word

        return result_string


class DataPerturber(object):
    """
    数据扰动器，输入:
    splited_string, num_of_samples, predictor
    提供一个perturb函数
    输出:
    neighbor_data：num_of_samples+1 x vocab_size,
    neighbor_labels: num_of_samples+1 x 2,
    distances: 1 x num_of_samples
    """

    def __init__(self, splited_string, num_of_samples, predictor):
        self.splited_string = splited_string
        self.num_of_samples = num_of_samples
        self.predictor = predictor
        self.vocab_size = splited_string.vocab_size()
        self.neighbor_wordbags = np.ones(((num_of_samples+1), self.vocab_size))   # 受扰动数据词袋，第一维是原数据词袋
        self.neighbor_rawstring = []
        self.neighbor_rawstring.append(splited_string.raw_string)
        self.neighbor_labels = []

    def perturbe(self):

        def distance_fn(neighbors):
            """距离函数，计算neighbor_data到其第一维（origin_data）的距离，参考lime源码
            """
            return sklearn.metrics.pairwise.pairwise_distances(
                neighbors, neighbors[0].reshape(1, -1), metric='cosine').ravel() * 100

        random.seed(99)         # 设置随机数种子，保证每次扰动结果相同
        perturbe_size = []      # 1 x num_of_samples-1 的数组，表示每个受扰动数据扰动word数量（1～vocab_size-1）
        for i in range(self.num_of_samples):
            perturbe_size.append(random.randint(1, self.vocab_size - 1))

        word_indexes = [i for i in range(self.vocab_size)]

        for i, size in enumerate(perturbe_size, start=1):
            indexes_samples = random.sample(word_indexes, size)
            self.neighbor_wordbags[i, indexes_samples] = 0
            string_after_removing = self.splited_string.remove_words(self.neighbor_wordbags[i])
            self.neighbor_rawstring.append(string_after_removing)

        self.neighbor_labels = self.predictor(self.neighbor_rawstring)
        distances = distance_fn(self.neighbor_wordbags)

        return self.neighbor_wordbags, self.neighbor_labels, distances


class TextExplainer(object):

    def __init__(self,
                 kernel_width=25,
                 class_names=['negative', 'postive'],
                 feature_selection='auto',):

        def kernel(d):
            """kernel_fn: function that transforms an array of distances into an
            array of proximity values (floats).
            """
            return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        self.kernel_fn = kernel
        self.class_names = class_names
        self.feature_selection = feature_selection

    def explain_string_by_mylime(self, raw_string, predictor, num_features=10, num_samples=5000):
        """
        using lime with a linear model to explain a text
        """
        self.splited_string = SplitedString(raw_string)
        perturber = DataPerturber(self.splited_string, num_samples, predictor)
        (neighbors, neighbor_labels, distance) = perturber.perturbe()

        base = my_lime.MyLime(self.kernel_fn)
        (self.list, predict_score, local_predict) = base.explain_instance_linear(neighbors, neighbor_labels, distance, label=1, num_features=num_features,
                                   feature_selection=self.feature_selection)

        return

    def show_as_list(self):
        vocab = self.splited_string.vocab
        words = []
        coef = []
        for x in self.list:
            words.append(vocab[x[0]])
            coef.append(x[1])

        self.output = sorted(zip(words, coef),
                       key=lambda x: np.abs(x[1]), reverse=True)

        print(self.output)
        return

    def explain_string_by_treelime(self, raw_string, predictor, num_features = 10, num_samples=5000, max_depth=5):
        """
        using lime with decision tree to explain
        """
        self.num_features = num_features
        self.splited_string = SplitedString(raw_string)
        perturber = DataPerturber(self.splited_string, num_samples, predictor)
        (neighbors, neighbor_labels, distance) = perturber.perturbe()

        base = my_lime.MyLime(self.kernel_fn)

        self.tree_model = base.explain_instance_tree(neighbors, neighbor_labels, distance, max_depth)

        return

    def show_tree(self):
        dot_data = export_graphviz(self.tree_model, out_file=None,
                                    feature_names= self.splited_string.vocab,
                                    class_names=self.class_names,
                                    filled=True, impurity=False)
        graph = graphviz.Source(dot_data, format="png")
        graph.render('text/tree_model')

        plot_tree(self.tree_model)

        self.importances = sorted(zip(self.splited_string.vocab, self.tree_model.feature_importances_),
                key=lambda x: np.abs(x[1]), reverse=True)
        self.importances = self.importances[:self.num_features]

        print(self.importances)

        return






