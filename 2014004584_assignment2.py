import numpy as np
import math
import codecs
from konlpy.tag import Okt
from collections import defaultdict
from pandas import read_table


def main():

    with codecs.open("ratings_test.txt", 'r+', encoding='utf-8') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
        data = data[0:]

    doc_list = list(zip(*data))[1]

    model = NaiveBayesClassifier()
    model.train("ratings_train.txt")

    percent = 0

    length = len(doc_list)
    for d in range(1, round(length)):
        data[d][2] = str(round(model.classify(doc_list[d])))
        if d % 100 == 0 :
            percent += 1
            print(str(percent) + '%')

    file = open("ratings_result.txt", 'w')
    for i in data:
        file.write('\t'.join(i))
        file.write('\n')


class NaiveBayesClassifier:

    t = Okt()

    def tokenize(self, message):
        word = self.t.pos(message)
        return word

    def __init__(self, k=0.5):
        self.k = k
        self.word_probs = []

    def load_corpus(self, path):
        corpus = read_table(path, sep='\t', encoding='utf-8')
        corpus = np.array(corpus)
        return corpus

    def count_words(self, training_set):
        # 학습데이터는 영화리뷰 본문(doc), 평점(point)으로 구성
        counts = defaultdict(lambda: [0, 0])
        for _, doc, point in training_set:
            # 영화리뷰가 text일 때만 카운트
            if self.isNumber(doc) is False:
                # 리뷰를 띄어쓰기 단위로 토크나이징
                words = self.tokenize(doc)
                for word in words:
                    counts[word][0 if point == 1 else 1] += 1
        return counts

    def isNumber(self, s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    def word_probabilities(self, counts, total_class0, total_class1, k):
        # 단어의 빈도수를 [단어, p(w|긍정), p(w|부정)] 형태로 반환
        return [(w,
                 (class0 + k) / (total_class0 + 2 * k),
                 (class1 + k) / (total_class1 + 2 * k))
                for w, (class0, class1) in counts.items()]

    def good_class_probability(self, word_probs, doc):
        # 별도 토크나이즈 안하고 띄어쓰기로만
        docwords = self.tokenize(doc)

        # 초기값은 모두 0으로 처리
        log_prob_if_good_class = log_prob_if_bad_class = 0.0

        # 모든 단어에 대해 반복
        for word, prob_if_good_class, prob_if_bad_class in word_probs:
            # 만약 리뷰에 word가 나타나면
            # 해당 단어가 나올 log 확률을 더해 줌
            if word in docwords:
                log_prob_if_good_class += math.log(prob_if_good_class)
                log_prob_if_bad_class += math.log(prob_if_bad_class)

            # 만약 리뷰에 word가 나타나지 않는다면
            # 해당 단어가 나오지 않을 log 확률을 더해 줌
            # 나오지 않을 확률은 log(1-나올 확률)로 계산
            else:
                log_prob_if_good_class += math.log(1.0 - prob_if_good_class)
                log_prob_if_bad_class += math.log(1.0 - prob_if_bad_class)

        prob_if_good_class = math.exp(log_prob_if_good_class)
        prob_if_bad_class = math.exp(log_prob_if_bad_class)
        return prob_if_good_class / (prob_if_good_class + prob_if_bad_class)

    def train(self, trainfile_path):
        training_set = self.load_corpus(trainfile_path)

        # 범주0(긍정)과 범주1(부정) 문서 수를 세어 줌
        good_class = len([1 for _, _, point in training_set if point == 1])
        bad_class = len(training_set) - good_class
        print (good_class)
        print (bad_class)

        # train
        word_counts = self.count_words(training_set)
        self.word_probs = self.word_probabilities(word_counts,
                                                  good_class,
                                                  bad_class,
                                                  self.k)
        print("train finish")

    def classify(self, doc):
        return self.good_class_probability(self.word_probs, doc)


main()
