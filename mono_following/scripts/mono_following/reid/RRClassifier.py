import numpy as np
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from scipy.spatial import distance
from sympy import gamma

import random
from random import choice
random.seed(3)

np.random.seed(3)

class RRClassifier:
    def __init__(self, alpha, sampleNumbers):
        # for long-term classification model
        self.clf = None
        self.posTrainSet = []
        self.negTrainSet = []
        self.posLabels = []
        self.negLabels = []
        self.posSamples = sampleNumbers
        self.negSamples = sampleNumbers
        self.negs = 10
        self.alpha = alpha
        self.nummm = 0

    def update_cache(self, features, labels):
        print("[Frame {}]".format(self.nummm))
        print("pos short-term: {}".format(len(self.posTrainSet)))
        print("neg short-term: {}".format(len(self.negTrainSet)))
        for i in range(len(labels)):
            # score_pos_st = score_neg_st = 0.0
            if labels[i] == 1:
                self.posTrainSet.append(features[i][0])
                self.posLabels.append(1)
            else:
                self.negTrainSet.append(features[i][0])
                self.negLabels.append(0)

            if len(self.posTrainSet) > self.posSamples:
                self.posTrainSet = self.posTrainSet[-self.posSamples:]
                self.posLabels = self.posLabels[-self.posSamples:]
            if len(self.negTrainSet) > self.negSamples:
                self.negTrainSet = self.negTrainSet[-self.negSamples:]
                self.negLabels = self.negLabels[-self.negSamples:]
        self.nummm +=1
    def update_classifier(self):
        trainSet = self.posTrainSet + self.negTrainSet
        labels = self.posLabels + self.negLabels
        self.clf = Ridge(alpha=self.alpha, random_state=1)
        # self.clf = KernelRidge(alpha=1.0, kernel="rbf",gamma=1.0)
        self.clf.fit(trainSet, labels)
    
    def predict(self, feature):
        return self.clf.predict([feature])[0]

"""Ridge Regression Classifier with strategy

"""
class RRClassifierWithStrategy(RRClassifier):
    def __init__(self, alpha, sampleNumbers, threshold):
        super(RRClassifierWithStrategy, self).__init__(alpha, sampleNumbers)
        self.clf = Ridge(alpha=self.alpha, random_state=1)
        self.heights = []
        self.ratios = []
        self.distances = []
        self.IMAGE_WIDTH = 1280
        self.IMAGE_HEIGHT = 720
        self.MAX_DISTANCE = 5.0
        self.MAX_RATIO = 5.0
        self.SLIDING_WINDOW_SIZE = 4
        self.factor_appearance = 0.3  
        self.factor_aspectRatio = 0.3
        self.factor_height = 0.2
        self.factor_distance = 0.2
        self.THRESHOLD = threshold
        self.alpha = alpha

        self.samples = sampleNumbers
        self.samples_st = sampleNumbers // 2
        self.samples_lt = sampleNumbers * 4

        self.posTrainSet_lt = []
        self.negTrainSet_lt = []
        self.posLabels_lt = []
        self.negLabels_lt = []
        self.posTrainSet_st = []
        self.negTrainSet_st = []
        self.posLabels_st = []
        self.negLabels_st = []

        self.nummm = 0

    def getBoxInfo(self, box):
        """get box information
        Input:
            box: List of (1,4) with x1, y1, x2, y2
        Output:
            box's width, box's aspect ratio
        """
        width = abs(int(box[0,0])-int(box[0,2]))
        height = abs(int(box[0,1])-int(box[0,3]))
        # print(height, height/width)
        return height, height/width
    
    def scaledHeight(self, height):
        """scale it to [0,1]
        """
        return height/self.IMAGE_HEIGHT
    
    def scaledAspectRatio(self, aspectRatio):
        """scale it to [0,1]
        aspect ratio=height/width
        """
        if aspectRatio > self.MAX_RATIO:
            return 1.0
        else:
            return aspectRatio/self.MAX_RATIO
    
    def scaledDistance(self, distance):
        """scale it to [0,1]
        """
        if distance > self.MAX_DISTANCE:
            return 1.0
        else:
            return distance/self.MAX_DISTANCE

    def scaledCosSimilarity(self, query, reference):
        """calculate mean cos similarity "difference" between the query and reference
        """
        meanCosSimilarity = 0.0
        for ref in reference:
            meanCosSimilarity = meanCosSimilarity + distance.cosine(ref, query)
        return meanCosSimilarity/len(reference)

    def preprocess(self, feature):
        """preprocess the feature
        Input:
            (descriptor, distance, box)
        Output:
            (descriptor, scaledHeight, scaledAspectRatio, scaledDistance)
        """
        scaledDistance = self.scaledDistance(feature[1])
        height, aspectRatio = self.getBoxInfo(feature[2])
        scaledHeight = self.scaledHeight(height)
        scaledAspectRatio = self.scaledAspectRatio(aspectRatio)
        return [feature[0], scaledHeight, scaledAspectRatio, scaledDistance]

    def getScore(self, query, reference):
        """check whether is a good sample
        Input:
            query: (descriptor, scaledHeight, scaledAspectRatio, scaledDistance)
            reference: List[NUMS, array(descriptor, scaledHeight, scaledAspectRatio, scaledDistance)]
        """
        reference_np = np.asarray(reference, dtype=object)
        length = len(reference)
        score = self.factor_appearance * self.scaledCosSimilarity(query[0], reference_np[:, 0])\
            + self.factor_height * abs(query[1] - sum(reference_np[:, 1])/length)\
            + self.factor_aspectRatio * abs(query[2] - sum(reference_np[:, 2])/length)\
            + self.factor_distance * abs(query[3] - sum(reference_np[:, 3])/length)
        return score, self.scaledCosSimilarity(query[0], reference_np[:, 0]), abs(query[1] - sum(reference_np[:, 1])/length), abs(query[2] - sum(reference_np[:, 2])/length), abs(query[3] - sum(reference_np[:, 3])/length)
    
    # def update_cache_with_dist(feature, dist):
        
    def update_cache(self, features, labels):
        """update our long term feature cache
        Input:
            features: [N, feature], feature = (descriptor, distance, box)
        """
        output = []
        for i in range(len(features)):
            score_pos_st = score_neg_st = 0.0
            feature = self.preprocess(features[i])
            dist = feature[3]
            if labels[i] == 1:
                ### entry requirements ###
                if len(self.posTrainSet)!=0:
                    score, a, b, c, d = self.getScore(feature, self.posTrainSet[-self.SLIDING_WINDOW_SIZE:])
                    # print("Score  diff: {:.3f}".format(score))
                    # print("CosSim diff: {:.3f}".format(a))
                    # print("Height diff: {:.3f}".format(b))
                    # print("AsRa   diff: {:.3f}".format(c))
                    # print("Dist   diff: {:.3f}".format(d))
                    if score > self.THRESHOLD:
                        self.posTrainSet_lt.append(feature)
                        self.posLabels_lt.append(1)
                self.posTrainSet_st.append(feature)
                self.posLabels_st.append(1)
                # output.append((1, score,a,b,c,d))
            else:
                ### entry requirements ###
                if len(self.negTrainSet)!=0:
                    score, a, b, c, d = self.getScore(feature, self.negTrainSet[-self.SLIDING_WINDOW_SIZE:])
                    if score > self.THRESHOLD:
                        self.negTrainSet_lt.append(feature)
                        self.negLabels_lt.append(0)
                self.negTrainSet_st.append(feature)
                self.negLabels_st.append(0)
                # output.append((0, score,a,b,c,d))
            # delete the oldest sample if the cache is full
            if len(self.posTrainSet_st) > self.samples_st:
                self.posTrainSet_st = self.posTrainSet_st[-self.samples_st:]
                self.posLabels_st = self.posLabels_st[-self.samples_st:]
                ### delete requirements 1 ###
                # goodSamples = []
                # goodLabels = []
                # for i in range(len(self.posTrainSet[:self.FilteringNums])):
                #     score, a, b, c, d = self.getScore(self.posTrainSet[:self.FilteringNums][i], self.posTrainSet[self.FilteringNums:])
                #     if score > self.THRESHOLD:
                #         goodSamples.append(self.posTrainSet[:self.FilteringNums][i])
                #         goodLabels.append(self.posLabels[:self.FilteringNums][i])
                # self.posTrainSet = goodSamples + self.posTrainSet[self.FilteringNums:]
                # self.posLabels = goodLabels + self.posLabels[self.FilteringNums:]
                ### delete requirements 2 ###

            if len(self.negTrainSet_st) > self.samples_st:
                self.negTrainSet_st = self.negTrainSet_st[-self.samples_st:]
                self.negLabels_st = self.negLabels_st[-self.samples_st:]
            
            if len(self.posTrainSet_lt) > self.samples_lt:
                self.posTrainSet_lt = self.posTrainSet_lt[-self.samples_lt:]
                self.posLabels_lt = self.posLabels_lt[-self.samples_lt:]
                
            if len(self.negTrainSet_lt) > self.samples_lt:
                self.negTrainSet_lt = self.negTrainSet_lt[-self.samples_lt:]
                self.negLabels_lt = self.negLabels_lt[-self.samples_lt:]
            
            ### Randomly choose from long, and add to short ###
            if len(self.posTrainSet_lt) > self.samples_st:
                sampleNums = int((self.samples-self.samples_st) /(self.samples_lt-self.samples_st) * len(self.posTrainSet_lt[:(self.samples_lt-self.samples_st)]))
                self.posTrainSet = random.sample(self.posTrainSet_lt[:(self.samples_lt-self.samples_st)], sampleNums) + self.posTrainSet_st
                self.posLabels = random.sample(self.posLabels_lt[:(self.samples_lt-self.samples_st)], sampleNums) + self.posLabels_st
            else:
                self.posTrainSet = self.posTrainSet_st
                self.posLabels = self.posLabels_st
            
            if len(self.negTrainSet_lt) > self.samples_st:
                sampleNums = int((self.samples-self.samples_st) /(self.samples_lt-self.samples_st) * len(self.negTrainSet_lt[:(self.samples_lt-self.samples_st)]))
                self.negTrainSet = random.sample(self.negTrainSet_lt[:(self.samples_lt-self.samples_st)], sampleNums) + self.negTrainSet_st
                self.negLabels = random.sample(self.negLabels_lt[:(self.samples_lt-self.samples_st)], sampleNums) + self.negLabels_st
            else:
                self.negTrainSet = self.negTrainSet_st
                self.negLabels = self.negLabels_st
        
        self.nummm +=1
           

        return output

    def update_classifier(self):
        trainSet = [x[0] for x in self.posTrainSet] + [x[0] for x in self.negTrainSet]
        labels = self.posLabels + self.negLabels
        self.clf.fit(trainSet, labels)

    def predict(self, feature):
        return self.clf.predict([feature])[0]

        
        

