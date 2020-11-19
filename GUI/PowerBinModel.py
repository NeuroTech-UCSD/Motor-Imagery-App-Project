import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score

from DataProcessingHelper import pyeeg, getIntervals, getPowerRatio

# Abstract class for enabling interchangable models
class myModel:
    def __init__(self):
        pass
    def fit(X, Y):
        pass
    def predict(X):
        pass


# Power Bin Model (may have other decision algorithms)
class PowerBinModel(myModel): 

    def __init__(self, chans, num_top=5):
        
        super().__init__()
        self.mod_types = [LogisticRegression, LinearDiscriminantAnalysis]
        self.mod_type = self.mod_types[0] 
        self.binning = [1, 4, 7, 12, 30]
        self.intervals = getIntervals(self.binning)
        self.model = self.mod_type()
        self.scaler = StandardScaler()
        self.chans = chans
        self.feat_names = [str(ch) + "_" + str(ints) for ch in self.chans for ints in self.intervals]
        self.feat_names.extend([str(ch) + "_mobility" for ch in self.chans])
        self.feat_names.extend([str(ch) + "_complexity" for ch in self.chans])
        self.feat_names.extend(['left_'+ str(interval) for interval in self.intervals])
        self.feat_names.extend(['right_'+ str(interval) for interval in self.intervals])
        self.feat_names.extend(['left_minus_right_'+ str(interval) for interval in self.intervals])

        self.feature_indx = None
        self.num_top = num_top
        
    def getFeatureNames(self):
        return self.feat_names
        
    def _getFeatures(self, eeg_datas) :
        feats = []
        mobs = []
        comps = []
        left_powers = {}
        right_powers = {}
        for i, eeg_data in enumerate(eeg_datas): 
            ratios = getPowerRatio(eeg_data, self.binning)
            
            feats.extend(ratios)

            mob, comp = pyeeg.hjorth(eeg_data)
            mobs.append(mob)
            comps.append(comp)
            ch = self.chans[i]
            for i, interval in enumerate(self.intervals) : 
                if (interval not in left_powers) or (interval not in right_powers): 
                    left_powers[interval] = 0
                    right_powers[interval] = 0
                if int(ch[1]) % 2 == 0: 
                    right_powers[interval] += ratios[i]
                else : 
                    left_powers[interval] += ratios[i]
        feats.extend(mobs)
        feats.extend(comps)
        left_powers_list = [] 
        right_powers_list = []
        for i, interval in enumerate(self.intervals) : 
            left_powers_list.append(left_powers[interval])
            right_powers_list.append(right_powers[interval])
        feats.extend(left_powers_list)
        feats.extend(right_powers_list)
        feats.extend(np.array(left_powers_list) - np.array(right_powers_list))
        
        return feats
    
    def _selectTopFeatures(self, X_features, Y) :
        # Get the unique classes of Y to be able to separate the features 
        unique_Y = np.unique(Y)
        X_features_mean = []
        X_features_std = []
        for un in unique_Y: 
            X_features_mean.append(np.mean(X_features[Y == un, :], 0))
            X_features_std.append(np.std(X_features[Y == un, :], 0) / np.sqrt(len(X_features[Y == un])))
        X_features_std_sum = np.sum(X_features_std, 0)
        X_features_mean_diff = np.abs(X_features_mean[0] - X_features_mean[1])
        X_features_mean_diff = X_features_mean_diff - X_features_std_sum
        
        # Get the index of the sorted mean difference array
        args = np.argsort(X_features_mean_diff)
        # Reverse args to largest features are first
        args = args[::-1]
        feature_indx = args[:self.num_top]
        X_features = np.transpose([X_features[:, i] for i in feature_indx])
        return X_features, feature_indx

    def fit(self, X, Y):
        # X shape must be (#Trials, #Chans, #Timepoints)
        
        # Check which model is the best 
        max_accs, best_model_idx = self.evaluate(X, Y)
        print("max accuracy:", max_accs)
        print("best model index:", best_model_idx)
        self.mod_type = self.mod_types[best_model_idx]
        self.model = self.mod_type()
        
        X_features = np.array([self._getFeatures(eeg_datas) for eeg_datas in X])
        X_features = self.scaler.fit_transform(X_features)
        X_features, self.feature_indx = self._selectTopFeatures(X_features, Y)

        self.model.fit(X_features,Y)
    
    def evaluate(self, X, Y): 
        unique_Y = np.unique(Y)
        loo = LeaveOneOut()
        X_features = np.array([self._getFeatures(eeg_datas) for eeg_datas in X])
        accs = [] 
        for mod in self.mod_types: 
            y_pred = []
            y_true = []
            for train_ix, test_ix in loo.split(Y):
                # split data
                X_train_i, X_test_i = X_features[train_ix, :], X_features[test_ix, :]
                y_train_i, y_test_i = Y[train_ix], Y[test_ix]

                scaler = StandardScaler()
                X_train_i = scaler.fit_transform(X_train_i)
                X_train_i, feature_indx = self._selectTopFeatures(X_train_i, y_train_i)
                
                # fit model
                model = mod()
                model.fit(X_train_i, y_train_i)

                X_test_i = scaler.transform(X_test_i)
                X_test_i = np.transpose([X_test_i[:, i] for i in feature_indx])

                # evaluate model
                yhat = model.predict(X_test_i)
                # store
                y_true.append(y_test_i[0])
                y_pred.append(yhat[0])

            # calculate accuracy
            acc = accuracy_score(y_true, y_pred)
            accs.append(acc)
        #print(accs)
        return max(accs), accs.index(max(accs))
        
    def predict(self, X):
        # X shape must be (#Trials, #Chans, #Timepoints)
        X_features = np.array([self._getFeatures(eeg_datas) for eeg_datas in X])
        X_features = self.scaler.transform(X_features)
        X_features = np.transpose([X_features[:, i] for i in self.feature_indx])
        
        return self.model.predict(X_features)

    def predict_proba(self, X):
        # X shape must be (#Trials, #Chans, #Timepoints)
        X_features = np.array([self._getFeatures(eeg_datas) for eeg_datas in X])
        X_features = self.scaler.transform(X_features)
        X_features = np.transpose([X_features[:, i] for i in self.feature_indx])
        return self.model.predict_proba(X_features)

