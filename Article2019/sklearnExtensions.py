
import numpy as np;

from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import issparse


class SumarizeTransformer(BaseEstimator):
    def __init__(self, agg = None):
        self.agg = agg

    def transform(self, X, y=None):
        if self.agg == 'min':
            return X.min(axis=1,keepdims=True);
        if self.agg == 'max':
            return X.max(axis=1,keepdims=True);
        if self.agg == 'mean':
            return X.mean(axis=1,keepdims=True);
        if self.agg is not None:
            return self.agg(X);
        return X;

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X=X, y=y)
    
    def fit_predict(self,X,y=None):
        return self.transform(X=X, y=y);
    
    def predict(self,X,y=None):
        return self.transform(X=X, y=y)
    
    
from sklearn.base import ClusterMixin, BaseEstimator
from sklearn.utils.validation import check_random_state

class DistanceNoveltyClassifier(BaseEstimator,ClusterMixin):
    
    def __init__(self,classes, clip=(1,99), random_state=None):
        self.random_state = random_state;
        self.classes = classes;
        self.clip =clip
        
    def _buildProfile(self,X,y):
        #building the profile matrix
        profiles_ = np.zeros((len(self.classes), X.shape[1]),dtype=np.float32);
        
        for i,p in enumerate(self.classes):
            profiles_[i,] += np.array(X[y==p,].sum(axis=0)).flatten();
        
        self.profiles_ = profiles_;
        
    
    def fit(self,X,y):
        y = np.array(y)
        
        self._buildProfile(X,y);
        
        #building the distance matrix
        self.metrics_ = ['cosine',fuzzyyulef];

        distances  =np.hstack([
            pairwise_distances(X,self.profiles_,metric=d).min(axis=1).reshape(-1,1)
            for d in self.metrics_
        ])
        
        inSet = np.array([1 if yi in self.classes else 0 for yi in y]);
        
        #downsampling because the out of classes is larger
        distances,inSet = upsampling(distances,inSet);
        
        self.clf_  = Pipeline([
            ('clip'  ,ClipTransformer(a_min=self.clip[0],a_max=self.clip[1])),
            ('transf',preprocessing.MaxAbsScaler()),
            ('clf'   ,linear_model.LogisticRegression(C=1,random_state=self.random_state,class_weight='balanced'))
        ]);
        self.clf_.fit(distances,inSet);
        
        return self;
            
    def predict(self,X):
        distances  =np.hstack([
            pairwise_distances(X,self.profiles_,metric=d).min(axis=1).reshape(-1,1)
            for d in self.metrics_
        ])
        
        return self.clf_.predict(distances);
    
    def predict_proba(self,X):
        distances  =np.hstack([
            pairwise_distances(X,self.profiles_,metric=d).min(axis=1).reshape(-1,1)
            for d in self.metrics_
        ])
        return self.clf_.predict_proba(distances);
    
    def fit_predict(self,X,y):
        self.fit(X,y);
        return self.predict(X);
    
    

from sklearn.base import ClusterMixin, BaseEstimator
from sklearn.utils.validation import check_random_state

class OneClassClassifier(BaseEstimator):
    def __init__(self,n_estimators=10,noise_proportion=0.1, random_state=None):
        self.n_estimators=n_estimators;
        self.noise_proportion=noise_proportion;
        self.random_state = random_state;
        
    def fit(self,X,y=None):
        random_state = check_random_state(self.random_state);
        self.estimators = [];
        l = X.shape[0] or len(X);
        y = np.ones(int(l*self.noise_proportion));
        y = np.hstack([y,np.zeros(l-len(y))]);
        
        for _ in range(self.n_estimators):
            est = linear_model.LogisticRegression(C=0.01,random_state=self.random_state);
            random_state.shuffle(y);
            est.fit(X,y);
            self.estimators.append(est);
        return self;
            
    def predict(self,X):
        pred = np.vstack([
            p.predict(X)[:,1] for p in self.estimators
        ]);
        return pred.sum(axis=0);
    
    def predict_proba(self,X):
        pred = np.vstack([
            p.predict_proba(X)[:,1] for p in self.estimators
        ]);
        return pred.sum(axis=0);
    
    def fit_predict(self,X):
        self.fit(X);
        return self.predict(X);


class ClipTransformer(BaseEstimator):
    def __init__(self, a_min=1,a_max=99):
        self.a_min = a_min
        self.a_max = a_max
        self.axis=1

    def transform(self, X, y=None):
        X = X.copy();
        for i in range(X.shape[self.axis]):
            X[:,i] = np.clip(X[:,i],self.min_[i],self.max_[i]);
        return X;

    def fit(self, X, y=None):
        self.min_ = np.zeros(X.shape[self.axis]);
        self.max_ = np.zeros(X.shape[self.axis]);
        
        for i in range(X.shape[self.axis]):
            self.min_[i], self.max_[i] = np.percentile(X[:,i],q=(self.a_min, self.a_max))
            
        return self

    def fit_transform(self, X, y=None):
        self.fit(X);
        
        return self.transform(X=X, y=y)


class DenseTransformer(BaseEstimator):
    """Convert a sparse array into a dense array."""

    def __init__(self, return_copy=True):
        self.return_copy = return_copy
        self.is_fitted = False

    def transform(self, X, y=None):
        """ Return a dense version of the input array.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples] (default: None)
        Returns
        ---------
        X_dense : dense version of the input X array.
        """
        if issparse(X):
            return X.toarray()
        elif self.return_copy:
            return X.copy()
        else:
            return X

    def fit(self, X, y=None):
        self.is_fitted = True
        return self

    def fit_transform(self, X, y=None):
        """ Return a dense version of the input array.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples] (default: None)
        Returns
        ---------
        X_dense : dense version of the input X array.
        """
        return self.transform(X=X, y=y)