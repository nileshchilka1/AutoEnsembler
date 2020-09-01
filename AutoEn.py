import numpy as np
from sklearn.linear_model import LogisticRegression,Lasso
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.svm import SVC,SVR
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,RandomForestRegressor,AdaBoostRegressor
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.metrics import accuracy_score,r2_score,confusion_matrix
import warnings 

def findCombinationsUtil(arr, index, num, reducedNum,size,unique_classes): 
    global combinations
    if (reducedNum < 0): 
        return; 


    if (reducedNum == 0): 
        comb = []
        for i in range(index): 
            comb.append(arr[i]/10)
            
        if len(comb) == size:
            comb = zero_padding(comb,unique_classes)
            combinations.append(comb)
        return; 


    prev = 1 if(index == 0) else arr[index - 1]; 

    
    for k in range(prev, num + 1): 


        arr[index] = k; 


        findCombinationsUtil(arr, index + 1, num, reducedNum - k,size,unique_classes); 
        

def findCombinations(n,size,unique_classes): 

    arr = [0] * n; 

    findCombinationsUtil(arr, 0, n, n,size,unique_classes)
    
    
def zero_padding(lst,size):
    l=len(lst)
    for i in range(size-l):
        lst.append(0)
    return lst

combinations = []
def find_all_combinations(unique_classes):
    global combinations
    n = 10
    for i in range(2,unique_classes+1):
        findCombinations(n,i,unique_classes);
        combinations += combinations
    zeros = [0] * unique_classes
    zeros[0] = 1
    combinations.append(zeros)
    
    from itertools import permutations 
    
    all_combinations = []
    for comb in combinations:
        perm = permutations(comb) 
        for i in list(perm): 
            if i not in all_combinations:
                all_combinations.append(i)
    
    return all_combinations


class AutoEnClassifier:
    
    def __init__(self,LR=True,SVC=False,RF=True,AB=False,KNN=False,random_state=0,GridSearch=False,optimize=None,scoring='accuracy'):
        
        self.__LR = LR
        self.__SVC = SVC
        self.__RF = RF
        self.__AB = AB
        self.__KNN = KNN
        self.__random_state = random_state
        self.__GridSearch = GridSearch
        self.__optimize = optimize
        if not GridSearch:
            warnings.warn('model will use RandomizedSearch')
        self.__scoring = scoring
        
        
        
    def fit(self,X_train,y_train,validation_split=0.2,validation_data=False):
       
        self.__storing_model_names = []
        self.__X_train = X_train
        self.__y_train = y_train
        if validation_data:
            self.__X_test = validation_data[0]
            self.__y_test = validation_data[1]
        else:
            self.__X_train,self.__X_test,self.__y_train,self.__y_test = train_test_split(X_train,y_train,test_size=validation_split,random_state=self.__random_state)
        
        if self.__LR:
            AutoEnClassifier.LR_model_fit(self,param_grid=None)
            self.__storing_model_names.append('LR_score')
        if self.__SVC:
            AutoEnClassifier.SVC_model_fit(self,param_grid=None)
            self.__storing_model_names.append('SVC_score')
        if self.__RF:
            AutoEnClassifier.RF_model_fit(self,param_grid=None)
            self.__storing_model_names.append('RF_score')
        if self.__AB:
            AutoEnClassifier.AB_model_fit(self,param_grid=None)
            self.__storing_model_names.append('AB_score')
        if self.__KNN:
            AutoEnClassifier.KNN_model_fit(self,list_neighbors=None)
            self.__storing_model_names.append('KNN_score')
            
        AutoEnClassifier.find_best(self)
        
        
    def LR_model_fit(self,param_grid=None):
            LR_model = LogisticRegression()
            if param_grid == None:
                parameters = {'C':[0.1,0.5,1,5,10],
                              'solver':['newton-cg', 'lbfgs', 'sag', 'saga'],
                              }
                if self.__GridSearch:
                    self.__LR_model = GridSearchCV(estimator=LR_model, param_grid=parameters, cv=5,scoring=self.__scoring,n_jobs=-1)
                else:
                    self.__LR_model = RandomizedSearchCV(estimator=LR_model, param_distributions=parameters, cv=5,scoring=self.__scoring,n_jobs=-1)
            else:
                if self.__GridSearch:
                    self.__LR_model = GridSearchCV(estimator=LR_model, param_grid=param_grid, cv=5,scoring=self.__scoring,n_jobs=-1)
                else:
                    self.__LR_model = RandomizedSearchCV(estimator=LR_model, param_distributions=param_grid, cv=5,scoring=self.__scoring,n_jobs=-1)
            self.__LR_model.fit(self.__X_train,self.__y_train)
            print(f'LR_score : {accuracy_score(self.__LR_model.predict(self.__X_test),self.__y_test)}')
            
    def SVC_model_fit(self,param_grid=None):
            SVC_model = SVC(probability=True)
            if param_grid == None:
                parameters = [{'kernel': ['rbf','poly'],
                               'gamma': [1e-3, 1e-4],
                               'C': [1, 10, 100, 1000]}]
                
                if self.__GridSearch:
                    self.__SVC_model = GridSearchCV(estimator=SVC_model, param_grid=parameters, cv=5,scoring=self.__scoring,n_jobs=-1)
                else:
                    self.__SVC_model = RandomizedSearchCV(estimator=SVC_model, param_distributions=parameters, cv=5,scoring=self.__scoring,n_jobs=-1)
            else:
                if self.__GridSearch:
                    self.__SVC_model = GridSearchCV(estimator=SVC_model, param_grid=param_grid, cv=5,scoring=self.__scoring,n_jobs=-1)
                else:
                    self.__SVC_model = RandomizedSearchCV(estimator=SVC_model, param_distributions=param_grid, cv=5,scoring=self.__scoring,n_jobs=-1)
            self.__SVC_model.fit(self.__X_train,self.__y_train)
            print(f'SVC_score : {accuracy_score(self.__SVC_model.predict(self.__X_test),self.__y_test)}')
            
            
    def RF_model_fit(self,param_grid=None):
            RF_model = RandomForestClassifier()
            if param_grid == None:
                parameters = {'n_estimators' :[10,50,100,500],
                              'max_depth' : [4,8,10,12,16],
                              'min_samples_leaf' : [0.1, 0.2, 0.3, 0.4, 0.5]
                              }
                
                if self.__GridSearch:
                    self.__RF_model = GridSearchCV(estimator=RF_model, param_grid=parameters, cv=5,scoring=self.__scoring,n_jobs=-1)
                else:
                    self.__RF_model = RandomizedSearchCV(estimator=RF_model, param_distributions=parameters, cv=5,scoring=self.__scoring,n_jobs=-1)
            else:
                if self.__GridSearch:
                    self.__RF_model = GridSearchCV(estimator=RF_model, param_grid=param_grid, cv=5,scoring=self.__scoring,n_jobs=-1)
                else:
                    self.__RF_model = RandomizedSearchCV(estimator=RF_model, param_distributions=param_grid, cv=5,scoring=self.__scoring,n_jobs=-1)
                
            self.__RF_model.fit(self.__X_train,self.__y_train)
            print(f'RF_score : {accuracy_score(self.__RF_model.predict(self.__X_test),self.__y_test)}')
            
            
    def AB_model_fit(self,param_grid=None):
            AB_model = AdaBoostClassifier()
            if param_grid == None:
                parameters = {'n_estimators' :[10,50,100,500],
                              'learning_rate' : [0.01,0.5,0.1,0.15,0.2],
                              }
                
                if self.__GridSearch:
                    self.__AB_model = GridSearchCV(estimator=AB_model, param_grid=parameters, cv=5,scoring=self.__scoring,n_jobs=-1)
                else:
                    self.__AB_model = RandomizedSearchCV(estimator=AB_model, param_distributions=parameters, cv=5,scoring=self.__scoring,n_jobs=-1)
            else:
                if self.__GridSearch:
                    self.__AB_model = GridSearchCV(estimator=AB_model, param_grid=param_grid, cv=5,scoring=self.__scoring,n_jobs=-1)
                else:
                    self.__AB_model = RandomizedSearchCV(estimator=AB_model, param_distributions=param_grid, cv=5,scoring=self.__scoring,n_jobs=-1)
                
            self.__AB_model.fit(self.__X_train,self.__y_train)
            print(f'AB_score : {accuracy_score(self.__AB_model.predict(self.__X_test),self.__y_test)}')
            
        
    
    def KNN_model_fit(self,list_neighbors=None):
            if list_neighbors == None:
                list_neighbors = [3,5,7,9,11,13,15]
                n_neighbor_score_model = [None,0,None] 
                for neighbor in list_neighbors:
                    self.__KNN_model = KNeighborsClassifier(n_neighbors=neighbor)
                    self.__KNN_model = self.__KNN_model.fit(self.__X_train,self.__y_train)
                    model_score = self.__KNN_model.score(self.__X_test,self.__y_test)
                    if model_score > n_neighbor_score_model[1]:
                        n_neighbor_score_model[0] = neighbor
                        n_neighbor_score_model[1] = model_score
                        n_neighbor_score_model[2] = self.__KNN_model
                
                self.__KNN_model = n_neighbor_score_model[2]
                y_predict = self.__KNN_model.predict_proba(self.__X_test)
                y_predict = np.argmax(y_predict,axis=1)
                print(f'KNN_score with {n_neighbor_score_model[0]} neighbors: {accuracy_score(self.__y_test,y_predict)}')
                
                
                
    def find_best(self):
        
        global combinations
        combinations = []
        Total_models = self.__LR + self.__SVC + self.__RF + self.__KNN + self.__AB
        
        combinations = find_all_combinations(Total_models)
        
        self.__best_score = [0] + [None] * Total_models
        count = 1
        flag_loop = 0
        
        for comb in combinations: 
                all_proba = []
                if self.__LR:
                    if flag_loop == 0:
                        LR_loc = count
                    LR_model_y_predict_proba = self.__LR_model.predict_proba(self.__X_test)
                    LR_model_y_predict_proba = np.multiply(LR_model_y_predict_proba,comb[LR_loc-1])
                    all_proba.append(LR_model_y_predict_proba)
                    if self.__best_score[LR_loc] == None:
                        count += 1
                        
                if self.__SVC:
                    if flag_loop == 0:
                        SVC_loc = count
                    SVC_model_y_predict_proba = self.__SVC_model.predict_proba(self.__X_test)
                    SVC_model_y_predict_proba = np.multiply(SVC_model_y_predict_proba,comb[SVC_loc-1])
                    all_proba.append(SVC_model_y_predict_proba)
                    if self.__best_score[SVC_loc] == None:
                        count += 1
                    
                if self.__RF:
                    if flag_loop == 0:
                        RF_loc = count
                    RF_model_y_predict_proba = self.__RF_model.predict_proba(self.__X_test)
                    RF_model_y_predict_proba = np.multiply(RF_model_y_predict_proba,comb[RF_loc-1])
                    all_proba.append(RF_model_y_predict_proba)
                    if self.__best_score[RF_loc] == None:
                        count += 1
                    
                if self.__AB:
                    if flag_loop == 0:
                        AB_loc = count
                    AB_model_y_predict_proba = self.__AB_model.predict_proba(self.__X_test)
                    AB_model_y_predict_proba = np.multiply(AB_model_y_predict_proba,comb[AB_loc-1])
                    all_proba.append(AB_model_y_predict_proba)
                    if self.__best_score[AB_loc] == None:
                        count += 1
                
                if self.__KNN:
                    if flag_loop == 0:
                        KNN_loc = count
                    KNN_model_y_predict_proba = self.__KNN_model.predict_proba(self.__X_test)
                    KNN_model_y_predict_proba = np.multiply(KNN_model_y_predict_proba,comb[KNN_loc-1])
                    all_proba.append(KNN_model_y_predict_proba)
                    if self.__best_score[KNN_loc] == None:
                        count += 1
                    
                y_predict = np.sum(all_proba,axis=0)
                
                y_predict = np.argmax(y_predict,axis=1)
                
                
                latest_score = accuracy_score(self.__y_test,y_predict)
                
                
                if latest_score > self.__best_score[0]:
                    if flag_loop==0 and self.__optimize == 'FP':
                        optimize_count = confusion_matrix(self.__y_test,y_predict)[1][0]
                        
                    if flag_loop==0 and self.__optimize == 'FN':
                        optimize_count = confusion_matrix(self.__y_test,y_predict)[0][1]
                        
                    self.__best_score[0] = latest_score
                    for i in range(0,len(comb)):
                        self.__best_score[i+1] = comb[i]
                        
                elif latest_score == self.__best_score[0] and self.__optimize == 'FP':
                    
                    FP_count = confusion_matrix(self.__y_test,y_predict)[1][0]
                    if FP_count < optimize_count:
                        print(f'optimized FP from {optimize_count} to {FP_count}')
                        optimize_count = FP_count
                        self.__best_score[0] = latest_score
                    for i in range(0,len(comb)):
                        self.__best_score[i+1] = comb[i]
                        
                elif latest_score == self.__best_score[0] and self.__optimize == 'FN':
                    
                    FN_count = confusion_matrix(self.__y_test,y_predict)[0][1]
                    if FN_count < optimize_count:
                        print(f'optimized FN from {optimize_count} to {FN_count}')
                        optimize_count = FN_count
                        self.__best_score[0] = latest_score
                    for i in range(0,len(comb)):
                        self.__best_score[i+1] = comb[i]
                
                        
                
                if flag_loop == 0:
                    flag_loop = 1
                    
        print(f'AutoEn_score : {self.__best_score[0]}')
        for i in range(len(self.__storing_model_names)):
            
            print(f'weight for {self.__storing_model_names[i]} : {self.__best_score[i+1]}')
            
        
    def predict(self,X_test):
        all_proba = []
        count = 1
        try:
        
            if self.__LR:
                LR_model_y_predict_proba = self.__LR_model.predict_proba(X_test)
                LR_model_y_predict_proba = np.multiply(LR_model_y_predict_proba,self.__best_score[count])
                all_proba.append(LR_model_y_predict_proba)
                count+=1
                
            if self.__SVC:
                SVC_model_y_predict_proba = self.__SVC_model.predict_proba(X_test)
                SVC_model_y_predict_proba = np.multiply(SVC_model_y_predict_proba,self.__best_score[count])
                all_proba.append(SVC_model_y_predict_proba)
                count+=1
        
            if self.__RF:
                RF_model_y_predict_proba = self.__RF_model.predict_proba(X_test)
                RF_model_y_predict_proba = np.multiply(RF_model_y_predict_proba,self.__best_score[count])
                all_proba.append(RF_model_y_predict_proba)
                count+=1
            
            if self.__AB:
                AB_model_y_predict_proba = self.__AB_model.predict_proba(X_test)
                AB_model_y_predict_proba = np.multiply(AB_model_y_predict_proba,self.__best_score[count])
                all_proba.append(AB_model_y_predict_proba)
                count+=1
                
            if self.__KNN:
                KNN_model_y_predict_proba = self.__KNN_model.predict_proba(X_test)
                KNN_model_y_predict_proba = np.multiply(KNN_model_y_predict_proba,self.__best_score[count])
                all_proba.append(KNN_model_y_predict_proba)
                count+=1
                
            y_predict = np.sum(all_proba,axis=0)            
           
            
        except AttributeError:
            print('model not fitted yet')
            return None
        
        
        y_predict = np.argmax(y_predict,axis=1)

        
        return y_predict
    
class AutoEnRegressor:
    def __init__(self,LR=True,SVR=False,RF=True,AB=False,KNN=False,random_state=0,GridSearch=False,scoring='r2'):
        
        
        self.__LR = LR
        self.__SVR = SVR
        self.__RF = RF
        self.__AB = AB
        self.__KNN = KNN
        self.__random_state = random_state
        self.__GridSearch = GridSearch
        if not GridSearch:
            warnings.warn('model will use RandomizedSearch')
        self.__scoring = scoring
        
            

        
    def fit(self,X_train,y_train,validation_split=0.2,validation_data=False):
       
        self.__storing_model_names = []
        self.__X_train = X_train
        self.__y_train = y_train
        if validation_data:
            self.__X_test = validation_data[0]
            self.__y_test = validation_data[1]
        else:
            self.__X_train,self.__X_test,self.__y_train,self.__y_test = train_test_split(X_train,y_train,test_size=validation_split,random_state=self.__random_state)
        
        if self.__LR:
            AutoEnRegressor.LR_model_fit(self,param_grid=None)
            self.__storing_model_names.append('LR_score')
        if self.__SVR:
            AutoEnRegressor.SVR_model_fit(self,param_grid=None)
            self.__storing_model_names.append('SVR_score')
        if self.__RF:
            AutoEnRegressor.RF_model_fit(self,param_grid=None)
            self.__storing_model_names.append('RF_score')
        if self.__AB:
            AutoEnRegressor.AB_model_fit(self,param_grid=None)
            self.__storing_model_names.append('AB_score')
        if self.__KNN:
            AutoEnRegressor.KNN_model_fit(self,list_neighbors=None)
            self.__storing_model_names.append('KNN_score')
            
        AutoEnRegressor.find_best(self)
        
        
    def LR_model_fit(self,param_grid=None):
            LR_model = Lasso()
            if param_grid == None:
                parameters = {'alpha':[0.01,0.5,1,2,5]
                              }
                if self.__GridSearch:
                    self.__LR_model = GridSearchCV(estimator=LR_model, param_grid=parameters, cv=5,scoring=self.__scoring,n_jobs=-1)
                else:
                    self.__LR_model = RandomizedSearchCV(estimator=LR_model, param_distributions=parameters, cv=5,scoring=self.__scoring,n_jobs=-1)
            else:
                if self.__GridSearch:
                    self.__LR_model = GridSearchCV(estimator=LR_model, param_grid=param_grid, cv=5,scoring=self.__scoring,n_jobs=-1)
                else:
                    self.__LR_model = RandomizedSearchCV(estimator=LR_model, param_distributions=param_grid, cv=5,scoring=self.__scoring,n_jobs=-1)
            self.__LR_model.fit(self.__X_train,self.__y_train)
            print(f'LR_score : {r2_score(self.__y_test,self.__LR_model.predict(self.__X_test))}')
            
    def SVR_model_fit(self,param_grid=None):
            SVR_model = SVR()
            if param_grid == None:
                parameters = [{'kernel': ['rbf','poly'],
                               'gamma': [1e-3, 1e-4],
                               'C': [1, 10, 100, 1000]}]
                
                if self.__GridSearch:
                    self.__SVR_model = GridSearchCV(estimator=SVR_model, param_grid=parameters, cv=5,scoring=self.__scoring,n_jobs=-1)
                else:
                    self.__SVR_model = RandomizedSearchCV(estimator=SVR_model, param_distributions=parameters, cv=5,scoring=self.__scoring,n_jobs=-1)
            else:
                if self.__GridSearch:
                    self.__SVR_model = GridSearchCV(estimator=SVR_model, param_grid=param_grid, cv=5,scoring=self.__scoring,n_jobs=-1)
                else:
                    self.__SVR_model = RandomizedSearchCV(estimator=SVR_model, param_distributions=param_grid, cv=5,scoring=self.__scoring,n_jobs=-1)
            self.__SVR_model.fit(self.__X_train,self.__y_train)
            print(f'SVR_score : {r2_score(self.__y_test,self.__SVR_model.predict(self.__X_test))}')
            
            
    def RF_model_fit(self,param_grid=None):
            RF_model = RandomForestRegressor()
            if param_grid == None:
                parameters = {'n_estimators' :[10,50,100,500],
                              'max_depth' : [4,8,10,12,16],
                              'min_samples_leaf' : [0.1, 0.2, 0.3, 0.4, 0.5]
                              }
                
                if self.__GridSearch:
                    self.__RF_model = GridSearchCV(estimator=RF_model, param_grid=parameters, cv=5,scoring=self.__scoring,n_jobs=-1)
                else:
                    self.__RF_model = RandomizedSearchCV(estimator=RF_model, param_distributions=parameters, cv=5,scoring=self.__scoring,n_jobs=-1)
            else:
                if self.__GridSearch:
                    self.__RF_model = GridSearchCV(estimator=RF_model, param_grid=param_grid, cv=5,scoring=self.__scoring,n_jobs=-1)
                else:
                    self.__RF_model = RandomizedSearchCV(estimator=RF_model, param_distributions=param_grid, cv=5,scoring=self.__scoring,n_jobs=-1)
                
            self.__RF_model.fit(self.__X_train,self.__y_train)
            print(f'RF_score : {r2_score(self.__y_test,self.__RF_model.predict(self.__X_test))}')
            
            
    def AB_model_fit(self,param_grid=None):
            AB_model = AdaBoostRegressor()
            if param_grid == None:
                parameters = {'n_estimators' :[10,50,100,500],
                              'learning_rate' : [0.01,0.5,0.1,0.15,0.2],
                              }
                
                if self.__GridSearch:
                    self.__AB_model = GridSearchCV(estimator=AB_model, param_grid=parameters, cv=5,scoring=self.__scoring,n_jobs=-1)
                else:
                    self.__AB_model = RandomizedSearchCV(estimator=AB_model, param_distributions=parameters, cv=5,scoring=self.__scoring,n_jobs=-1)
            else:
                if self.__GridSearch:
                    self.__AB_model = GridSearchCV(estimator=AB_model, param_grid=param_grid, cv=5,scoring=self.__scoring,n_jobs=-1)
                else:
                    self.__AB_model = RandomizedSearchCV(estimator=AB_model, param_distributions=param_grid, cv=5,scoring=self.__scoring,n_jobs=-1)
                
            self.__AB_model.fit(self.__X_train,self.__y_train)
            print(f'AB_score : {r2_score(self.__y_test,self.__AB_model.predict(self.__X_test))}')
            
        
    
    def KNN_model_fit(self,list_neighbors=None):
            if list_neighbors == None:
                list_neighbors = [3,5,7,9,11,13,15]
                n_neighbor_score_model = [None,0,None] 
                for neighbor in list_neighbors:
                    self.__KNN_model = KNeighborsRegressor(n_neighbors=neighbor)
                    self.__KNN_model = self.__KNN_model.fit(self.__X_train,self.__y_train)
                    model_score = self.__KNN_model.score(self.__X_test,self.__y_test)
                    if model_score > n_neighbor_score_model[1]:
                        n_neighbor_score_model[0] = neighbor
                        n_neighbor_score_model[1] = model_score
                        n_neighbor_score_model[2] = self.__KNN_model
                
                self.__KNN_model = n_neighbor_score_model[2]
                y_predict = self.__KNN_model.predict(self.__X_test)
                print(f'KNN_score with {n_neighbor_score_model[0]} neighbors: {r2_score(self.__y_test,y_predict)}')
                
                
                
    def find_best(self):
        
        global combinations
        combinations = []
        Total_models = self.__LR + self.__SVR + self.__RF  + self.__AB + self.__KNN
        
        combinations = find_all_combinations(Total_models)
        
        
        self.__best_score = [0] + [None] * Total_models
        count = 1
        flag_loop = 0
        
        for comb in combinations: 
                all_proba = []
                if self.__LR:
                    if flag_loop == 0:
                        LR_loc = count
                    LR_model_y_predict = self.__LR_model.predict(self.__X_test)
                    LR_model_y_predict = np.multiply(LR_model_y_predict,comb[LR_loc-1])
                    all_proba.append(LR_model_y_predict)
                    if self.__best_score[LR_loc] == None:
                        count += 1
                        
                if self.__SVR:
                    if flag_loop == 0:
                        SVR_loc = count
                    SVR_model_y_predict = self.__SVR_model.predict(self.__X_test)
                    SVR_model_y_predict = np.multiply(SVR_model_y_predict,comb[SVR_loc-1])
                    all_proba.append(SVR_model_y_predict)
                    if self.__best_score[SVR_loc] == None:
                        count += 1
                    
                if self.__RF:
                    if flag_loop == 0:
                        RF_loc = count
                    RF_model_y_predict = self.__RF_model.predict(self.__X_test)
                    RF_model_y_predict = np.multiply(RF_model_y_predict,comb[RF_loc-1])
                    all_proba.append(RF_model_y_predict)
                    if self.__best_score[RF_loc] == None:
                        count += 1
                    
                if self.__AB:
                    if flag_loop == 0:
                        AB_loc = count
                    AB_model_y_predict = self.__AB_model.predict(self.__X_test)
                    AB_model_y_predict = np.multiply(AB_model_y_predict,comb[AB_loc-1])
                    all_proba.append(AB_model_y_predict)
                    if self.__best_score[AB_loc] == None:
                        count += 1
                
                if self.__KNN:
                    if flag_loop == 0:
                        KNN_loc = count
                    KNN_model_y_predict = self.__KNN_model.predict(self.__X_test)
                    KNN_model_y_predict = np.multiply(KNN_model_y_predict,comb[KNN_loc-1])
                    all_proba.append(KNN_model_y_predict)
                    if self.__best_score[KNN_loc] == None:
                        count += 1
                    
                y_predict = np.sum(all_proba,axis=0)
                
                          
                latest_score = r2_score(self.__y_test,y_predict)
          
                if latest_score > self.__best_score[0]:
                    self.__best_score[0] = latest_score
                    for i in range(0,len(comb)):
                        self.__best_score[i+1] = comb[i]
                        
                
                if flag_loop == 0:
                    flag_loop = 1
                    
        print(f'AutoEn_score : {self.__best_score[0]}')
        for i in range(len(self.__storing_model_names)):
            
            print(f'weight for {self.__storing_model_names[i]} : {self.__best_score[i+1]}')
            
        
    def predict(self,X_test):
        all_proba = []
        count = 1
        try:
        
            if self.__LR:
                LR_model_y_predict = self.__LR_model.predict(X_test)
                LR_model_y_predict = np.multiply(LR_model_y_predict,self.__best_score[count])
                all_proba.append(LR_model_y_predict)
                count+=1
                
            if self.__SVR:
                SVR_model_y_predict = self.__SVR_model.predict(X_test)
                SVR_model_y_predict = np.multiply(SVR_model_y_predict,self.__best_score[count])
                all_proba.append(SVR_model_y_predict)
                count+=1
        
            if self.__RF:
                RF_model_y_predict = self.__RF_model.predict(X_test)
                RF_model_y_predict = np.multiply(RF_model_y_predict,self.__best_score[count])
                all_proba.append(RF_model_y_predict)
                count+=1
            
            if self.__AB:
                AB_model_y_predict = self.__AB_model.predict(X_test)
                AB_model_y_predict = np.multiply(AB_model_y_predict,self.__best_score[count])
                all_proba.append(AB_model_y_predict)
                count+=1
                
            if self.__KNN:
                KNN_model_y_predict = self.__KNN_model.predict(X_test)
                KNN_model_y_predict = np.multiply(KNN_model_y_predict,self.__best_score[count])
                all_proba.append(KNN_model_y_predict)
                count+=1
                
            y_predict = np.sum(all_proba,axis=0)          
           
            
        except AttributeError:
            print('model not fitted yet')
            return None

        
        return y_predict
        
            
        







