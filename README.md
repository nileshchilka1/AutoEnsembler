# AutoEnsembler

### Overview
   This is an AutoEnsembler package, it helps you to find best ensemble model for Classification and Regression problem. As we know that every model has its own speciality, by assuming this, I built AutoEnClassifer and AutoEnRegressor on top of LogisticRegression/Lasso, SVC/SVR, RandomForestClassifier/RandomForestRegressor, AdaBoostClassifer/AdaBoostRegressor and KNeighborsClassifier/KNeighborsRegressor.
        
        
### Uniqueness
- In AutoEnClassifer, you can pass a parameter that you want to optimize i.e. 'FN' / 'FP'
- While training, by default it will split the data into training data and validation data by 0.2(you can also specify) and it will show you the accuracy_score/r2_score(with respect to each model you selected and of AutoEnClassifer/AutoEnRegressor) on validation data
- while initiating the model you can also specifiy which models should be used for ensembling and what type of search you want to use viz. GridSearchCV/RandomizedSearchCV
        
### Motivation 
   I participated in various competitions and I learned many things from it. I wanted to share my knowledge to this amazing field of Data Science and Machine Learning. If you are beginner in this field then try to compete this model with your model.

### Installation

```markdown
 pip install AutoEnsembler
```
### How to use
   After installing, you can import as shown below. By default LogisticRegression/Lasso and RandomForestClassifier/RandomForestRegressor is selected. while fitting the model I passed 0.25 as validation_split(by default is 0.2). You can also see the accuracy_score/r2_score of individual model and of AutoEn model on validation_split data. And you can also see the weight for individual models for prediction.
   
![Screenshot1](https://github.com/nileshchilka1/AutoEnsembler/blob/master/Screenshot%201.png)
   
   
   Here, I specified to use all the models by passing True as a parameter with respect to each model name. By default False parameter is passed to GridSearch and it will use RandomizedSearchCV and also you can see in warnings.
   
![Screenshot1](https://github.com/nileshchilka1/AutoEnsembler/blob/master/Screenshot.png)


   Now, I enabled GridSearch and I passed own validation data to it and now the score will compute on validation_data.
   
![Screenshot1](https://github.com/nileshchilka1/AutoEnsembler/blob/master/Screenshot%202.png)
