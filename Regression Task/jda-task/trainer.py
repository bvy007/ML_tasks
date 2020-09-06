from fileservice import CSVService
from config import global_config as glob
from config.yaml_config import model_parameters
from transformers.custom_transformers import *
from model.estimator import ensembleRegressor

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline



class Trainer():
    """  Class to handle the training process and to view the performance of it  
    
    _load_data() : loads the input data
    _data_test_train_split(): splits the given input data into train and test set
    _pipeline(): Initialize the pipeline with transformers and a model in it
    fit(): fits the data with respect to the pipeline
    predict(): predicts the count_of_rentals data with respect to the given model in pipeline
    fit_predict(): It calls the fit() and followed by predict()
    score(): returns the r square coefficient and MAD score  """
    
    def __init__(self):
        self.bike_data = self._load_data()
        self.X_train, self.X_test, self.y_train, self.y_test = self._data_test_train_split(data = self.bike_data)
        self.pipe = self._pipeline()
        
    def _load_data(self):
        input_file = model_parameters['arguments']['csvtables']['input_file_name']
        output_file = model_parameters['arguments']['csvtables']['output_file_name']
        csv = CSVService(input_file_path = input_file,
                         output_file_path = output_file,
                         root_path = glob.package_dir, new_cols = model_parameters['arguments']['data_columns'])
        bike_data = csv.doRead()
        return bike_data
    
    def _data_test_train_split(self, data):
        features = [item for item in data.columns if item != 'total_rental_bikes']
        X_train, X_test, y_train, y_test = train_test_split(data[features], 
                                                    data['total_rental_bikes'], 
                                                    test_size=0.25, 
                                                    random_state=42)
        
        return X_train, X_test, y_train, y_test
    
    def _pipeline(self):
        
        mlpipeline = Pipeline([('cleaning', data_cleaner(new_column_names = model_parameters['arguments']['data_columns'])),
                     ('onc', one_hot_encoder()),
                     ('regressor', ensembleRegressor(max_depth=4, random_state=42,  n_estimators=50))
                    ])
        return mlpipeline
    
    def fit(self):
        return self.pipe.fit(self.X_train, self.y_train)
    
    def predict(self, model, X):
        model
        predicted_values = model.predict(X)
        
        return predicted_values

    def fit_predict(self):
        print('\n')
        print('Starting to fit the data with the selected model')
        model = self.fit()
        print('Fitting the data with selected model has completed successfully')
        print('Predictions over the test data has started!')
        predictions = self.predict(model, self.X_test)
        print('Predictions over the test data is done!')
        return predictions
    
    def score(self):
        model = self.fit()
        predictions = self.predict(model, self.X_test)
        print('\n')
        print('R-squared and MAD scores are as following:')
        r2_value_train = model.score(self.X_train, self.y_train)
        print("R-Squared on train dataset = {}".format(r2_value_train))
        r2_value_test = model.score(self.X_test, self.y_test)
        print("R-Squared on test dataset = {}".format(r2_value_test))
        mean_absolute_deviations = model['regressor'].maescore(self.y_test, predictions)
        print("Mean Absolute Deviation on test dataset = {}".format(mean_absolute_deviations))
        return r2_value_train, r2_value_test, mean_absolute_deviations
        
        
        