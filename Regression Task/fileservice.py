import os
import pandas as pd
from config import global_config

class CSVService():
    """
    This handle file reading from CSV service and file writing to CSV service
    doRead(): This method read the data from the CSV file
    doWrite(): This method writes the data to a CSV Format
    """
    
    def __init__(self, input_file_path = "", output_file_path = "", delimiter=",", encoding="latin_1", schema_map=None, 
                 root_path = "", **kwargs):
        
        if input_file_path == "" and output_file_path == "":
          raise ValueError("Either input_file_path or reading_file_path should be passed as a parameter")
        self.input_path = os.path.join(root_path, input_file_path)
        self.output_path = os.path.join(root_path, output_file_path)
        self.delimiter = delimiter
        self.encoding = encoding
        self.schema_map = schema_map
        self.kwargs = kwargs['new_cols']

    def doRead(self, **kwargs):
        try:
            df = pd.read_csv(filepath_or_buffer=self.input_path, encoding=self.encoding, delimiter=self.delimiter, **kwargs)  
        except IOError:
            print('Input file not found in the given path: ' , self.input_path)
            print('Please verify the given path for input file again!')
        else:
            del_cols = self.kwargs['del_columns']
            category_cols = self.kwargs['category_columns']
            numeric_columns = self.kwargs['numeric_columns']
            target_column = self.kwargs['target_column']
            new_cols_dict = dict(del_cols, **category_cols, **numeric_columns, **target_column)

            df = df.rename(columns = new_cols_dict)
            print("CSV Service Read from File: " + str(self.input_path))
            return df

    def doWrite(self, X):
        X.to_csv(path_or_buf=self.output_path, encoding=self.encoding, sep=self.delimiter, **self.kwargs)
        print("CSV Service Output to File: " + str(self.path))
 
