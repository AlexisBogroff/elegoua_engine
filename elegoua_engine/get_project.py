import random
import os
from kaggle.api import KaggleApi
import zipfile

"""
Define a project adapted for the student, manage project databases, manage project questions
"""
def eval_python_level(student_answer:dict):
    """
    Provide the student's level
    """
    student_level = student_answer['auto_eval_level']
    return student_level


def eval_student_interests(student_answer):
    """
    Provide the student's subject related to his interests
    """
    all_interests = []
    all_interests.extend(student_answer['main_interest'])
    all_interests.extend(student_answer['other_interests'])
    chosen_interest = random.choice(all_interests)
    return chosen_interest


class Project:
    """ Analyze student answer to provide adapted project (db and questions) """
    
    def __init__(self, student_answer:dict, student_interests:list=None, student_level:str=""):
        """
        Example student answer:
        ```python
        student_answer = {
            'level': {'auto_eval_level': 1},
            'interests': {
                'main_interest': 'ecology',
                'other_interests': ['rse', 'finance'],
            },
        }
        ```
        """
        self.student_answer = student_answer
        self.student_interests = student_interests
        self.student_level = student_level

    def define_project(self):
        """ Get the project dbs and questions """
        self.eval_student_answer()
        level = self.student_level
        subject = self.student_subject
        dataset = self.select_dataset(subject, level)
        return level,subject,dataset

    def eval_student_answer(self):
        """ Evaluate the student answer """
        student_level = self.student_answer['level']
        student_interests = self.student_answer['interests']
        self.student_level = eval_python_level(student_level)
        self.student_subject = eval_student_interests(student_interests)
    

    def select_dataset(self,subject,level):
        """ Select the dataset based on the subject and level
        The dataset from kaggle api is located in the frontend path : webapp/data/datasets
        If no csv are found in the first zip file, a new zip file is downloaded the function that does
        not stop until a csv file is found. At the the function returns the name of the csv file and all of the csv files and zip that were used to find the dataset are deleted."""
        api = KaggleApi()
        api.authenticate()
        # search for datasets based on keyword, file type, and tag id
        search_results = api.dataset_list(search=subject, tag_ids=level, min_size=200000, max_size=200000000) # api request to get all datasets from kaggle api related to the subject and level
        num_datasets = len(search_results)
        path = os.path.abspath(os.path.join("webapp", "data", "datasets")) # path to the datasets folder in the frontend
        if num_datasets == 0:
            print("No datasets found.")
            return
        random_index = random.randint(0, num_datasets - 1)
        random_dataset = search_results[random_index] # we select a random dataset from the search results
        api.dataset_download_files(random_dataset.ref, path=path, quiet=False) # we download the csv files from the kaggle api
        # extract the downloaded files
        max_size = 0
        dataset_file = None
        for file in os.listdir(path):
            zip_file_path = os.path.abspath(os.path.join(path, file))
            if file.endswith(".zip"):
                with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
                    zip_ref.extractall(path=path) # extract all files of the zip file
                    for csv_file in zip_ref.namelist():
                        if csv_file.endswith(".csv"):      
                            print(csv_file)                    
                            csv_path = os.path.abspath(os.path.join(path, csv_file))    
                            csv_size = os.stat(csv_path).st_size
                            # check if the csv file is larger than the max_size to get the dataset file
                            if csv_size > max_size:
                                max_size = csv_size
                                dataset_file = csv_file
                        else:
                            os.remove(os.path.abspath(os.path.join(path, csv_file))) # delete the csv file if it is not the dataset file
                os.remove(zip_file_path) # delete the zip file
                if dataset_file is not None:
                    break # found a CSV file, exit loop
        if dataset_file is None:
            print("No CSV files found.")
            return
        print(f"Extracted CSV file: {dataset_file}")
        return dataset_file


    def select_questions(self):
        return 'test_questions_1'


class DbManagement:
    """ Handles dbs for student project """
    def __init__(self):
        raise NotImplementedError

    def select(self):
        raise NotImplementedError
