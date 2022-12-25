def eval_python_level(student_answer:dict):
    """
    Provide a score and complementary information about the student level
    """
    student_level = student_answer['auto_eval_level']
    return student_level


def eval_student_interests(student_answer):
    """
    Provide student main interests
    """
    student_interests = [student_answer['main_interest']]
    student_interests += student_answer['other_interests']
    return student_interests


class Project:
    """ Analyze student answer to provide adapted project (db and questions) """
    
    def __init__(self, student_answer:dict, student_interests:list=None, student_level:int=0):
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
        dbs = self.select_dbs()
        questions = self.select_questions()
        return dbs, questions

    def eval_student_answer(self):
        student_level = self.student_answer['level']
        student_interests = self.student_answer['interests']
        self.student_level = eval_python_level(student_level)
        self.student_interests = eval_student_interests(student_interests)

    def select_dbs(self):
        return 'test_db_1'

    def select_questions(self):
        return 'test_questions_1'


class DbManagement:
    """ Handles dbs for student project """
    
    def __init__(self):
        raise NotImplementedError

    def create_db(self):
        raise NotImplementedError
    
    def insert(self):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError

    def delete(self):
        raise NotImplementedError

    def select(self):
        raise NotImplementedError

    
