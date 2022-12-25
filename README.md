# elegoua_engine
Models for improving students learning

## Example use:
```Python
from elegoua_engine import get_project

student_answer = {
    'level': {'auto_eval_level': 1},
    'interests': {
        'main_interest': 'ecology',
        'other_interests': ['rse', 'finance'],
    },
}
project = get_project.Project(student_answer)
student_level, student_questions = project.define_project()
```
