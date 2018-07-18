
class Student:

    school_name = "ABC_XYZ"

    def __init__(self, name, id, last_name):
        self.name = name
        self.id = id
        self.last_name = last_name

    def __str__(self):
        return "Student " + self.name

    def get_name_capital(self):
        return self.name.capitalize()