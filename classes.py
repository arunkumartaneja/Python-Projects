students = []


class Student:

    school_name = "ABC_XYZ"

    def __init__(self, name):
        self.name = name
        self.id = 3
        students.append(self)

    def __str__(self):
        return "Student " + self.name

    def get_name_capital(self):
        return self.name.capitalize()


class HighSchoolStudent(Student):
    school_name = "High School"

    def get_name_capital(self):
        return "High school student name: " + super().get_name_capital()


james = HighSchoolStudent("james")
print(james.get_name_capital())
