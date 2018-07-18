from student import Student


class HighSchoolStudent(Student):

    """
    High school class for high school
    students.
    :param school_name: string - school name
    """
    school_name = "High School"

    def get_name_capital(self):
        return "High school student name: " + super().get_name_capital()