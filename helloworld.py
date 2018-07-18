
student_names = []


def save_file(student):
    try:
        f = open("student.txt", "a")
        f.write(student + "\n")
        f.close()
    except Exception as error:
        print(error)
        print("File could not save")


def read_file():
    try:
        f = open("student.txt", "r")
        for student in f.readlines():
            add_student(student)
        f.close()
    except Exception:
        print("Could not read file")


def add_student(name):
    student_names.append(name)


def print_student():
    for name in student_names:
        print(name)


read_file()
print_student()
student_name = input("Enter name: ")
add_student(student_name)
save_file(student_name)
