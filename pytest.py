import pytest
from main import eng_to_python

source = "Write a python function for addition of two numbers"

py_code = eng_to_python(source)
# print()
print("success")