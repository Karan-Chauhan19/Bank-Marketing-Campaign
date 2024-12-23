'''
Author : Karan Chauhan
Github : Karan-Chauhan19
Organization : L.J University
'''
from setuptools import  setup, find_packages

setup(
    name='Bank Marketing Campaign',  # project name
    version='1.0',  # version number
    description="This project uses a dataset collected from a bank's marketing campaign. The goal is to develop a machine learning model to predict whether a customer will subscribe to a term deposit or not, based on various customer attributes and past campaign details.",  # description of the project
    packages = find_packages(),  # find all packages
    author='Karan-Chauhan' ,# author of the package
    author_email='kc879022@gmail.com', # email of the author
    url='https://github.com/Karan-Chauhan19/Bank-Marketing-Campaign.git', # url of the project
    install_requires=['pandas', 'numpy', 'sklearn', 'matplotlib','seaborn','keras','tensorflow','imbalanced-learn','torch','joblib'],
    # list of the dependencies required by the package
    classifiers=['Programming Language :: python :: 3.12.3']
)