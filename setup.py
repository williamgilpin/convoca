from distutils.core import setup

setup(
    name='convoca',
    version='0.1',
    description='Train convolutional neural networks to represent cellular automata',
    author='William Gilpin',
    author_email='firstnamelastname(as one word)@googleemailservice',
    requires=[ 'numpy', 'matplotlib', 'tensorflow'],
    py_modules=['config'],
    package_data={
    'ca_funcs': ['*'],
    'train_ca': ['*'],
    'utils': ['*'],
    'tests.test_ca': ['*'],
    },
)