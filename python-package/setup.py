from distutils.core import setup

setup(
    name='tgboost',
    version='1.0',
    description='tiny gradient boosting tree',
    author='wepon',
    author_email='wepon@pku.edu.cn',
    url='http://wepon.me',
    packages=['tgboost'],
    package_data={'tgboost': ['tgboost.jar']},
    package_dir={'tgboost': 'tgboost'},
)
