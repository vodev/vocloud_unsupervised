from setuptools import setup, find_packages
# from os import path
# root = path.abspath(path.dirname(__file__))
# get the long discription from README.rst file
# with open(path.join(root, 'README.rst')) as f:
#     long_description = f.read()


setup(
        name='vocloud_unsupervised',
        version='0.0.1',
        packages=find_packages(),
        url='',
        license='MIT',
        author='Ksenia Shakurova',
        author_email='nona493@gmail.com',
        description='Package with some sequence and parallel algorithms of unsupervised learning',
        install_requires=[
            'numpy',
            'pandas',
            'scikit-learn',
        ]
)
