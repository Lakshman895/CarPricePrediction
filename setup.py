from setuptools import setup, find_packages

HYPEN_E_DOT = '-e .'
def get_requirements(file_path):
    requirements=[]
    with  open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.strip() for req in requirements]
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements


setup(
    name='CarPricePrediction',
    version='0.0.1',
    author='Lakshman',
    author_email='lakshman9440823127@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)