import setuptools


with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='milton', 
    version='1.0.0',
    author='Marcin Karpinski',
    author_email='marcin.karpinski@astrazeneca.com',
    description='MachIne Learning PhenoType associatONs',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/AstraZeneca-CGR/zMILTON',
    packages=setuptools.find_packages(),
    scripts=['scripts/init.sh'],
    package_data={'milton': ['resources/*.jinja', 'resources/*.json']},
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
)
