from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name = 'deeplense_domain_adaptation',
    version = '0.1.0',
    license='MIT',
    description = 'A PyTorch-based collection of Unsupervised Domain Adaptation methods applied to strong gravitational lenses',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author = 'Marcos Tidball',
    author_email = 'marcostidball@gmail.com',
    url = 'https://github.com/zysymu/DeepLense-Domain-Adaptation',
    packages=find_packages(include=[
        'deeplense_domain_adaptation',
        'deeplense_domain_adaptation.algorithms',
        'deeplense_domain_adaptation.data',
        'deeplense_domain_adaptation.networks',
        ]),
    keywords = ['Gravitational Lensing', 'Unsupervised Domain Adaptation', 'Deep Learning', 'Dark Matter'],
    install_requires=[
        'numpy==1.21.2',
        'torch==1.9.0',
        'e2cnn==0.1.9',
        'torchvision==0.10.0',
        'matplotlib==3.4.3',
        'scikit-learn==0.24.2',
        'scipy==1.7.1',
        'seaborn==0.11.2',
        ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
    ],
)