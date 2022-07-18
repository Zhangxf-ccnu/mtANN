import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mtANN", 
    version="1.0.0",
    py_modules=['mtANN.mtANN', 'mtANN.models', 'mtANN.mmd', 'mtANN.params', 'mtANN.preprocess', 'mtANN.select_gene','mtANN.train','mtANN.utils'],
    author="Yi-Xuan Xiong",
    author_email="xyxuana@mails.ccnu.edu.cn",
    description="Ensemble Multiple References for Single-cell RNA Seuquencing Data Annotation and Unseen Cells Identification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # url="https://github.com/jianhuupenn/ItClust",
    packages=setuptools.find_packages(),
    # install_requires=["keras","pandas","numpy","scipy","scanpy","anndata","natsort","sklearn"],
    #install_requires=[],
    # classifiers=[
    #     "Programming Language :: Python :: 3",
    #     "License :: OSI Approved :: MIT License",
    #     "Operating System :: OS Independent",
    # ],
    python_requires='>=3.7',
)