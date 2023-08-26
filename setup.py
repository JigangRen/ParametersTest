# python setup.py build_ext --inplace
# 运行方法如上，此外如果是win运行这个可能会报错，需要c++编译环境（简单的说就是需要装个visio studio）


from distutils.core import setup, Extension
import numpy
from distutils.sysconfig import *
from distutils.command.build_py import build_py_2to3 as build_py
from Cython.Distutils import build_ext
import os

description = '这是用来编译NMF.cpp的文件'
"""
下面这个地方要改名成生成的那个文件名，然后再运行一次（因为没写环境检测代码，自己搞吧）
比如一般装了python3.9的amd64机器上就会叫NMF.cp39-win_amd64.pyd
而linux设备就会叫xxxx.so忘了叫啥了，加油
"""
path = "NMF.cp36-win_amd64.pyd"

if os.path.isfile('DataPreprocessing/NMF/NMF.pyd'):
    os.remove('DataPreprocessing/NMF/NMF.pyd')

extra_compile_args = ['-O2']

# data files
data_files = []

# scripts
scripts = []

# Python include
py_inc = [get_python_inc()]

# NumPy include
np_inc = [numpy.get_include()]

# cmdclass
cmdclass = {'build_py': build_py}

# Extension modules
ext_modules = []
cmdclass.update({'build_ext': build_ext})
ext_modules += [
    Extension("NMF",
              ["DataPreprocessing/NMF/c_NMF.cpp",
               "DataPreprocessing/NMF/NMF.pyx"],
              language='c++',
              include_dirs=py_inc + np_inc)
]

packages = ['nmf']

setup(name='nmf',
      version='1.0',
      description=description,
      author='Dong Chen Miao',
      author_email='hsiaoo@qq.com',
      packages=packages,
      cmdclass=cmdclass,
      ext_modules=ext_modules,
      scripts=scripts,
      data_files=data_files,
      )

print('==============================================')
print('Setup succeeded!\n')

if os.path.isfile(path):
    os.rename(path, 'DataPreprocessing/NMF/NMF.pyd')
