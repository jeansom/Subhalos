#from distutils.core import setup
#from Cython.Build import cythonize

#setup(
#    ext_modules = cythonize("pulsars/*.pyx")
#)



# from distutils.core import setup, Extension
# from Cython.Build import cythonize
# import numpy

# setup(
#     ext_modules=[
#                  Extension("pulsars/*.pyx",
#                            include_dirs=[numpy.get_include()] 
#                            ) 
#                            #extra_compile_args=["-ffast-math",'-O3']),
#                  ],
#     )

# Or, if you use cythonize() to make the ext_modules list,
# include_dirs can be passed to setup()

# setup(
#      ext_modules=cythonize("pulsars/*.pyx"),
#      include_dirs=[numpy.get_include()],
#      extra_compile_args=["-ffast-math",'-O3']
#      )

#["pulsars/likelihood_psf.c","pulsars/special.c"]



from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension("*", ["pulsars/*.pyx"],
        include_dirs=[numpy.get_include()], extra_compile_args=["-ffast-math",'-O3']),
    Extension("*", ["sim/*.pyx"],
        include_dirs=[numpy.get_include()], extra_compile_args=["-ffast-math",'-O3']) ,
    Extension("*", ["config/*.pyx"],
        include_dirs=[numpy.get_include()], extra_compile_args=["-ffast-math",'-O3']) 
]
setup(
    #name = "My hello app",
    ext_modules = cythonize(extensions),
)
