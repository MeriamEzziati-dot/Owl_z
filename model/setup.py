from setuptools import Extension,setup
from Cython.Build import cythonize

openmp_arg = "-fopenmp"
ext_modules = [
    Extension(
        "model.mlt_c",
        ["model/mlt_c.pyx"],
        extra_compile_args=[openmp_arg],
        extra_link_args=[openmp_arg],
    ),
    Extension(
        "model.qso_c",
        ["model/qso_c.pyx"],
        extra_compile_args=[openmp_arg],
        extra_link_args=[openmp_arg],
    )
]
setup(
    name="mlt_with_cython",
    ext_modules=cythonize(ext_modules)

)
