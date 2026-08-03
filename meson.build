project('optop', 'c', 'fortran',
  version: '1.0.0',
  meson_version: '>=0.64.0',
  default_options: ['warning_level=1', 'buildtype=release'])

py = import('python').find_installation(pure: false)
py_dep = py.dependency()

# NumPy / f2py include dirs and the f2py runtime source
incdir_numpy = run_command(py,
  ['-c', 'import numpy; print(numpy.get_include())'], check: true).stdout().strip()
incdir_f2py = run_command(py,
  ['-c', 'import numpy.f2py; print(numpy.f2py.get_include())'], check: true).stdout().strip()
inc_np   = include_directories(incdir_numpy)
inc_f2py = include_directories(incdir_f2py)
fortranobject_c = incdir_f2py / 'fortranobject.c'

# gfortran needs long free-form lines for order_parameters.f90
fc = meson.get_compiler('fortran')
if fc.get_id() == 'gcc'
  add_project_arguments('-ffree-line-length-none', language: 'fortran')
endif

fortran_sources = files(
  'src/optop/fortran/consts.f90',
  'src/optop/fortran/spherical_harmonics.f90',
  'src/optop/fortran/order_parameters.f90',
  'src/optop/fortran/op_frame_wrapper.f90',
)

# f2py: generate the C module + Fortran wrappers for _op_fortran
op_module = custom_target('_op_fortran_module',
  input: fortran_sources,
  output: ['_op_fortranmodule.c',
           '_op_fortran-f2pywrappers.f',
           '_op_fortran-f2pywrappers2.f90'],
  command: [py, '-m', 'numpy.f2py', '@INPUT@',
            '-m', '_op_fortran', '--lower', '--build-dir', '@OUTDIR@'])

py.extension_module('_op_fortran',
  fortran_sources + [op_module, fortranobject_c],
  include_directories: [inc_np, inc_f2py],
  dependencies: py_dep,
  install: true,
  subdir: 'optop',
)

# pure-Python sources
py.install_sources(
  ['src/optop/__init__.py',
   'src/optop/cli.py',
   'src/optop/calculator.py',
   'src/optop/_trajectory.py',
   'src/optop/_fortran_ext.py',
   'src/optop/_constants.py'],
  subdir: 'optop')

py.install_sources(
  ['src/optop/ml/__init__.py',
   'src/optop/ml/config.py',
   'src/optop/ml/pipeline.py'],
  subdir: 'optop/ml')
