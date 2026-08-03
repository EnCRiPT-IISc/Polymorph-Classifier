# Build _op_fortran Python extension using f2py
#
# Usage:
#   make          -- build and install the .so into the op_ml package
#   make clean    -- remove build artefacts

FC       = mpiifx
CC       = mpiicx
FFLAGS   = -O3
BUILD    = /tmp/f2py_build_op
MODULE   = _op_fortran
# Auto-detect where op_ml is installed
PKGDIR   = $(shell python3 -c "import op_ml, os; print(os.path.dirname(op_ml.__file__))")

SOURCES  = consts.f90 spherical_harmonics.f90 order_parameters.f90 op_frame_wrapper.f90

.PHONY: all build copy clean

all: build copy

build:
	python3 -m numpy.f2py -c $(SOURCES) \
	    --f90exec=$(FC) \
	    --f90flags="$(FFLAGS)" \
	    -m $(MODULE) \
	    --build-dir $(BUILD)
	FC=$(FC) CC=$(CC) meson setup $(BUILD)/bbdir $(BUILD) --wipe
	FC=$(FC) CC=$(CC) meson compile -C $(BUILD)/bbdir

copy:
	cp $(BUILD)/bbdir/$(MODULE)*.so $(PKGDIR)/
	@echo "Extension installed to $(PKGDIR)/"

clean:
	rm -rf $(BUILD) $(PKGDIR)/$(MODULE)*.so
