PYTHON_COMMAND=python3
INCLUDE_PATH=/usr/lib/python3/dist-packages/numpy/core/include/

vols: vols_setup.py
	CFLAGS="-I$(INCLUDE_PATH)" $(PYTHON_COMMAND) vols_setup.py build_ext --inplace
