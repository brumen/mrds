PYTHON_COMMAND=python3
INCLUDE_PATH=/usr/lib/python3/dist-packages/numpy/core/include/

quartic: quartic_setup.py
	CFLAGS="-I$(INCLUDE_PATH)" $(PYTHON_COMMAND) quartic_setup.py build_ext --inplace

quartic_file:
	nvcc quartic.cu -o quartic_file
