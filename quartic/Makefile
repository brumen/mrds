PYTHON_COMMAND=python3

quartic: quartic_setup.py
	$(PYTHON_COMMAND) quartic_setup.py build_ext --inplace

quartic_file:
	nvcc quartic.cu -o quartic_file
