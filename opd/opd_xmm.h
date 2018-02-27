void add4(PyObject *r, PyObject *a, PyObject *b, PyObject *c, PyObject *d, PyObject *y,
	  int n);

void mul4(PyObject *r, PyObject *a, PyObject *b, double c, PyObject *y,
	  int n);

void mul5(PyObject *r, PyObject *a, PyObject *b, double c, PyObject *y,
	  int n);

void mul6(PyObject *r, PyObject *a, PyObject *b, double c, PyObject *y,
	  int n);

void mul7(PyObject *r, PyObject *a, PyObject *b, double c, PyObject *y,
	  int n);

void skew_fom(double F, PyObject *delta_X, 
	      double c1_ch, double qv, double c2_ch, double c3_ch, 
	      PyObject *sim_fom, 
	      size_t nb_sim);

double num_quad(PyObject *vec1, PyObject *vec2, int v_len);
