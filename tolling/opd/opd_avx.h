void add4(PyObject *r, PyObject *a, PyObject *b, PyObject *c, PyObject *d, PyObject *y, int n);
void mul4(PyObject *r, PyObject *a, PyObject *b, double c, PyObject *y, int n);

void skew_fom( double F
             , PyObject *delta_X
             , double c1_ch
             , double qv
             , double c2_ch
             , double c3_ch
             , PyObject *sim_fom
             , int nb_sim);

double num_quad(PyObject *vec1, PyObject *vec2, int v_len);


void do_start_shut(PyObject *dc_can
                   , PyObject *dc_force
                   , PyObject *is_profitable
                   , PyObject *do_action
                   , int nb_sim );

void do_start_shut_simple(PyObject *dc_can
                   , PyObject *dc_force
                   , PyObject *is_profitable
                   , PyObject *do_action
                   , int nb_sim );


void startup_cost(PyObject *is_cold_start,
		  PyObject *starts,
		  double startup_cost_cold,
		  double startup_cost_p,
		  PyObject *fuel_prices,
		  double start_fuel_cold,
		  double start_fuel,
		  PyObject *res,
		  int nSize);

void is_start_profitable(PyObject *startup_sp_in,
			 double shutdown_sp_in,
			 PyObject *fixed_and_fuel_startup_cost,
			 double xud_startup_horizon,
			 double xud_shutdown_horizon,
			 double max_cap,
			 PyObject *shutdown_gen_profit,
			 PyObject *pp_arg_max,
			 PyObject *is_shutdown_profitable,
			 PyObject *is_startup_profitable,
			 int nSize);
