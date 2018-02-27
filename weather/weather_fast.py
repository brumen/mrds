#
# File defines:
#   mrd skew model for commodities (state reference)
#   a general diffusion model

import config 
import weather as we

def T_sim_inn_wrap (T_m_v_d, T_m_d_d, t_step, N_step, sp, hp, Z_m_d1, 
                    range_gpu2, range_gpum12, 
                    date_p, date_o):
    return we.T_sim_inn_d(T_m_v_d, T_m_d_d, t_step, N_step, sp, hp, 
                          Z_m_d1, range_gpu2, range_gpum12,
                          date_p, date_o )
