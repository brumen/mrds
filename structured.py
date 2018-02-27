import config
import numpy as np
import pricers
import cuda_ops as co
import pycuda.gpuarray as gpa


def dual_binary(mm, params):
    """
    # computes the value of the dual binary
    # K ... vector of strikes (2x1)
    # T ... expiry time
    # mm ... market model
    # fwd_idx ... always 0 (just 1 contract simulated)
    """
    fwd_idx = params[0]
    K = params[1]
    T = mm.simulation_times [fwd_idx]

    return mm.DF (T) * np.mean((mm.simulated_curves[0][3, fwd_idx, :] < K[0]) *
                               (mm.simulated_curves[1][3, fwd_idx, :] > K[1]))


def call(mm, params):
    """
    european call option (for testing purposes)
    """
    K = params[0]
    fwd_idx = params[1]
    sim_time_idx = params[2]
    sc = mm.simulated_curves[0][sim_time_idx, fwd_idx, :]
    return np.mean(mm.DF(mm.simulation_times[sim_time_idx]) * (sc - K) * (sc - K > 0))


def forward(mm, fwd_idx):
    return [np.mean(mm.simulated_curves[0][0, fwd_idx, :]),
            np.mean(mm.simulated_curves[1][0, fwd_idx, :])]


def forward_1(mm, params):
    return np.mean ( mm.simulated_curves[0][:,:,:] - params[0] , axis=2 )


def coupon_strip(mm, params):
    """
    coupon strip (monthly payments, monthly observations)
    """
    coupon = params[0]
    barr_v = params[1]
    sc = mm.simulated_curves[0]
    barr_ind = (np.diff(sc[:, 12, :], axis=0) < barr_v[0]) & \
               (np.diff(sc[:, 12, :], axis=0) > barr_v[1])
    discounts = np.array(mm.DF(mm.simulation_times[1:len(mm.simulation_times)]))
    return np.mean(np.sum(coupon * barr_ind * discounts, axis=0))


def basket_best_worst(mm, params):
    """
    basket best worst conditional on second asset
    S0 ... initial asset prices
    fwd ... forward contract considered
    w ... weight vector
    strike ... strike price
    """
    S0 = params[0]
    fwd = params[1]
    barrier = params[2]
    strike = params[3]
    w = params[4]

    nb_sim = mm.simulated_curves[0].shape[2]

    S_last_1_v = mm.simulated_curves[0][mm.nb_time_steps-1, fwd, :] / S0[0]
    S_last_2_v = mm.simulated_curves[1][mm.nb_time_steps-1, fwd, :] / S0[1]
    S_last_3_v = mm.simulated_curves[2][mm.nb_time_steps-1, fwd, :] / S0[2]

    payoff = w[0] * S_last_1_v + w[1] * S_last_2_v + w[2] * S_last_3_v
    payoff = (payoff - strike) * (payoff - strike > 0)

    S_1 = mm.simulated_curves[0][:, fwd, :] / S0[0]
    S_2 = mm.simulated_curves[1][:, fwd, :] / S0[1]
    S_3 = mm.simulated_curves[2][:, fwd, :] / S0[2]

    # maximum of three matrices in 2 steps
    ma_12 = S_1* (S_1 > S_2) + S_2 * (S_1 <= S_2)
    ma = ma_12 * (ma_12 > S_3) + S_3 * (ma_12 <= S_3)
    ma_log = ma >= barrier
    ma_log = np.kron (np.ones ( (mm.nb_time_steps, 1) ) , np.array (map (lambda x: x.all(), ma_log.T )) )
    discounts = np.array(mm.DF(mm.simulation_times[0:len(mm.simulation_times)]))

    return np.mean(ma_log * payoff * discounts)


def sb_easy(com, d_today, sim_times, barrier_strike,
            nb_sim=50000):
    """
    TO BE COMPLETED
    """
    mm = mrds.mrds_calib(com, d_today, 12)
    mm.update_sim_times(sim_times)
    mm.simulate_curves(nb_sim, tenor_list=[[11]])
    return sb(mm, barrier_strike)


def sb(mm, params):
    """
    single barrier
    :param fwd: which forward contract chosen
    """
    barrier = params[0]
    strike = params[1]
    sc = mm.simulated_curves[0]
    nb_sim_times, nb_sim = sc.shape[0], sc.shape[2]

    payoff = sc[-1, 0, :]  # last row of S
    payoff = (payoff - strike) * (payoff - strike > 0)
    ma_log = sc[:, 0, :] >= barrier
    ma_log_pos = np.sum(ma_log, axis=0) == nb_sim_times
    disc = mm.DF(mm.simulation_times[-1])

    return disc * np.mean(ma_log_pos * payoff)


def sb_cuda(mm, sc_d, params):
    """
    single barrier
    """
    barrier = params[0]
    strike = params[1]
    nb_sim_times, nb_sim = sc_d.shape[0], sc_d.shape[2]

    payoff = sc_d[-1, 0, :]  # last row of S
    payoff = (payoff - strike) * (payoff - strike > 0)
    ma_log = sc_d[:, 0, :] >= barrier
    ma_log_pos = co.colsum_cuda_last(ma_log) == nb_sim_times
    disc = mm.DF(mm.simulation_times[-1])
    return disc * gpa.sum(ma_log_pos * payoff).get() / nb_sim


def sb_cuda_fast(payoff_d, paths_d, params, disc):
    """
    single barrier
    """
    barrier = params[0]
    strike = params[1]
    nb_sim_times, nb_sim = paths_d.shape[0], paths_d.shape[1]
    payoff_call = (payoff_d - strike) * (payoff_d - strike > 0)
    paths_above = paths_d >= barrier
    paths_above_always = co.colsum_cuda_last(paths_above) == nb_sim_times

    return disc * gpa.sum(paths_above_always * payoff_call).get() / nb_sim


def db(mm, params):
    """
    double barrier (same as single, just has to barriers)
    """
    (barrier_d, barrier_u), strike = params
    sc = mm.simulated_curves[0][:, 0, :]

    payoff = sc[-1, :]  # last row of S
    payoff = (payoff - strike) * (payoff - strike > 0)
    # maximum of three matrices in 2 steps
    ma_up = np.sum(sc[:, :] >= barrier_u, axis=0) == 0
    ma_dw = np.sum(sc[:, :] <= barrier_d, axis=0) == 0
    discount = mm.DF(mm.simulation_times[-1])

    return discount * np.mean(payoff * ma_up * ma_dw)


def db_cuda(mm, params):
    """
    double barrier (same as single, just has to barriers)
    """
    (barrier_d, barrier_u), strike = params
    sc = mm.simulated_curves[0][:, 0, :]
    nb_sim = sc.shape[1]
    payoff = sc[-1, :]  # last row of S
    payoff = (payoff - strike) * (payoff - strike > 0)
    # maximum of three matrices in 2 steps
    ma_up = co.colsum_cuda_last(sc >= barrier_u) == 0
    ma_dw = co.colsum_cuda_last(sc <= barrier_d) == 0
    discount = mm.DF(mm.simulation_times[-1])

    return discount * gpa.sum(payoff * ma_up * ma_dw) / nb_sim


def sb_fs_gen(mm, params):
    """
    forward starting single barrier KI
     fwd ... index of the forward contract
     barrier ... barrier for the performance
     strike ... strike for performance
    """
    fwd = params[0]
    barrier_ratio = params[1]
    strike_ratio = params[2]

    nb_sim = mm.simulated_curves[0].shape[2]

    payoff = mm.simulated_curves[0][mm.nb_time_steps-1, fwd, :]  # last row of S
    strike = strike_ratio * mm.simulated_curves[0][0, fwd, :] # first row of S multiplied by strike
    payoff = (payoff - strike) * (payoff - strike > 0)
    payoff = np.kron ( np.ones ((mm.nb_time_steps,1)), payoff)

    # maximum of three matrices in 2 steps
    ma_log = mm.simulated_curves[0][:,fwd,:] >= barrier_ratio * np.kron ( np.ones ((mm.nb_time_steps,1)), mm.simulated_curves[0][0,fwd,:])
    ma_log = np.kron(np.ones((mm.nb_time_steps, 1)),
                     np.array(map(lambda x: x.any(), ma_log.T)))  # logical and on columns (therefore transpose)

    discounts = np.array(mm.DF(mm.simulation_times[0:len(mm.simulation_times)]))
    discounts = np.kron(discounts.reshape(mm.nb_time_steps, 1), np.ones((1, nb_sim)))

    return np.mean(ma_log * payoff * discounts)/mm.forward_curve_list[0][fwd]


def sbc (mm, params):
    """
    single barrier conditional second asset
    """
    fwd_1 = params[0]
    fwd_2 = params[1]
    barrier = params[2]
    strike = params[3]

    sc1 = mm.simulated_curves[0]
    sc2 = mm.simulated_curves[1]
    nb_sim = sc1.shape[2]

    payoff = sc1[mm.nb_time_steps-1, fwd_1, :]
    payoff = (payoff - strike) * (payoff - strike > 0)
    payoff = np.kron(np.ones((mm.nb_time_steps, 1)), payoff)

    ma_log = sc2[:, fwd_2, :] >= barrier
    ma_log = np.kron(np.ones((mm.nb_time_steps, 1)),
                     np.array(map(lambda x: x.any(), ma_log.T)))  # logical and on columns (therefore transpose)

    discounts = np.array(mm.DF(mm.simulation_times[0:len(mm.simulation_times)]))
    discounts = np.kron(discounts.reshape(mm.nb_time_steps, 1), np.ones((1, nb_sim)))

    return np.mean(ma_log * payoff * discounts)


def otapo(mm, params):
    """
    one touch APO pricer
    """
    fwd = params[0]
    barrier = params[1]
    strike = params[2]

    nb_sim = mm.simulated_curves[0].shape[2]

    payoff = np.mean (mm.simulated_curves[0][:,fwd,:], axis=0) # average over columns (which is time axis)
    payoff = (payoff - strike) * (payoff - strike > 0)
    payoff = np.kron ( np.ones ((mm.nb_time_steps,1)), payoff)

    # maximum of three matrices in 2 steps
    ma_log = mm.simulated_curves[0][:,fwd,:] >= barrier
    ma_log = np.kron (np.ones ( (mm.nb_time_steps, 1) ) , np.array (map (lambda x: x.all(), ma_log.T )) ) # logical and on columns (therefore transpose)

    discounts = np.array(mm.DF ( mm.simulation_times[0:len(mm.simulation_times)] ) )
    discounts = np.kron ( discounts.reshape (mm.nb_time_steps,1), np.ones ( (1,nb_sim) ) )

    return np.mean ( ma_log * payoff * discounts )


# call option on performance
def call_perf (mm, params):
    fwd = params[0]
    K = params[1]

    payoff = mm.simulated_curves[0][-1,fwd,:] / mm.simulated_curves[0][0,fwd,:] # performance
    discounts = mm.DF ( mm.simulation_times[-1] )
    return  discounts * np.mean ( (payoff - strike) * (payoff - strike > 0) )



# variable quantity call option
def var_quant_call (mm, params):
    fwd = params[0]
    K = params[1]

    payoff = mm.simulated_curves[0][-1,fwd,:] / mm.simulated_curves[0][0,fwd,:] # performance
    discounts = mm.DF ( mm.simulation_times[-1] )
    linear_st = (L_u - mm.simulated_curves[0][-1,fwd,:] ) / ( L_u - L_l )
    var_quant = linear_st * (linear_st < 1.) * (linear_st > 0.) + (linear_st > 1.) # callspread on [0,1]

    return  discounts * np.mean ( var_quant * payoff )


# early redemption callspread
def er_cs (mm, params):
    fwd = params[0] # fwd index
    B = params[1] # barrier
    C = params[2]
    F = params[3]

    # TO BE CORRECTED, TO BE CORRECTED
    perf = mm.simulated_curves[0][:,fwd,:] / mm.simulated_curves[0][0,fwd,:] # performance
    cs = C * (perf -1. > C) + (perf - 1.) * (F < perf - 1.) * (perf - 1. < C) + F * (perf - 1. < F) # callspread
    # breach = perf > B # for every sim. the index when it was breached
    breach = cumsum (cumsum (perf>B, axis = 0), axis=0) == 1
    discounts = np.array(mm.DF ( mm.simulation_times ) ) # vector of discounts
    #indic = zeros ( mm.simulated_curves[0].shape[2] ) #indicator if the first elt.
    payoff = 0.
    #for t_ind in xrange (mm.simulated_curves[0].shape[0]):
    payoff += cs[t_ind,:] * breach[t_ind,:] * (indic == 0.) * discounts[t_ind]
    indic += breach[t_ind,:] > 0. # which are nonzeros

    return np.mean ( sum ( breach * cs, axis = 0) )


# KIKO (knock in, knock out) - 1 asset
# knocks in when S < B_l , knocks out when S > B_u
#
def kiko_rebate (mm, params):
    fwd = params[0]
    B_l = params[1][0]
    B_u = params[1][1]
    strike = params[2]

    nb_sim = mm.simulated_curves[0].shape[2]

    perf = mm.simulated_curves[0][:,fwd,:] / mm.simulated_curves[0][0,fwd,:] # performance
    ki = sum (perf <= B_l , axis = 0)
    ko = sum (perf >= B_u , axis = 0)
    payoff = perf[-1,:] # terminal prices
    discounts = np.array(mm.DF ( mm.simulation_times ) )
    discounts = np.kron ( discounts.reshape (mm.nb_time_steps,1), np.ones ( (1,nb_sim) ) )
    payoff = discounts * (payoff - strike) * (payoff - strike > 0.) * ( ki > 0) * (ko == 0)

    return np.mean ( payoff )

# KO (knock out) - 1 asset
# knocks out when S > B_u
#
def ko_rebate (mm, params):
    fwd = params[0]
    B_u = params[1][0]
    strike = params[2]

    nb_sim = mm.simulated_curves[0].shape[2]

    perf = mm.simulated_curves[0][:,fwd,:] / mm.simulated_curves[0][0,fwd,:] # performance
    ko = sum (perf >= B_u , axis = 0)
    discounts = np.array(mm.DF ( mm.simulation_times ) )
    discounts = np.kron ( discounts.reshape (mm.nb_time_steps,1), np.ones ( (1,nb_sim) ) )
    payoff = discounts * (ko == 0)

    return np.mean ( payoff )


# zero coupon bond rebate
# first part is the same as in ko, then zcb yields
def zcb_ko_rebate (mm, params):
    fwd = params[0]
    B_u = params[1][0]
    strike = params[2]
    zcb_yield = params[3]

    nb_sim = mm.simulated_curves[0].shape[2]

    perf = mm.simulated_curves[0][:,fwd,:] / mm.simulated_curves[0][0,fwd,:] # performance
    ko = sum (perf >= B_u , axis = 0)
    discounts = np.array(mm.DF ( mm.simulation_times ) )
    discounts = np.kron ( discounts.reshape (mm.nb_time_steps,1), np.ones ( (1,nb_sim) ) )
    payoff = discounts * (ko == 0) * np.kron ( zcb_yield.reshape (mm.nb_time_steps, 1), np.ones ( (1, nb_sim) ) )

    return np.mean (payoff)



# basket KIKO
def basket_kiko (mm, params):

    fwd = params[0]
    B_l = params[1][0]
    B_u = params[1][1]
    K = params[2]

    basket = 0.5 * ( mm.simulated_curves[0][:,fwd,:]/mm.simulated_curves[0][0,fwd,:] + \
                     mm.simulated_curves[1][:,fwd,:]/mm.simulated_curves[1][0,fwd,:] )

    basket_ki = sum ( basket.T > B_l )
    basket_ko = sum ( basket.T < B_u )

    payoff = ( basket[-1,:] - K ) * (basket[-1,:] - K > 0.0) * (basket_ki > 0) * (basket_ko == 0)
    discounts = mm.DF ( mm.simulation_times[-1] )

    return discounts * np.mean ( payoff )


# implements basket Worst of kiko  (2 asset restriction )
def basket_wo_kiko (mm, params):

    fwd = params[0]
    B = params[1]
    B_l = params[2][0]
    B_u = params[2][1]
    rebate = params[3]

    asset_1 = mm.simulated_curves[0][:,fwd,:]/mm.simulated_curves[0][0,fwd,:]
    asset_2 = mm.simulated_curves[1][:,fwd,:]/mm.simulated_curves[1][0,fwd,:]

    worst = asset_2 * (asset_1 > asset_2 ) + asset_1 * (asset_1 <= asset_2 )

    worst_ki = sum (worst <= B_l , axis = 0)
    worst_ko = sum (worst >= B_u , axis = 0)

    payoff = ( worst[-1,:] - B ) * (worst[-1,:] > B ) * ( worst_ki > 0) * (worst_ko == 0)
    discounts = mm.DF ( mm.simulation_times[-1] )

    return discounts * np.mean ( payoff )


def basket_best_lockin_apo (mm, params):

    fwd = params[0]
    B = params[1]
    L = params[2]

    # worst of basket
    asset_1 = mm.simulated_curves[0][:,fwd,:]/mm.simulated_curves[0][0,fwd,:]
    asset_2 = mm.simulated_curves[1][:,fwd,:]/mm.simulated_curves[1][0,fwd,:]


    avg_1 = np.mean (asset_1, axis=0)
    avg_2 = np.mean (asset_2, axis=0)

    rep = 0.5 * avg_1 * (avg_1 < avg_2) + 0.5 * avg_2 * (avg_1 >= avg_2) + 0.5 * L

    #print worst_ki
    #print worst_ko

    payoff = ( rep - B ) * (rep > B )
    discounts = mm.DF ( mm.simulation_times[-1] )

    return discounts * np.mean ( payoff )




# basket barrier (including one-touch) rebate APO
def bb_APO (mm, params):
    fwd = params[0]
    B = parmas[1]
    K = params[2]
    rebate = params[3]

    perf = []
    for asset_nb in xrange (mm.nb_assets):
        perf.append ( mm.simulated_curves[asset_nb][:,fwd,:]/mm.simulated_curves[asset_nb][0,fwd,:] )


    payoff_ind = sum ( np.mean( perf ) > B , axis = 0 )
    payoff_ind = payoff_ind >=1

    payoff = np.mean ( np.mean (perf) , axis = 0) # terminal APOs
    payoff = (payoff - strike) * (payoff - strike > 0)

    discounts = mm.DF ( mm.simulation_times[-1] )

    return discounts * np.mean ( payoff * payoff_ind )


# one-touch basket barrier with rebate
def bb_rebate (mm, params):
    fwd, B, K, rebate = params

    perf = mm.simulated_curves[0][:, fwd, :]/ mm.simulated_curves[0][0, fwd, :] / np.double(mm.nb_assets)
    for asset_nb in range(1, mm.nb_assets):
        sc = mm.simulated_curves[asset_nb]
        perf += sc[:, fwd, :] / sc[0, fwd, :] / np.double(mm.nb_assets)

    T = len (mm.simulated_curves[0][:, 1, 1]) - 1  # last time point
    payoff_ind = np.sum(perf > B, axis=0) == 0

    payoff = (perf[T, :] - K) * (perf[T, :] - K > 0)
    discounts = mm.DF(mm.simulation_times[-1])

    return discounts * np.mean(payoff * payoff_ind)



# basket barrier APO locally capped
def b_APO_lc (mm, params):
    fwd = params[0]
    B = parmas[1]
    K = params[2]
    rebate = params[3]

    avg_perf = 0.
    for asset_nb in xrange (mm.nb_assets):
        perf = np.mean (mm.simulated_curves[asset_nb][:,fwd,:]/mm.simulated_curves[asset_nb][0,fwd,:], axis=0)
        # applying callspread
        avg_perf += ( C * (perf -1. > C) + (perf - 1.) * (F < perf - 1.) * (perf - 1. < C) + \
                      F * (perf - 1. < F) ) / double (mm.nb_assets)

    mp = np.mean (avg_perf) # basket out of it
    payoff = (mp - strike) * (mp - strike > 0)

    discounts = mm.DF ( mm.simulation_times[-1] )

    return discounts * np.mean ( payoff )



# positive replacement
def pos_rep (mm, params):
    fwd = params[0]
    K = params[1] # asset strike
    A = params[2] # coupon level
    C = params[3] # cap level = K
    E = params[4] # option strike
    w = params[5] # weights

    nb_sims = mm.simulated_curves[0].shape[2]

    p_avg = []
    p_clsp = []
    p_und = []
    for asset_nb in xrange (mm.nb_assets):
        p_avg.append ( np.mean ( mm.simulated_curves[asset_nb][:,fwd,:]/mm.simulated_curves[asset_nb][0,fwd,:] , axis = 0 ) )
        p_clsp.append ( A * ( p_avg[asset_nb] > K ) )
        p_und.append ( w[asset_nb] * ( p_clsp[asset_nb] - (C - p_avg[asset_nb]) * (C-p_avg[asset_nb] > 0) ) )

    p_bsk = reduce (lambda x,y: x + y, p_und )

    discounts = mm.DF ( mm.simulation_times[-1] )

    return np.mean ( (p_bsk - E) * (p_bsk - E > 0) ) * discounts


# positive replacement
def local_floor_pos_rep (mm, params):
    fwd = params[0]
    K = params[1] # asset strike
    A = params[2] # coupon level
    C = params[3] # cap level = K
    F = params[4]
    E = params[5] # option strike
    w = params[6] # weights

    nb_sims = mm.simulated_curves[0].shape[2]

    p_avg = []
    p_mm = []
    p_clsp = []
    p_und = []
    for asset_nb in xrange (mm.nb_assets):
        # print mm.simulated_curves[asset_nb][:,fwd,:]/mm.simulated_curves[asset_nb][0,fwd,:]
        p_avg.append ( mm.simulated_curves[asset_nb][-1,fwd,:]/mm.simulated_curves[asset_nb][0,fwd,:] )
        p_clsp.append ( A * ( p_avg[asset_nb] > K ) )
        p_mm.append ( F * (p_avg[asset_nb] < F) + p_avg[asset_nb] * (F < p_avg[asset_nb] ) * (p_avg[asset_nb] < C) +
                      C * (p_avg[asset_nb] > C) )
        p_und.append ( w[asset_nb] * (p_clsp[asset_nb] + p_mm[asset_nb] ) )

    p_bsk = reduce (lambda x,y: x + y, p_und )

    discounts = mm.DF ( mm.simulation_times[-1] )
    return np.mean ( (p_bsk - E) * (p_bsk -E > 0) ) * discounts


def apo_avg (mm, params):
    # apo w/ averaging strikes
    # fwd = -1 ... 1 NB contract
    # fwd >= 0 ... fixed mat. contract
    fwd = params[0]
    strike = params[1]
    avg_per = params[2] # averaging period indicators
    apo_per = params[3] # apo period indicators

    # nb_sim = mm.simulated_curves[0].shape[2]

    if fwd >= 0: # fixed maturity
        apo = np.mean (mm.simulated_curves[0][apo_per,fwd,:], axis=0) # average over columns (which is time axis)
        avg = np.mean (mm.simulated_curves[0][avg_per,fwd,:], axis=0) # average over columns (which is time axis)
        payoff = (apo/avg - strike) * (apo/avg - strike > 0)
    else: # 1NB contract
        apo = np.mean (mm.simulated_curves[0][apo_per,:], axis=0) # average over columns (which is time axis)
        avg = np.mean (mm.simulated_curves[0][avg_per,:], axis=0) # average over columns (which is time axis)
        payoff = (apo/avg - strike) * (apo/avg - strike > 0)

    # maximum of three matrices in 2 steps
    discounts = mm.DF ( mm.simulation_times[-1] )

    return np.mean ( payoff  ) * discounts


def wo_er (mm, params):
    """
    # WO early redemption
    """
    S0 = params[0]
    fwd = params[1]
    barrier = params[2]
    strike = params[3]
    w = params[4]

    nb_sim = mm.simulated_curves[0].shape[2]

    S_last_1_v = mm.simulated_curves[0][mm.nb_time_steps-1,fwd,:] / S0[0]
    S_last_2_v = mm.simulated_curves[1][mm.nb_time_steps-1,fwd,:] / S0[1]
    S_last_3_v = mm.simulated_curves[2][mm.nb_time_steps-1,fwd,:] / S0[2]

    payoff = w[0] * S_last_1_v + w[1] * S_last_2_v + w[2] * S_last_3_v
    payoff = (payoff - strike) * (payoff - strike > 0)
    payoff = np.kron(np.ones((mm.nb_time_steps, 1)), payoff)

    S_1 = mm.simulated_curves[0][:,fwd,:] / S0[0]
    S_2 = mm.simulated_curves[1][:,fwd,:] / S0[1]
    S_3 = mm.simulated_curves[2][:,fwd,:] / S0[2]

    # maximum of three matrices in 2 steps
    ma_12 = S_1* (S_1 > S_2) + S_2 * (S_1 <= S_2)
    ma = ma_12 * (ma_12 > S_3) + S_3 * (ma_12 <= S_3)
    ma_log = ma >= barrier

    ma_log = np.kron(np.ones((mm.nb_time_steps, 1)),
                     np.array(map(lambda x: x.all(), ma_log.T)))  # logical and on columns (therefore transpose)

    discounts = np.array(mm.DF(mm.simulation_times[0:len(mm.simulation_times)]))
    discounts = np.kron(discounts.reshape(mm.nb_time_steps, 1), np.ones((1, nb_sim)))

    return np.mean(ma_log * payoff * discounts)





# basket perofrmance locally capped
def basket_lc (mm, params):
    nb_sims = mm.simulated_curves[0].shape[2]

    p_rel =  [mm.simulated_curves[asset_nb][-1,fwd,:]/mm.simulated_curves[asset_nb][0,fwd,:] for asset_nb in xrange (self.nb_assets) ]

    # UNFINISHED


    p_avg = []
    p_mm = []
    p_clsp = []
    p_und = []
    for asset_nb in xrange (mm.nb_assets):
        # print mm.simulated_curves[asset_nb][:,fwd,:]/mm.simulated_curves[asset_nb][0,fwd,:]
        p_avg.append ( mm.simulated_curves[asset_nb][-1,fwd,:]/mm.simulated_curves[asset_nb][0,fwd,:] )
        p_clsp.append ( A * ( p_avg[asset_nb] > K ) )
        p_mm.append ( F * (p_avg[asset_nb] < F) + p_avg[asset_nb] * (F < p_avg[asset_nb] ) * (p_avg[asset_nb] < C) +
                      C * (p_avg[asset_nb] > C) )
        p_und.append ( w[asset_nb] * (p_clsp[asset_nb] + p_mm[asset_nb] ) )

    p_bsk = reduce (lambda x,y: x + y, p_und )

    discounts = mm.DF ( mm.simulation_times[-1] )

    return np.mean ( (p_bsk - E) * (p_bsk -E > 0) ) * discounts



# basket option moment matching
def basket_mm (mm, params):

    w_v = params[0] # vector of weights
    F_v = params[1] # vector of prices
    mu_v = w_v * F_v # mu vector
    sigma_v = params[2] # vector of standard deviations
    rho_m = params[3] # matrix of correlations
    K = params[4]
    DF = params[5] # HACK HACK HACK, THIS NEEDS TO BE CORRECTED
    T  = 1. # OPTION EXPIRTY
    t = T # CHECK THIS CHECK CHECK CHECK

    def compute_third_moment (mu_v, sigma_v, rho_m, n_ln_ind ='ln'):

        # E(X^3)
        def power3 (i, mu_v, sigma_v):
            if n_ln_ind == 'n': # normal model
                return mu_v[i]**3 + 3. * mu_v[i] * sigma_v[i]**2
            else: # log-normal model
                return mu_v[i]**3 * exp (3. * sigma_v[i]**2 * t)

        # E(X Y^2 )
        def power12 (i,j, mu, sigma, rho_m):
            mu_i = mu[i]
            mu_j = mu[j]
            sigma_i = sigma[i]
            sigma_j = sigma[j]
            rho_ij = rho_m[i,j]

            if n_ln_ind =='n': # normal model
                return mu_i * mu_j**2 + mu_i * sigma_j**2 + mu_j * sigma_i * sigma_j * rho_ij
            else: # ln model
                return mu_i * mu_j**2 * exp ( ( sigma_j**2 + 2 * rho_ij * sigma_i * sigma_j ) * t )

        # E(X Y Z),
        # rho_mat is the correlation between the three variables
        # n_ln_in ... normal/log-normal indicator (n or ln, ln default)
        def power123 ( i, j, k, mu, sigma, rho_m ):
            mu_i = mu[i]
            mu_j = mu[j]
            mu_k = mu[k]
            sigma_i = sigma[i]
            sigma_j = sigma[j]
            sigma_k = sigma[k]
            rho_ij = rho_m[i,j]
            rho_ik = rho_m[i,k]
            rho_jk = rho_m[j,k]

            if n_ln_ind == 'n':
                return mu_i * mu_j * mu_k + mu_i * sigma_j * sigma_k * rho_m[1,2] + mu_j * sigma_i * sigma_j * rho_m[0,2] + \
                       mu_k * sigma_i * sigma_j * rho_m[0,1]
            else: # ln
                return mu_i * mu_j * mu_k * exp ( (sigma_i * sigma_j * rho_ij + sigma_i * sigma_k * rho_ik + sigma_j * sigma_k * rho_jk)*t )


        # E(X_1 + ... X_n )^3 = E(X^3) + 3 E(XY^2) + 6 E(XYZ)
        res1 = sum ( [ power3 (i, mu_v, sigma_v) for i in xrange (len (mu_v)) ] )
        res2 = sum ( [ power12 (i, j, mu_v, sigma_v, rho_m) \
                       for i in xrange(len(mu_v)) for j in xrange (len(sigma_v)) if i != j ] )
        # sigma and mu have to _be_ of the same size
        res3 = sum ( [ power123 (i, j, k, mu_v, sigma_v, rho_m ) \
                       for i in xrange (len(mu_v)) for j in xrange (len(sigma_v)) for k in xrange (len(sigma_v)) if i != j and i != k and j != k ] )

        return res1 + 3*res2 + res3 # divided by 6


    def compute_second_moment ( mu_v, sigma_v, rho_m, n_ln_ind='ln' ):

        # E(X^2)
        def power2 (i, mu_v, sigma_v):
            if n_ln_ind == 'n': # normal model
                return mu_v[i]**2 + sigma_v[i]**2 * t  # CHECK POWER3 AS WELL, CHECK CHECK
            else: # log-normal model
                return mu_v[i]**2 * exp (sigma_v[i]**2 * t)

        # E(X Y )
        def power12 (i,j, mu_v, sigma_v, rho_m):
            if n_ln_ind =='n': # normal model
                return rho_m[i,j] * sigma_v[i] * sigma_v[j] + mu_v[i] * mu_v[j]
            else: # ln model
                return mu_v[i] * mu_v[j] * exp ( rho_m[i,j] * sigma_v[i] * sigma_v[j] * t )

        res1 = sum( [ power2 (i, mu_v, sigma_v) for i in xrange (len (mu_v)) ] )
        res2 = sum ( [ power12 (i, j, mu_v, sigma_v, rho_m) for i in xrange (len(sigma_v)) for j in xrange (len (sigma_v)) if i != j ] )

        return res1 + res2


    M_1 = sum (mu_v)
    M_2 = compute_second_moment (mu_v, sigma_v, rho_m)
    M_3 = compute_third_moment (mu_v, sigma_v, rho_m)

    m_1 = M_1
    m_2 = M_2 - M_1**2
    m_3 = M_3 - 3. * M_2 * M_1 + 2. * M_1**3 # E(X-M_1)^3 = E X^3 - 3 E(X^2 M_1) + 3 E(X)M_1^2 - M_1^3 = M_3 - 3M_2 M_1 + 2 M_1^3

    c = m_3 / m_2**1.5
    delta = c**2 + 4.
    # numpy complains when (neg)^(1/3), so this workaround
    y_t1 = 0.5 * ( c + np.sqrt (delta) )
    y_t2 = 0.5 * ( c - np.sqrt (delta) )
    y = sign(y_t1) * abs(y_t1)**(1./3.) + sign(y_t2)* abs(y_t2) **(1./3.)
    v = np.sqrt (log (1. + y**2))
    a = np.sqrt ( m_2 / y**2)
    b = m_1 - a

    if K - b < 0 :
        return np.array([a - DF * (K-b), 1., 0., 0., - DF * log(DF)/T * K, - DF * T * K ] ) # I THINK THE LAST ONE, RHO IS INCORRECT, CHECK CHECK
    else:
        return pricers.black_greeks(a, K-b, -np.log(DF)/T, v, T, 0)  # compute final option

