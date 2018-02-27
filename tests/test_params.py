import config
import tolling_cmg

CONST_MIN = 1e-3


def test_simplest_toll_params():
    params = tolling_cmg.tolling_params()

    params.cuda = False
    params.nb_days = 31
    params.HR = 6.717
    params.hrAtMax = params.HR
    params.hrAtMin = 7.6
    params.SF = 5946.
    params.startFuel = params.SF
    params.startFuelCold = 11928.
    params.E_S = 580.
    params.MRL = 0.
    params.MDL = 400.
    params.minDisp = params.MDL
    params.MDT = 24
    params.minDownTime = params.MDT
    params.MUT = 24
    params.minRunTime = params.MUT
    params.capacity = 861.52
    params.maxCap = params.capacity

    params.fixedStartupCost = 0.
    params.fixedStartupCostCold = params.fixedStartupCost
    params.maxMonthlyStarts = 1000
    params.addFuelCost = 0.08
    params.coldStartup = 24.1
    params.startupHorizon = 16
    params.shutdownHorizon = 1000000
    params.rampUpCost = CONST_MIN
    params.rampDownCost = CONST_MIN
    params.rampUpHorizon = CONST_MIN
    params.rampDownHorizon = CONST_MIN
    params.rampRate = 1.e9
    params.VC = 2.52
    params.shutdownSPin = CONST_MIN
    params.rampUpSPin = 0.
    params.rampDownSPin = 0.
    params.fixedCostPerMonth = 0.
    params.revenue_strike = 35.e6

    return params


def test_simplest_toll_market():
    days_block = [[0, 1, 2, 3, 4], [5, 6]]
    hours_block = [[8, 16], [8, 16]]
    hours_block_names = [['offpeak_7x8', 'peak'],
                         ['offpeak_7x8', 'offpeak_2x16']]
    days_block_names = ['weekday', 'weekend']
    power_bl_names = [['ATSI_7X8', 'ATSI-PEAK'],
                      ['ATSI_7X8', 'ATSI_2X16']]
    fuel_idx_name = 'NG_MICHCON_GD-PEAK'
    cash_vols = {'power': [['PJMW-OFFPEAK_CV', 'PJMW-PEAK_CV'],
                           ['PJMW-OFFPEAK_CV', 'PJMW-PEAK_CV']],
                 'fuel': [['NG_MICHCON_CASHVOL', 'NG_MICHCON_CASHVOL'],
                          ['NG_MICHCON_CASHVOL', 'NG_MICHCON_CASHVOL']]}

    return {'days_block': days_block,
            'hours_block': hours_block,
            'hours_block_names': hours_block_names,
            'days_block_names': days_block_names,
            'power_bl_names': power_bl_names,
            'fuel_idx_name': fuel_idx_name,
            'cash_vols': cash_vols}