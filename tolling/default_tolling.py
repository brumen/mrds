# default values of the tolling params


def tolling_params_default():
    """ Returns the default tolling params.
        """

    return { 'hrAtMax'      : 6.
           , 'hrAtMin'      : 7.
           , 'maxCap'       : 1000.  # maximum capacity
           , 'minDisp'      : 100.  #  minimum dispatch
           , 'startFuel'    : 10.  # startFuel - startup fuel
           , 'startFuelCold': 15.  # startFuelCold - startup fuel from cold.
           , 'addFuelCost'  : 5.  #  - added fuel costs
           , 'VC'                  : 10.  # VC - variable costs
           , 'rampRate'            : 3.
           , 'shutdownSPin'        : 0.1  # shutdown shadow price in
           , 'minDownTime'         : 8  # minimum downtime
           , 'minRunTime'          : 16  # minimum run time.
           , 'fixedStartupCost'    : 10.
           , 'fixedStartupCostCold': 10.
           , 'maxMonthlyStarts': 5
           , 'coldStartup'     : 10.
           , 'startupHorizon'  : 16
           , 'shutdownHorizon' : 16
           , 'rampUpSPin'      : 10.
           , 'rampDownSPin'    : 10.
           , 'rampUpCost'      : 10.
           , 'rampDownCost'    : 10.
           , 'rampUpHorizon'   : 15.
           , 'rampDownHorizon' : 25.
           , }


def default_partitions(peak_name : str ='WTI', offpeak_name : str ='BRENT'):
    """ Returns a sample partition of days/hours.

    :param peak_name: peak curve name
    :param offpeak_name: offpeak curve name
    :returns:
    """

    days_partition  = { 'WEEKDAY': (0, 1, 2, 3, 4,), 'WEEKEND': (5, 6,) }

    hours_partition = { 'WEEKDAY': [(peak_name, 8), (offpeak_name, 16), ]
                       , 'WEEKEND': [(peak_name, 16), (offpeak_name, 8), ] }

    return days_partition, hours_partition