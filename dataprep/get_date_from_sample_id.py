import sys
from datetime import datetime

def get_date_from_sample_id(sid):

    # input: sample id of the form YYYY/jjj/II
    #        where II = 1-96
    # output: datetime object

    if len(sid) != 11:
        print('sid length error in get_date_from_sample_id',flush=True)
        sys.exit('stop here')

    dstring = sid[0:9]
    tstring = sid[9:11]

    tvalue = int(tstring)-1

    if tvalue < 0:
        print('tvalue < 0 error in get_date_from_sample_id',flush=True)
        sys.exit('stop here')
    if tvalue > 95:
        print('Exit: tvalue > 95 error in get_date_from_sample_id',flush=True)
        sys.exit('stop here')

    hour = int(tvalue/4)
    minute = int(15*(tvalue%4))

    dstring += str(hour).zfill(2) + str(minute).zfill(2)
    sdate = datetime.strptime(dstring,'%Y/%j/%H%M')

    return sdate
