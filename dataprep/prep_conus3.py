import numpy as np
from datetime import datetime
import os
import sys

from read_data import read_data
from read_conus3_file import read_conus3_file
from get_date_from_sample_id import get_date_from_sample_id

stime = datetime.now()

#####

if sys.argv[1] == 'A':
    areg = 'A'
    ascl = 'A'
    skip_cold_season = True
elif sys.argv[1] == 'B':
    areg = 'B'
    ascl = 'A'
    skip_cold_season = True
else:
    sys.exit('arg not found')

print('running prep_jason_conus3 for ', areg,
      ascl, skip_cold_season, flush=True)

#####

outfile = 'conus3/jason_conus3/sample_numbers_dates_reg'+areg+'_scl'+ascl+'.txt'
print('opening ', outfile, flush=True)
fout = open(outfile, 'w')

#####

bad = []
f = open('bad_samples_all_conus3.txt', 'r')
for aline in f:
    bad.append(aline.strip())
f.close()

dates = []
sfile = 'all_conus3_samples.txt'
# sfile = 'sample_list_eval.txt'
f = open(sfile, 'r')
for aline in f:
    adate = aline.strip().replace('"', '')
    if adate in bad:
        dates.append(None)
    else:
        dates.append(adate)
f.close()

#####

scalings = {}
scalings['A'] = {}
scalings['A']['c07'] = {'min': 200., 'max': 300.}
scalings['A']['c09'] = {'min': 200., 'max': 250.}
scalings['A']['c13'] = {'min': 200., 'max': 300.}
scalings['A']['glm'] = {'min': 0.1, 'max': 50.}
scalings['A']['mrms'] = {'min': 0., 'max': 60.}

#####

# hrrr
nlon = 1799
nlat = 1059

regions = {}
regions['A'] = {'nx': 1536, 'ny': 768, 'yoff': 146, 'xoff': -110}
regions['B'] = {'nx': 896, 'ny': 768, 'yoff': 130, 'xoff': -250}

for akey in regions:
    yoff = regions[akey]['yoff']
    xoff = regions[akey]['xoff']
    ny = regions[akey]['ny']
    nx = regions[akey]['nx']
    regions[akey]['iy0'] = 0 + yoff
    regions[akey]['iy1'] = ny + yoff
    regions[akey]['ix0'] = nlon - nx + xoff
    regions[akey]['ix1'] = nlon + xoff

#####

nchan_in = 4
ny = regions[areg]['ny']
nx = regions[areg]['nx']

xdata = np.zeros((ny, nx, nchan_in), dtype=np.float32)
ydata = np.zeros((ny, nx), dtype=np.float32)

fcnt = {'train': 1, 'test': 1, 'val': 1}

for adate in dates:

    if adate == None:
        print('skipping bad sample', flush=True)
        continue

    thedate = get_date_from_sample_id(adate)
    print(adate, thedate, flush=True)

    if skip_cold_season:
        if thedate.month < 4:
            continue
        if thedate.month > 9:
            continue

    if thedate.year in [2018, 2019]:
        c07file = 'conus3/'+adate+'_ABIC07.bin'
        c09file = 'conus3/'+adate+'_ABIC09.bin'
        c13file = 'conus3/'+adate+'_ABIC13.bin'
        glmfile = 'conus3/'+adate+'_GLMGED.bin'
        mrmsfile = 'conus3/'+adate+'_MRMSREFC.bin'
        if not os.path.exists(c07file):
            continue
        if not os.path.exists(c09file):
            continue
        if not os.path.exists(c13file):
            continue
        if not os.path.exists(glmfile):
            continue
        if not os.path.exists(mrmsfile):
            continue
        c07 = read_data(c07file)
        c09 = read_data(c09file)
        c13 = read_data(c13file)
        glm = read_data(glmfile)
        mrms = read_data(mrmsfile)
    else:
        sfolder = '/mnt/lhmlnas/conus3/yoonjin_conus3/' + \
            thedate.strftime('%Y/%m%d/')
        sdate = thedate.strftime('%Y%m%d%H%M')
        c07file = sfolder+'ABI_C07_'+sdate+'.bin.gz'
        c09file = sfolder+'ABI_C09_'+sdate+'.bin.gz'
        c13file = sfolder+'ABI_C13_'+sdate+'.bin.gz'
        glmfile = sfolder+'GLM_GED_'+sdate+'.bin.gz'
        mrmsfile = sfolder+'MRMS_REFC_'+sdate+'.bin.gz'
        if not os.path.exists(c07file):
            continue
        if not os.path.exists(c09file):
            continue
        if not os.path.exists(c13file):
            continue
        if not os.path.exists(glmfile):
            continue
        if not os.path.exists(mrmsfile):
            continue
        c07 = read_conus3_file(c07file)
        c09 = read_conus3_file(c09file)
        c13 = read_conus3_file(c13file)
        glm = read_conus3_file(glmfile)
        mrms = read_conus3_file(mrmsfile)

    c07 = c07[regions[areg]['iy0']:regions[areg]['iy1'],
              regions[areg]['ix0']:regions[areg]['ix1']]
    c09 = c09[regions[areg]['iy0']:regions[areg]['iy1'],
              regions[areg]['ix0']:regions[areg]['ix1']]
    c13 = c13[regions[areg]['iy0']:regions[areg]['iy1'],
              regions[areg]['ix0']:regions[areg]['ix1']]
    glm = glm[regions[areg]['iy0']:regions[areg]['iy1'],
              regions[areg]['ix0']:regions[areg]['ix1']]
    mrms = mrms[regions[areg]['iy0']:regions[areg]['iy1'],
                regions[areg]['ix0']:regions[areg]['ix1']]

    bad = (c07 <= 0) | (c09 <= 0) | (c13 <= 0) | (glm < 0) | (mrms <= -999)

    c07 = (scalings[ascl]['c07']['max'] - c07) / (scalings[ascl]
                                                  ['c07']['max'] - scalings[ascl]['c07']['min'])
    c09 = (scalings[ascl]['c09']['max'] - c09) / (scalings[ascl]
                                                  ['c09']['max'] - scalings[ascl]['c09']['min'])
    c13 = (scalings[ascl]['c13']['max'] - c13) / (scalings[ascl]
                                                  ['c13']['max'] - scalings[ascl]['c13']['min'])
    glm = (glm - scalings[ascl]['glm']['min']) / (scalings[ascl]
                                                  ['glm']['max'] - scalings[ascl]['glm']['min'])
    mrms = (mrms - scalings[ascl]['mrms']['min']) / \
        (scalings[ascl]['mrms']['max'] - scalings[ascl]['mrms']['min'])

    c07[c07 < 0] = 0
    c09[c09 < 0] = 0
    c13[c13 < 0] = 0
    glm[glm < 0] = 0
    mrms[mrms < 0] = 0

    c07[c07 > 1] = 1
    c09[c09 > 1] = 1
    c13[c13 > 1] = 1
    glm[glm > 1] = 1
    mrms[mrms > 1] = 1

    c07[bad] = 0
    c09[bad] = 0
    c13[bad] = 0
    glm[bad] = 0
    mrms[bad] = 0

    xdata[:, :, 0] = c07
    xdata[:, :, 1] = c09
    xdata[:, :, 2] = c13
    xdata[:, :, 3] = glm

    ydata = mrms

    afolder = None
    if thedate.year in [2018, 2019, 2020]:
        afolder = 'train'
    elif thedate.year == 2021:
        afolder = 'val'
    elif thedate.year == 2022:
        afolder = 'test'

    outfile = 'conus3/jason_conus3/'+afolder+'/conus3_reg' + \
        areg+'_scl'+ascl+'_'+str(fcnt[afolder]).zfill(6)+'.npz'
    print('writing ', outfile, flush=True)

    np.savez(outfile, xdata=xdata, ydata=ydata)

    fout.write(afolder+' '+str(fcnt[afolder]).zfill(6) +
               ' '+thedate.strftime('%Y-%m-%d_%H:%MZ')+'\n')

    fcnt[afolder] += 1

fout.close()

etime = datetime.now()
print('seconds ellapsed = ', (etime-stime).total_seconds(), flush=True)
