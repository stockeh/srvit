import numpy as np
import gzip

# HRRR CONUS mass grid
nlon = 1799
nlat = 1059

count = nlat*nlon
shape = (nlat, nlon)
dtype = np.float32


def read_conus3_file(filename):
    f = gzip.open(filename, 'rb')
    # data = np.fromstring(f.read(),dtype=dtype,count=count).reshape(shape)
    data = np.frombuffer(f.read(), dtype=dtype, count=count).reshape(shape)
    f.close()
    return data
