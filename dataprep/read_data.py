import numpy as np

nlon = 1799
nlat = 1059
dtype = np.float32
count = nlat * nlon
shape = (nlat, nlon)


def read_data(filename):
    f = open(filename, 'rb')
    data = np.fromfile(f, dtype=dtype, count=count).reshape(shape)
    f.close()
    return data
