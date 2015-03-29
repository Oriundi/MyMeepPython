#! /usr/bin/env python2

import meep_mpi as meep

# import math
import numpy
import matplotlib.pyplot as plt
from scipy.constants import c, pi

from exeptions import *

res = 10.0
gridSizeX = 16.0
gridSizeY = 16.0
wgLengthX = gridSizeX
wgWidth = 2.0   # width of the waveguide
wgHorYCen = 0
srcFreqCenter = 0.15   # gaussian source center frequency
srcPulseWidth = 0.1   # gaussian source pulse width
srcComp = meep.Hz   # gaussian source component
mirrorDirection = meep.Y
mirrorFactor = complex(-1, 0)

lambda_rec = 0.633   # um
k = 2 * pi / lambda_rec
lambda_read = 0.633
freq_read = c / lambda_read
rz = 1   # m
rx = [0.1, -0.1, -0.1, 0.1]
ry = [0.1, 0.1, -0.1, -0.1]
amp = [1, 1, 1, 1]
phase = [0, 0, 0, 0]
modulation = 0.01

vol = meep.vol2d(gridSizeX, gridSizeY, 1 / res)

if len(rx) != len(ry):
    raise DimSizeError('Size of rx and ry is different. Halt')
    # break

r = numpy.zeros([len(rx)])
cos_fi = numpy.zeros([len(rx)])
sin_fi = numpy.zeros([len(rx)])
cos_theta = numpy.zeros([len(rx)])
sin_theta = numpy.zeros([len(rx)])
kz = numpy.zeros([len(rx)])
kx = numpy.zeros([len(rx)])
ky = numpy.zeros([len(rx)])

for ii in range(0, len(rx)):
    r[ii] = numpy.sqrt(rx[ii] ** 2 + ry[ii] ** 2)
    cos_fi[ii] = rx[ii] / r[ii]
    sin_fi[ii] = ry[ii] / r[ii]
    cos_theta[ii] = rz / numpy.sqrt(rz ** 2 + r[ii] ** 2)
    sin_theta[ii] = r[ii] / numpy.sqrt(rz ** 2 + r[ii] ** 2)

    kz[ii] = k * cos_theta[ii]
    kx[ii] = k * sin_theta[ii] * cos_fi[ii]
    ky[ii] = k * sin_theta[ii] * sin_fi[ii]


# this function plots the waveguide material as a function of a vector(X,Y)
class epsilon(meep.CallbackMatrix2D):
    def __init__(self):
        meep.CallbackMatrix2D.__init__(self)
        meep.master_printf("Creating the material matrix....\n")
        self.meep_resolution = int(res)
        eps_matrix = numpy.zeros([gridSizeX * res, gridSizeY * res], dtype=complex)
        # _eps_matrix = numpy.zeros([gridSizeX * res, gridSizeY * res], dtype=float)
        len_x = eps_matrix.shape[0]
        len_y = eps_matrix.shape[1]

        print("len_x = %i, len_y = %i" % (len_x, len_y))

        for nn in range(0, len(rx)):
            for x in range(0, len_x):
                for y in range(0, len_y):
                    eps_matrix[x, y] = eps_matrix[x, y] + amp[nn] * numpy.exp(-1j *
                        numpy.sqrt((rx[nn] + x / res) ** 2 + (rz + y / res) ** 2) *
                        numpy.sqrt((kx[nn] + x / kx[nn] / res) ** 2 + (kz[nn] + y / kz[nn] / res) ** 2) +
                        1j * phase[nn])

        eps_matrix = numpy.absolute(eps_matrix / numpy.amax(eps_matrix)) ** 2 * modulation
        print(eps_matrix[10, 10])

        # grating = numpy.abs(eps_matrix) ** 2
        # plt.figure(1)
        # plt.imshow(_eps_matrix, cmap='hot', extent=[0, gridSizeX, 0, gridSizeY])
        # plt.colorbar()
        # plt.show()

        meep.master_printf("Setting the material matrix...\n")
        self.set_matrix_2D(eps_matrix, vol)
        # self.setMatrix(grating)
        self.stored_eps_matrix = eps_matrix    # to prevent the garbage collector from cleaning up the matrix...
        meep.master_printf("MeepMaterial object initialized.\n")

meep.set_EPS_Callback(epsilon().__disown__())
struct = meep.structure(vol, EPS, no_pml())

fld = meep.fields(struct)
fld.add_volume_source(Ex, gaussian_src_time(freq_read / c, 1.5e9 / c), vol)

while fld.time() / c < 30e-15:
    fld.step()

meep.print_f.get_field(Ex, meep.vec(0.5e-6, 0.5e-6, 3e-6))

meep.del_EPS_Callback()
