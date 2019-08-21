import numpy

##
# Closed-loop maintenance of a dark hole in a coronagraph (based on https://arxiv.org/abs/1902.01880) via a combination of wavefront control (Electric Field Conjugation - EFC) and estimation (Extended Kalman Filter - EKF).
# The example compares the contrast in the dark hole as the wavefront error accumulate: in the absence of control (open loop) the contrast deteriorates; with EFC and EKF (closed loop) - the contrast is slightly lower to begin with but remains constant.
#
# The code is based on a FACLO model of the WFIRST coronagraph (https://github.com/ajeldorado/falco-matlab).
# To make this example independent of FALCO, a linearized model of the WFIRST dark hole is used.
# The dark hole was created by FALCO (at five different wavelength) and is considered the "nominal state" of this simulation.
# The first order effects of deformable mirror (DM) and the effects of wavefront errors (WFE) at the nominal state were computed using FALCO and are given by dm_influence_matrix and wfe_influence_matrix.
#
# To visualize the intensity, one has to map the 1D pixels vector onto their position in the 2D image.
# This can be done as follows:
# img = numpy.load("pixel_mask.npy").astype(numpy.float)
# n_pixels = int(numpy.sum(img))
# img[numpy.where(img)] = get_I_closed_loop()[:n_pixels] #first wavelength
# matplotlib.pyplot.imshow(img)
##
DRIFT_MAGNITUDE = 3.2e-11 #(sigma_d)
DITHER_MAGNITUDE = 5e-3 #(sigma_u)

class LinearTelescopeModel:
    def __init__(self):
        self.dm_influence_matrix = numpy.load("dm_influence_matrix_x_1890000.npy")/1890000.0 #DM Jacobian stored as int8 to save space and scaled back to its original range
        self.wfe_influence_matrix = numpy.load("wfe_influence_matrix.npy") #This matrix is for simulating speckles drift
        self.E_dark_hole = numpy.load("E_dark_hole.npy") #Electric field after digging the dark hole
        self.dark_current = 0.25
        self.intensity_to_photons = 5e9 #A scaling factor to convert from electric field to average number of photons per frame
        self.cmd = numpy.zeros(self.dm_influence_matrix.shape[1]) #DM command (0 at dark hole setting)
        self.wfe_coeff = numpy.zeros(self.wfe_influence_matrix.shape[1]) #The state of the wavefront errors as they drift
        self.last_image = None

        max_zernikes_order = 6
        drift_std = [DRIFT_MAGNITUDE/(p+1)**2 for p in range(max_zernikes_order) for _ in range(p+1)] #Standard deviation of incrments of Zernike coefficient should decrease with polynomial order
        self.drift_covariance = numpy.diag(numpy.square(drift_std))

    def advance(self, dm_command):
        self.cmd = dm_command
        self.wfe_coeff += numpy.random.multivariate_normal(numpy.zeros(self.wfe_coeff.shape), self.drift_covariance)#WFE random walk
        self.last_image = None

    def get_E_open_loop(self):
        return self.E_dark_hole + self.wfe_influence_matrix.dot(self.wfe_coeff) #Elecrtic field if the DM were fixed

    def get_E_closed_loop(self):
        return self.get_E_open_loop() + self.dm_influence_matrix.dot(self.cmd)

    def get_I_open_loop(self):
        E_OL = self.get_E_open_loop()
        return E_OL[::2]**2 + E_OL[1::2]**2
        
    def get_I_closed_loop(self):
        E_CL = self.get_E_closed_loop()
        return E_CL[::2]**2 + E_CL[1::2]**2

    def get_last_image(self):
        if self.last_image is None:
            I = self.get_I_closed_loop()
            I_photons = I*self.intensity_to_photons + self.dark_current #Total effective intensity in photons per frame. Incoherent light is ignored for simplicity.
            self.last_image = numpy.random.poisson(I_photons)

        return self.last_image

tm = LinearTelescopeModel()
print("Computing EFC gain")
EFC_gain = numpy.linalg.inv(tm.dm_influence_matrix.T.dot(tm.dm_influence_matrix) + numpy.eye(tm.dm_influence_matrix.shape[1])*1e-8).dot(tm.dm_influence_matrix.T)

SS = 2#Pixel state size. Two for real and imaginary parts of the electric field. If incoherent intensity is not ignored, SS should be 3 and the EKF modified accordingly.
BS = SS*1#EKF block size - number of pixels per EKF (currently 1). Computation time grows as the cube of BS.
SL = EFC_gain.shape[1]#Total length of the sate vector (all pixels).

#3D matrices that include all the 2D EKF matrices for all pixels at once
H = numpy.zeros((SL//BS,BS//SS,BS))
H_indices = numpy.where([numpy.kron(numpy.eye(BS//SS), numpy.ones(SS)) for _ in range(len(H))])

R = numpy.zeros((SL//BS,BS//SS,BS//SS))
R_indices = numpy.where([numpy.eye(BS//SS) for _ in range(len(H))])

#The drift covariance matrix for each pixel (or block of pixels). Needs to be estimated if the model is not perfectly known.
block_influence_matrix = tm.wfe_influence_matrix.reshape((-1,BS,len(tm.drift_covariance)))
Q = numpy.matmul(block_influence_matrix.dot(tm.drift_covariance), block_influence_matrix.transpose(0,2,1))*tm.intensity_to_photons

#For simplicity, start with a perfect estimate (presumably obtained during them dark hole digging stage)
x_hat = tm.E_dark_hole*tm.intensity_to_photons**0.5 #Rhe estimate of the electric field (x_hat) is scaled for convenience
P = Q*0.0

u = numpy.zeros(EFC_gain.shape[0])

for t in range(512):
    E_open_loop = tm.get_E_open_loop()
    EKF_err = numpy.sum(numpy.abs(x_hat/tm.intensity_to_photons**0.5 - E_open_loop))/numpy.sum(numpy.abs(E_open_loop))
    print ("frame %d; open loop contrast %.2e; closed loop contrast %.2e; EKF avg. error %d%%"%(t, numpy.mean(tm.get_I_open_loop()), numpy.mean(tm.get_I_closed_loop()), EKF_err*100))

    tm.advance(u)#This simulates applying control and taking an image

    y_measured = tm.get_last_image() #Measured photon counts

    x_hat_CL = x_hat + tm.dm_influence_matrix.dot(u)*tm.intensity_to_photons**0.5 #Estimate of the closed loop electric field
    y_hat = x_hat_CL[::2]**2 + x_hat_CL[1::2]**2 + tm.dark_current #Predicted measurement

    #Standard EKF equations applied to all pixels at once
    H[H_indices] = 2*x_hat_CL
    R[R_indices] = y_hat
    H_T = H.transpose(0,2,1)

    P = P + Q
    P_H_T = numpy.matmul(P, H_T)
    S = numpy.matmul(H, P_H_T) + R
    S_inv = numpy.linalg.inv(S)
    K = numpy.matmul(P_H_T, S_inv)
    P = P - numpy.matmul(P_H_T, K.transpose(0,2,1))

    dx_hat = numpy.matmul(K, (y_measured - y_hat).reshape((-1,BS//SS,1))).reshape(-1) #Correction
    x_hat = x_hat + dx_hat

    dither = numpy.random.normal(0, DITHER_MAGNITUDE, len(u))
    
    u = -EFC_gain.dot(x_hat/tm.intensity_to_photons**0.5) + dither #Closed loop control: EFC based on electric field estimate and dither (to keep EKF stable)
    ##
    # In the case of non-linear model, the base dark hole DM command needs to be periodically shifted (recalibrated every n~=50 time steps).
    # One way tp ahieve this is to replace tm.advance(u) with tm.advance(u0+u), where u0 is the shifted base command which is updated every once in a while via
    # u0 += -EFC_gain.dot(x_hat/tm.intensity_to_photons**0.5)
    # The state estimate should also be shifted accordingly:
    # x_hat += -tm.dm_influence_matrix.dot(u0)*tm.intensity_to_photons**0.5
    ##
