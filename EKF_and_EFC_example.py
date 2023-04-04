import numpy
import matplotlib.pyplot as plt
from matplotlib import animation

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
DITHER_MAGNITUDE = 5e-3 #(sigma_u) #Original value: 5e-3 with IFS on
USE_IFS = False
N_WAVELENGTHS = 5
N_PIXELS = 2608

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

        if not USE_IFS:
            I_OL = numpy.zeros((N_PIXELS,))
            #Separate real and imaginary part of E-field
            E_OL_re = E_OL[::2]
            E_OL_im = E_OL[1::2]
            #Sum over wavelengths
            for i in range(N_WAVELENGTHS):
                I_OL += E_OL_re[i*N_PIXELS:(i+1)*N_PIXELS]**2
                I_OL += E_OL_im[i*N_PIXELS:(i+1)*N_PIXELS]**2
            return I_OL

        return E_OL[::2]**2 + E_OL[1::2]**2
        
    def get_I_closed_loop(self):
        E_CL = self.get_E_closed_loop() #26080 x 1

        if not USE_IFS:
            I_CL = numpy.zeros((N_PIXELS,))
            #Separate real and imaginary part of E-field
            E_CL_re = E_CL[::2]
            E_CL_im = E_CL[1::2]
            #Sum over wavelengths
            for i in range(N_WAVELENGTHS):
                I_CL += E_CL_re[i*N_PIXELS:(i+1)*N_PIXELS]**2
                I_CL += E_CL_im[i*N_PIXELS:(i+1)*N_PIXELS]**2
            return I_CL

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
if USE_IFS:
    H = numpy.zeros((SL//BS,BS//SS,BS))
    H_indices = numpy.where([numpy.kron(numpy.eye(BS//SS), numpy.ones(SS)) for _ in range(len(H))])
    R = numpy.zeros((SL//BS,BS//SS,BS//SS))
    R_indices = numpy.where([numpy.eye(BS//SS) for _ in range(len(H))])

else:
    #H = numpy.zeros((N_PIXELS, N_PIXELS*N_WAVELENGTHS*2))
    #R = numpy.zeros((N_PIXELS, N_PIXELS))
    H = numpy.zeros((N_PIXELS, 1, N_WAVELENGTHS*2))
    R = numpy.zeros((N_PIXELS, 1, 1))

#The drift covariance matrix for each pixel (or block of pixels). Needs to be estimated if the model is not perfectly known.
block_influence_matrix = tm.wfe_influence_matrix.reshape((-1,BS,len(tm.drift_covariance)))
Q = numpy.matmul(block_influence_matrix.dot(tm.drift_covariance), block_influence_matrix.transpose(0,2,1))*tm.intensity_to_photons

if not USE_IFS:
    new_Q = numpy.zeros((N_PIXELS, N_WAVELENGTHS*2, N_WAVELENGTHS*2))
    for i in range(N_PIXELS):
        for wv in range(N_WAVELENGTHS):
            new_Q[i, wv*2, wv*2] = Q[i+wv*N_PIXELS, 0, 0]
            new_Q[i, wv*2+1, wv*2+1] = Q[i+wv*N_PIXELS, 1, 1]
            new_Q[i, wv*2, wv*2+1] = Q[i+wv*N_PIXELS, 0, 1]
            new_Q[i, wv*2+1, wv*2] = Q[i+wv*N_PIXELS, 1, 0]
    """
    new_Q = numpy.zeros((N_PIXELS*N_WAVELENGTHS*2, N_PIXELS*N_WAVELENGTHS*2))
    for i in range(N_PIXELS*N_WAVELENGTHS):
        new_Q[i*2, i*2] = Q[i, 0, 0]
        new_Q[i*2 +1, i*2 + 1] = Q[i, 1, 1]
        new_Q[i*2+1, i*2] = Q[i, 1, 0]
        new_Q[i*2, i*2+1] = Q[i, 0, 1]
    """
    Q = new_Q

#For simplicity, start with a perfect estimate (presumably obtained during them dark hole digging stage)
x_hat = tm.E_dark_hole*tm.intensity_to_photons**0.5 #Rhe estimate of the electric field (x_hat) is scaled for convenience
P = Q*0.0

u = numpy.zeros(EFC_gain.shape[0])
EKF_errs = []
n_frames = 512
E_CL = numpy.zeros((tm.get_E_closed_loop().shape[0], n_frames))
E_OL = numpy.zeros((tm.get_E_closed_loop().shape[0], n_frames))
I_OL = numpy.zeros((tm.get_I_open_loop().shape[0], n_frames))
I_CL = numpy.zeros((tm.get_I_closed_loop().shape[0], n_frames))
images = numpy.zeros((tm.get_last_image().shape[0], n_frames))

save_dir = 'data'
rerun=True

if rerun:
    for t in range(n_frames):
        E_open_loop = tm.get_E_open_loop()
        E_CL[:, t] = tm.get_E_closed_loop()
        E_OL[:, t] = E_open_loop
        EKF_err = numpy.sum(numpy.abs(x_hat/tm.intensity_to_photons**0.5 - E_open_loop))/numpy.sum(numpy.abs(E_open_loop))
        EKF_errs.append(EKF_err)

        print ("frame %d; open loop contrast %.2e; closed loop contrast %.2e; EKF avg. error %d%%"%(t, numpy.mean(tm.get_I_open_loop()), numpy.mean(tm.get_I_closed_loop()), EKF_err*100))

        tm.advance(u)#This simulates applying control and taking an image

        y_measured = tm.get_last_image() #Measured photon counts
        images[:, t] = y_measured
        I_CL[:, t] = tm.get_I_closed_loop()
        I_OL[:, t] = tm.get_I_open_loop()

        x_hat_CL = x_hat + tm.dm_influence_matrix.dot(u)*tm.intensity_to_photons**0.5 #Estimate of the closed loop electric field
        if not USE_IFS:
            y_hat = numpy.zeros((N_PIXELS,))
            for i in range(N_WAVELENGTHS):
                y_hat += x_hat_CL[::2][i*N_PIXELS:(i+1)*N_PIXELS]**2
                y_hat += x_hat_CL[1::2][i*N_PIXELS:(i+1)*N_PIXELS]**2
            y_hat += tm.dark_current
        else:
            y_hat = x_hat_CL[::2]**2 + x_hat_CL[1::2]**2 + tm.dark_current #Predicted measurement

        #Standard EKF equations applied to all pixels at once
        if not USE_IFS:
            #for i in range(N_PIXELS):
            #    for wv in range(N_WAVELENGTHS):
            #        H[i, wv*N_WAVELENGTHS + i] = x_hat_CL[::2][i]
            #        H[i, wv*N_WAVELENGTHS + i + 1] = x_hat_CL[1::2][i]
            #R = numpy.diag(y_hat)
            for i in range(N_PIXELS):
                for wv in range(N_WAVELENGTHS):
                    H[i, 0, wv*2] = 2*x_hat_CL[::2][i+wv*N_PIXELS]
                    H[i, 0, wv*2 + 1] = 2*x_hat_CL[1::2][i+wv*N_PIXELS]
            R = y_hat.reshape((N_PIXELS, 1, 1))
        else:
            H[H_indices] = 2*x_hat_CL
            R[R_indices] = y_hat

        H_T = H.transpose(0,2,1)
        P = P + Q
        P_H_T = numpy.matmul(P, H_T)
        S = numpy.matmul(H, P_H_T) + R
        S_inv = numpy.linalg.inv(S)
        K = numpy.matmul(P_H_T, S_inv)
        P = P - numpy.matmul(P_H_T, K.transpose(0,2,1))

        if USE_IFS:
            dx_hat = numpy.matmul(K, (y_measured - y_hat).reshape((-1,BS//SS,1))).reshape(-1) #Correction
        else:
            dx_hat_reshape = numpy.matmul(K, (y_measured - y_hat).reshape((-1, BS//SS, 1)))
            dx_hat = numpy.zeros((x_hat.shape))
            dx_hat_real = numpy.zeros((N_WAVELENGTHS*N_PIXELS))
            dx_hat_im = numpy.zeros((N_WAVELENGTHS*N_PIXELS))
            for wv in range(N_WAVELENGTHS):
                dx_hat_real[wv*N_PIXELS:(wv+1)*N_PIXELS] = dx_hat_reshape[:, wv*2, 0]
                dx_hat_im[wv*N_PIXELS:(wv+1)*N_PIXELS] = dx_hat_reshape[:, wv*2+1, 0]
            dx_hat[::2] = dx_hat_real
            dx_hat[1::2] = dx_hat_im

        x_hat = x_hat + dx_hat

        dither = numpy.random.normal(0, DITHER_MAGNITUDE, len(u))
        
        u = -EFC_gain.dot(x_hat/tm.intensity_to_photons**0.5) + dither #Closed loop control: EFC based on electric field estimate and dither (to keep EKF stable)

        ##
        # In the case of non-linear model, the base dark hole DM command needs to be periodically shifted (recalibrated every n~=50 time steps).
        # One way tp ahieve this is to replace tm.advance(u) with tm.advance(u0+u), where u0 is the shifted base command which is updated every once in a while via
        # The state estimate should also be shifted accordingly:
        #x_hat += -tm.dm_influence_matrix.dot(u0)*tm.intensity_to_photons**0.5

    #Save data
    numpy.savetxt(f"{save_dir}/images.csv", images, delimiter=",")
    numpy.savetxt(f"{save_dir}/I_CL.csv", I_CL, delimiter=",")
    numpy.savetxt(f"{save_dir}/I_OL.csv", I_OL, delimiter=",")
    numpy.savetxt(f"{save_dir}/E_CL.csv", E_CL, delimiter=",")
    numpy.savetxt(f"{save_dir}/E_OL.csv", E_OL, delimiter=",")
    numpy.savetxt(f"{save_dir}/EKF_err.csv", numpy.array(EKF_errs), delimiter=",")

"""
Questions
-----------

- How does the size of I/E (which I think is # of pixels) relate to the final conversion to the 68x68 image for plotting?
    Because there are different wavelengths (5 or 7) and we are only plotting the first wavelength

- Is the DM influence matrix the same as G (the Jacobian of the coronagraph operator) from the paper?
- Why are there 21 WFE coefficients?

List of matrix sizes
---------------------

E, I, u
----------
u: (3469,) ------- Commanded DM actuations
    Size comes from # of actuators in the DM
dither: (3469,) -- Random additions to DM actuations to stabilize EKF
    Size comes from u
E_CL: (26080,) --- Closed-loop electric field, equal to open-loop E_field + E_field resulting from DM control
    Size comes from # of pixels * 2 (to capture real & imaginary components)
E_OL: (26080,) --- Open-loop electric field, equal to E_field for dark hole + E_field resulting from WFE
    Size comes from # of pixels * 2 (to capture real & imaginary components)
I_CL: (13040,) --- Closed-loop intensity, equal to closed-loop E_field**2 
    Size comes from # of pixels
I_OL: (13040,) --- Open-loop intensity, equal to open-loop E_field**2
    Size comes from # of pixels
Telescope image: (13040,) --- Image simulating photons incident on detector, by multiplying I_CL by the constant converting
    intensity to photons, and adding dark current to get the mean intensity, then modelling the number of photons using a 
    Poisson process centered at the mean intensity

    Size comes from # of pixels

WFE and DM coefficients
------------------------
DM influence matrix: (26080, 3469) --- dx/du, Jacobian of E-field with respect to DM command 
    Size comes from E-field size + # of DM actuators
WFE influence matrix: (26080, 21) ---- dx/d(WFE) Jacobian of E-field with respect to wavefront error
    Size comes from E-field size + # of WFE coefficients
Block influence matrix: (13040, 2, 21) -- Reshaped WFE influence matrix to break out real & imaginary components of E-field
    Size comes from E-field size + # of WFE coefficients
EFC gain: (3469, 26080) -- State-space control gain matrix, such that optimal DM actuations (excluding dither) = -EFC_gain*E-field
    Size comes from E_field size + # of DM actuators
WFE coeff: (21,) --- WFE coefficients, such that dx/d(WFE) = WFE influence matrix*WFE coeff
    Size is just the # of WFE coefficients, which drift randomly

EKF
--------
Q: (13040, 2, 2) --- Process noise covariance (for EKF), describes noise resulting from model inaccuracies
    Size comes from E-field size (broken up into real + imaginary components here)
R: (13040, 1, 1) --- Observation noise covariance (for EKF), describes noise on image
    Size comes from # of pixels
H: (13040, 1, 2) --- dh/dx (for EKF), Jacobian of observation (image) with respect to state (E-field)
    Size comes from # of pixels + E-field size
P: (13040, 2, 2) --- State covariance (for EKF), estimates covariance of state (E-field)
    Size comes from E-field size (broken up into real + imaginary components here)
S: (13040, 1, 1) --- Innovation covariance (for EKF), describes covariance of (y_hat - telescope image)
    Size comes from # of pixels
K: (13040, 2, 1) --- Kalman gain (for EKF), used to update state in EKF update step (x_hat = x_hat + K * innovation)
    Size comes from # of pixels + E-field size
x_hat: (26080,) ---- State estimate, for electric field
    Size comes from E-field size
y_hat: (13040,) ---- Expected observation, for intensity
    Size comes from # of pixels
"""
#Load data and plot
EKF_errs = numpy.loadtxt(f"{save_dir}/EKF_err.csv", delimiter=",")
tm_images = numpy.loadtxt(f"{save_dir}/images.csv", delimiter=",")
I_CL = numpy.loadtxt(f"{save_dir}/I_CL.csv", delimiter=",")
I_OL = numpy.loadtxt(f"{save_dir}/I_OL.csv", delimiter=",")

plt.figure()
plt.plot(EKF_errs)
plt.title("2a: Single-pixel estimation error")
plt.show()

#Plotting image with error
fig = plt.figure()
img = numpy.load("pixel_mask.npy").astype(float)
n_pixels = int(numpy.sum(img))
img[numpy.where(img)] = tm_images[:n_pixels, 0]
im = plt.imshow(img)
plt.title(f"2b: Field image with error (frame 1 out of {n_frames})")
plt.colorbar()

def animate(i):
    img = numpy.load("pixel_mask.npy").astype(float)
    n_pixels = int(numpy.sum(img))
    img[numpy.where(img)] = tm_images[:n_pixels, i] #first wavelength
    im.set_data(img)
    plt.title(f"2b: Field image with error (frame {i+1} out of {n_frames})")
    return [im] 

anim = animation.FuncAnimation(fig, animate, frames=n_frames, interval=200)
plt.show()

#Plotting CL intensity with error
fig = plt.figure()
img = numpy.load("pixel_mask.npy").astype(float)
n_pixels = int(numpy.sum(img))
img[numpy.where(img)] = I_CL[:n_pixels, 0]
im = plt.imshow(img)
plt.title(f"2c: Closed-loop Intensity (frame 1 out of {n_frames})")
plt.colorbar()

def animate(i):
    img = numpy.load("pixel_mask.npy").astype(float)
    n_pixels = int(numpy.sum(img))
    img[numpy.where(img)] = I_CL[:n_pixels, i] #first wavelength
    im.set_data(img)
    plt.title(f"2c: Closed-loop Intensity (frame {i+1} out of {n_frames})")
    return [im] 

anim = animation.FuncAnimation(fig, animate, frames=n_frames, interval=200)
plt.show()

#Plotting OL intensity with error
fig = plt.figure()
img = numpy.load("pixel_mask.npy").astype(float)
n_pixels = int(numpy.sum(img))
img[numpy.where(img)] = I_CL[:n_pixels, 0]
im = plt.imshow(img)
plt.title(f"2c: Open-loop Intensity (frame 1 out of {n_frames})")
plt.colorbar()

def animate(i):
    img = numpy.load("pixel_mask.npy").astype(float)
    n_pixels = int(numpy.sum(img))
    img[numpy.where(img)] = I_CL[:n_pixels, i] #first wavelength
    im.set_data(img)
    plt.title(f"2c: Open-loop Intensity (frame {i+1} out of {n_frames})")
    return [im] 

anim = animation.FuncAnimation(fig, animate, frames=n_frames, interval=200)
plt.show()
