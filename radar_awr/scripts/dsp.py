import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.image import AxesImage
from math import ceil
#import mmwave.dsp as dsp
import utils as ut
from timing import Timer
from mmwave.dsp import cfar, compensation


frame_idx = 0

FTL = 3e8 # speed of light
BW = 1.5351e9 #Bandwidth of the radar
FC = 77e9 # carrier frequency

""" These are the settings for now """
CHIRP_ACTIVE = 60e-6
CHIRP_IDLE = 48e-6
CHIRP_TIME = CHIRP_ACTIVE + CHIRP_IDLE
RANGE_RES = FTL / (2 * BW) # resolution of the range for the range-chirp map
LAMBDA_MMWAVE = FTL / FC
FRAME_TIME = 128 * CHIRP_TIME # resoltion of the velocity for the range-doppler map, gets set in plot_map, default to 128 chirps per frame
VELOCITY_RES = (FTL/FC) / (2 * FRAME_TIME)

def movieMaker(fig, ims, title, save_dir):
    import matplotlib.animation as animation

    # Set up formatting for the Range Azimuth heatmap movies
    Writer = animation.FFMpegWriter
    writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)

    plt.title(title)
    print('Done')
    im_ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=3000, blit=True)
    print('Check')
    im_ani.save(save_dir, writer=writer)
    #im_ani.save(save_dir)
    print('Complete')

        
def hadamard2_decode(data):
    # We put TDM also in this decode function beacause we adjust the channel order to group the TX1 data in the first half, TX2 data in the second half.
    #TODO: this can probably be done faster
    decode1 = (data[:,0::2,:] + data[:,1::2,:]) /2
    decode2 = (data[:,0::2,:] - data[:,1::2,:]) /2
    decode = np.concatenate((decode1,decode2), axis=1)
    #decode channel order is RX1TX1,RX2TX1,RX3TX1,RX4TX1,RX1TX2,RX2TX2,RX3TX2,RX4TX2
    print(np.complex == decode.dtype)
    return decode

def hadamard2_decode_openradar(data):
    # shape is (samples, channels, chirps)
    decode1 = (data[:, 0:4, :] + data[:, 4:8, :]) / 2
    decode2 = (data[:, 0:4, :] - data[:, 4:8, :]) / 2
    decode = np.concatenate((decode1, decode2), axis=1)
    #print(decode.shape, decode.dtype)
    return decode

def phase_compensation(frame, doppler_bins, num_tx):
    phase_vector  = (np.arange(doppler_bins) - doppler_bins / 2) * np.pi / (doppler_bins)
    compen_vector = np.exp(1j * phase_vector, dtype=np.complex64)
    #print(compen_vector.dtype)
    normal_half = frame[:, 0:4, :]
    compensated_half = frame[:, 4:8, :] * compen_vector
    #print("dtype after comp", normal_half.dtype, compensated_half.dtype)
    return np.concatenate((normal_half, compensated_half), 1)

def phase_compensation_old(frame, doppler_bins, num_tx):
    phase_vector  = (np.arange(doppler_bins) - doppler_bins / 2) * np.pi / (doppler_bins)
    compen_vector = np.exp(1j * phase_vector)
    normal_half = frame[:, 0::2, :]
    compensated_half = frame[:, 1::2, :] * compen_vector
    return np.concatenate((normal_half, compensated_half), 1)

def separate_tx(signal, num_tx, vx_axis=1, axis=0):
    """Separate interleaved radar data from separate TX along a certain axis to account for TDM radars.

    Args:
        signal (ndarray): Received signal.
        num_tx (int): Number of transmit antennas.
        vx_axis (int): Axis in which to accumulate the separated data.
        axis (int): Axis in which the data is interleaved.

    Returns:
        ndarray: Separated received data in the

    """
    # Reorder the axes

    reordering = np.arange(len(signal.shape))
    reordering[0] = axis
    reordering[axis] = 0
    #print("signal shape", signal.shape)
    #print(reordering)
    signal = signal.transpose(reordering)

    out = np.concatenate([signal[i::num_tx, :, :] for i in range(num_tx)], axis=vx_axis) #128,4,256
    #print("out shape", out.shape) # out shape 64, 8, 256 -> 64, 4even 4 odd, 256
    out = out.transpose(reordering)
    
    return out

def separate_tx_old(signal, num_tx, vx_axis=1, axis=0):
    """
    signal: shape (128, 4, 256)
    out: shape (64, 8, 256) 8 channels even + odd + ...
    """
    reordering = np.arange(len(signal.shape))
    reordering[0] = axis
    reordering[axis] = 0
    #print("signal shape", signal.shape)
    #print(reordering)
    signal = signal.transpose(reordering)
    out = np.ndarray((64, 8, 256), dtype=np.complex64)
    for i in range(4):
        out[:, i * num_tx, :] = signal[0::2, i, :]
        out[:, i * num_tx + 1, :] = signal[1::2, i, :]
    #print("out shape", out.shape) # out shape 64, 8, 256 -> 64, even odd, even, 256
    out = out.transpose(reordering)
    
    return out

def rc_fft(data, window, scaled, range_samples, remove_clutter=False):
    """
    FFT for range map for each channel
    params:
    return: 
    radar_cube shape is (number doppler bins, channels, adc samples) 
    """
    
#    data = np.moveaxis(data, 0, 1) # shape before: (64, 256, ?), shape after: (256, 64, ?)

    #radar_cube = dsp.range_processing(data, ut.Window.HANNING) / scaled * (2.0 / num_samples)
    #print("rc dtype before windowing", data.dtype)
    data = ut.windowing(data, window, axis=-1)
    #print("rc dtype after windowing", data.dtype)
    radar_cube = np.fft.fft(data, axis=-1) / scaled * (2.0 / range_samples)
    if remove_clutter:
        radar_cube = compensation.clutter_removal(radar_cube)
    #print("shape of radar_cube: ", radar_cube.shape)
        
#    print("spectrum after rc_fft:", spectrum.shape)
    return radar_cube

def range_doppler_fft_openradar(radar_cube, window, scaled, num_tx, num_chirps):
    open = True
    if open:
        num_tx = 2
        _, doppler_fft_out = dsp.doppler_processing(radar_cube, num_tx, interleaved=False, window_type_2d=ut.Window.HANNING, accumulate=False)
        # aoa_out.shape is numRangeBins, numVirtualAntennas, numDopplerBins
        doppler_fft_out = doppler_fft_out / scaled * (2.0 / (num_chirps/2))
        doppler_fft_out = np.fft.fftshift(doppler_fft_out, axes=-1)
        return doppler_fft_out
    
def range_doppler_fft(radar_cube, window, scaled, doppler_bins):
    """
    FFT for range-doppler map using all channels
    fft_samples is the number of chirps per channel

    radar_cube shape is (num of range bins, channels, doppler bins)
    """
    #print("rd_fft dtype", radar_cube.dtype)
    doppler_cube = radar_cube.transpose(2, 1, 0)
    #doppler_cube = radar_cube.transpose(1, 0)
    doppler_cube = ut.windowing(doppler_cube, window, -1)
    doppler_cube = np.fft.fft(doppler_cube) / scaled * (2.0 / doppler_bins)
    doppler_cube = np.fft.fftshift(doppler_cube, axes=-1)
    #print("doppler_cube dtype", doppler_cube.dtype)
    return doppler_cube


def range_doppler_fft_hector(radar_cube, window, scaled, nu_tx, num_chirps, num_samples, num_channels):
    """
    FFT for range-doppler map on only the first channel
    return:
        doppler and range spectrum with clutter removed, shapes () resp. ()
    """
    data = np.moveaxis(data, 0, 1) # This could be in the right format out of extraction.py, data shape after moving axis is (256, 64, 8)
    print("shape of spectrum and window:", range_spectrum.shape, window_doppler.shape)
    doppler_cube = range_spectrum * window_doppler # np.multiply does not work here, although same shape. Why? "tuple not callable", temp array must be really slow
#    np.multiply(range_spectrum, window_doppler, doppler_spectrum)
    doppler_cube =np.fft.fft(doppler_cube, num_chirps_effective, 1) / scaled_doppler * (2.0 / (num_chirps_effective))
    doppler_cube = np.fft.fftshift(doppler_cube, axes=1)
#        print("doppler spectrum", doppler_spectrum)
    return doppler_cube


def update_micro_doppler(md_map, rd_map, length, frame_num):
    if frame_num < length:
        md_map[frame_num, :] = np.sum(rd_map, 0) #reduce range bins
        print("#### rd_map and md_map shape:", rd_map.shape, md_map.shape)
        print("### md frame number", frame_num, "of", length)
    else:
        md_map = np.roll(md_map, -1, 0)
        print("roll")
        md_map[-1, :] = np.sum(rd_map, 0)
    return md_map
        
def plot_rc_map(rc_map, fig, suffix=""):
    rg_bin, dp_bin = rc_map.shape
    axis_range = np.arange(rg_bin) * RANGE_RES
    axis_doppler = np.arange(dp_bin) + 1

    fig_colormesh = plt.pcolormesh(axis_doppler, axis_range, rc_map, edgecolors="None")
    fig.colorbar(fig_colormesh)
    fig_title      =  plt.title('Range vs Chirps - Only one Channel - Frame %04d' % frame_idx);
    fig_xlabel     =  plt.xlabel('Chirps');
    fig_ylabel     =  plt.ylabel('Range (meters)');
    name_string    =  "fig/RangeVersus" + "chirps_Frame_" + '%04d' % (frame_idx) + suffix + '.png'
    fig.savefig(name_string, bbox_inches='tight')
    plt.pause(0.05)
    plt.clf()


def plot_rd_map(rd_map, fig, suffix=''):
    """

    """
    rg_bin, dp_bin = rd_map.shape
    axis_range = np.arange(rg_bin) * RANGE_RES
    axis_doppler = (np.arange(dp_bin) - dp_bin / 2) * VELOCITY_RES

    rd_map = np.where(rd_map > 50, 50, rd_map) # set saturation value to highlight target.
    fig_colormesh  =  plt.pcolormesh(axis_doppler, axis_range, rd_map, edgecolors='None')
    fig.colorbar(fig_colormesh)
    fig_title      =  plt.title('Range Doppler Map');
    fig_xlabel     =  plt.xlabel('Velocities (m/s)');
    fig_ylabel     =  plt.ylabel('Range (meters)');
    name_string    =  'fig/RangeDopplerMap_Frame_'+'%04d'%(frame_idx)+suffix+'.png'
    fig.savefig(name_string, bbox_inches='tight')
    plt.pause(0.05)
    plt.clf()

    
def dsprocessing(dsp_pipe, plot_pipe, params):
    frame_idx = 0

    FRAME_TIME = params.get("chirps") * CHIRP_TIME
    VEL_RESOLUTION = LAMBDA_MMWAVE / (2 * FRAME_TIME)

    """
    Parameters:
        dsp_pipe: end of pipe or queue. 1 frame at a time. Data is in shape (128, 4, 64)
        plot_pipe: end of pipe or queue. All plots together.

    declaring and initializing ndarrays here to save time, otherwise they get initialized in every loop
    There is also the option to override array structure recieved from the pipe/queue!
    TODO: maybe calc num_chirps_channel = num_chirps//2, mind that num_tx = 2!!!, include that in online_processing.py
    """

    scaled_range = ut.window_scalar(params.get("samples"), ut.Window.HANNING)
    scaled_doppler = ut.window_scalar(params.get("doppler"), ut.Window.HANNING)

    # this is for preparing the plot
    #micro_doppler = np.zeros((params("micro_doppler"), params("doppler")), dtype=float)
    micro_doppler = np.random.rand(params("micro_doppler"), params("doppler"))
    init_micro_doppler = np.zeros((111,64))
    init_range = np.zeros((64, 128))
    init_doppler = np.zeros((128,  64), dtype=np.float32)
    method = 0

    run = True
    
    #### Timing
    pipe_timer = Timer("pipe", 256)
    rc_fft_timer = Timer("rc fft", 256)
    seperate_timer = Timer("separate", 256)
    rd_fft_timer = Timer("rd fft", 256)
    plot_timer = Timer("plot", 256)
    frame_timer = Timer("frame", 256)
    ####
    
    # Movie
    #ims = np.empty(100, dtype=AxesImage)
    frames = np.zeros((128,128,4,256), dtype=np.complex64)
    while run:
        frame_timer.start()
        pipe_timer.start()
        try:
            frame = dsp_pipe.get(timeout=2)
            if frame_idx < 128:
                #print(frame)
                frames[frame_idx,:,:,:] = frame
        except:
            break
        pipe_timer.stop()

        seperate_timer.start()
        frame = separate_tx(frame, params.get("tx")) # frame_in 128 chirps x 4 ch x 256 samples
        seperate_timer.stop()
        # frame_out - shape 128x 4ev 4od x 256
        
        #print(frame[:, 0, :])
        #radar_cube = dsp.range_processing(frame, ut.Window.HANNING)
        
        rc_fft_timer.start()
        radar_cube = rc_fft(frame, ut.Window.HANNING, scaled_range, params("samples"), remove_clutter=True)
        rc_fft_timer.stop()
        
        rd_fft_timer.start()
        #doppler_map = range_doppler_fft_openradar(radar_cube[:,:,0:128], window_doppler_2d, scaled_doppler, num_transmitters, num_chirps)
        doppler_cube = range_doppler_fft(radar_cube[:,:,0:128], ut.Window.HANNING, scaled_doppler, params.get("doppler"))
        rd_fft_timer.stop()

        doppler_cube = phase_compensation(doppler_cube, params.get("doppler"), params.get("tx"))
        doppler_cube = hadamard2_decode_openradar(doppler_cube)
        doppler_map = np.sum(np.abs(doppler_cube), 1)
        micro_doppler = update_micro_doppler(micro_doppler, doppler_map, params("micro_doppler"), frame_idx)

        #detected_objects = cfar.ca(doppler_map)
        #print(detected_objects)
        
        """
        if frame_idx % 2:
            micro_doppler[frame_idx, :] = 2000
        """
        frame_idx += 1
        if (plot_pipe):
            plots = params("plots")
            if (plots=="rd"):
                plot_pipe.put((doppler_map, frame_idx))
                print("sent md map to plot")
            elif (plots=="md"):
                plot_pipe.put((micro_doppler, frame_idx))
            else:
                plot_pipe.put((range_map, frame_idx))
            
        frame_timer.stop()
        #if frame_idx >= 300:
        #    run = False
        print("dsp frame:",frame_idx)
    timers = [pipe_timer, seperate_timer, rc_fft_timer, rd_fft_timer, plot_timer, frame_timer]
    pipe_timer.evaluate(timers)
    np.savez_compressed("raw.npz", frames)
    print("end of dsp")
    #movieMaker(figure, ims, "Range-Doppler-Map_100_frames", "./Range-Doppler.mp4")
