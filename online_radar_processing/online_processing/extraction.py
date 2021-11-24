import numpy as np
import time
import timing

num_frames = 256
num_chirps = 128

def extract_numbers(frame_pipe, dsp_pipe, num_samples, num_rec, num_bytes_sample):
    byte_idx = 0
    frame_num = 0
    chirp_num = 0

#    LVDS = np.empty([num_rec*num_chirps*num_samples*num_frames], dtype=np.complex64)
#    adc_data_old = np.empty([num_rec, num_chirps*num_frames*num_samples], dtype=np.complex64)
#    lvds_idx = 0

    #1 frame, half the number of chirps because of the virtual recievers
    adc_data = np.empty((int(num_chirps/2), num_samples, 2 * num_rec), dtype=np.complex64)
    sample_idx = 0
    channel_idx = [0,2,4,6] #index of the virtual channels in the adc_data array, hard coded, not good
    channel_offset = 0 #offset that tells 1 or 0
    
    dump = open("raw.txt", "wb")
    """ Timing """
    times_chirp = np.empty((num_frames * num_chirps), dtype=float)
    times_frame = np.empty((num_frames), dtype=float)
    complex_times = np.empty((num_frames * num_chirps * num_rec * int(num_samples/2)), dtype=float)
    add_times = np.empty((num_frames * num_chirps * num_rec * int(num_samples/2)), dtype=float)
    queue_times = np.empty((num_frames * num_chirps), dtype=float)
    tci = 0
    tfi = 0
    tai = 0
    tqi = 0
#    chirp = bytearray()
    #this is for one chirp
    while True:

        time_frame = time.time()

        frame = frame_pipe.get() #shape (128, 4, 256)
        
        chirp_num = 0 #reset the chirp indexes after every frame
        chirp_idx_effective = 0
        
        while (chirp_num < num_chirps):

            time_chirp = time.time()

            rec_idx = 0 #reciever index in the current chirp
            byte_idx = 0
            
            while (rec_idx < num_rec):
                channel_num = channel_idx[rec_idx] + channel_offset
                time_add = time.time()

                adc_data[chirp_idx_effective, :, channel_num] = frame[chirp_num, rec_idx, :]

                add_times[tai] = time.time() - time_add
                tai += 1

                byte_idx += 8

                rec_idx += 1 #after 1024 bytes the data of the next reciever begins?

            """is the next chirp odd or even? set the offset from the normal channel index (channel_idx) in the adc_data array"""
            chirp_num += 1
            
            if (chirp_num % 2):         #odd chirp
                channel_offset = 1 #IS THIS WRONG?
            else:                       #even chirp
                channel_offset = 0
                chirp_idx_effective += 1 #only after 2 chirps (even+odd) the index should increase
            times_chirp[tci] = time.time() - time_chirp
            tci += 1
            
        dsp_pipe.put(adc_data) # shape is (64, 256, 8)
            
        frame_num += 1
        times_frame[tfi] = time.time() - time_frame
        tfi += 1
        print("frame done: ", frame_num)

    print("all frames done: ", frame_num)
#    print("timings: ", times)

#    adc_old(adc_data_old, LVDS, num_samples, num_rec)
    np.savez_compressed("raw.npz", adc_data)
#    print("shape of adc_data: ", adc_data.shape) #shape: 1,64,256,8
#    print(adc_data)

    time_list = [times_frame, times_chirp, complex_times, add_times, queue_times]
    timing.timing(time_list)
