import numpy as np

def extract_numbers(chirp_queue, num_samples, num_rec, num_bytes_sample):
    numbers_dump = open("numbers_dump.txt", "w")
    complex_chirps = []
    idx = 0
    length = num_samples * num_rec * num_bytes_sample
    complex_chirp = np.empty([num_samples * num_rec], dtype=np.complex64)
    while True: #TODO: change condition
        print("get chirp")
        chirp = chirp_queue.get()
        idx = 0
        print(type(chirp))
#        print(len(chirp))
        while ((idx) < 4096):
            re_1 = int.from_bytes(chirp[idx:idx+2], byteorder="big", signed=True)
            re_2 = int.from_bytes(chirp[idx+2:idx+4], byteorder="big", signed=True)
            im_1 = int.from_bytes(chirp[idx+4:idx+6], byteorder="big", signed=True)
            im_2 = int.from_bytes(chirp[idx+6:idx+8], byteorder="big", signed=True)
            np.append(complex_chirp, re_1 + im_1*1j) #TODO: does this work?
            np.append(complex_chirp, re_2 + im_2*1j)
            numbers_dump.write(str(re_1 + im_1*1j)+"\n")
            numbers_dump.write(str(re_2 + im_2*1j)+"\n")
            idx += 8
    numbers_dump.close()
