# Copyright 2019 The OpenRadar Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import codecs
import socket
import struct
import time
from enum import Enum

import numpy as np


class CMD(Enum):
    """
    RESET_FPGA_CMD_CODE = '0100'
    RESET_AR_DEV_CMD_CODE = '0200'
    CONFIG_FPGA_GEN_CMD_CODE = '0300'
    CONFIG_EEPROM_CMD_CODE = '0400'
    RECORD_START_CMD_CODE = '0500'
    RECORD_STOP_CMD_CODE = '0600'
    PLAYBACK_START_CMD_CODE = '0700'
    PLAYBACK_STOP_CMD_CODE = '0800'
    SYSTEM_CONNECT_CMD_CODE = '0900'
    SYSTEM_ERROR_CMD_CODE = '0a00'
    CONFIG_PACKET_DATA_CMD_CODE = '0b00'
    CONFIG_DATA_MODE_AR_DEV_CMD_CODE = '0c00'
    INIT_FPGA_PLAYBACK_CMD_CODE = '0d00'
    READ_FPGA_VERSION_CMD_CODE = '0e00'
    """

    RECORD_START_CMD = b'\x5a\xa5\x05\x00\x00\x00\xaa\xee' # 0500
    RECORD_STOP_CMD = b'\x5a\xa5\x06\x00\x00\x00\xaa\xee' # 0600
    SYSTEM_CONNECT_CMD = b'\x5a\xa5\x09\x00\x00\x00\xaa\xee' # 0900
    READ_FPGA_VERSION_CMD = b'\x5a\xa5\x0e\x00\x00\x00\xaa\xee' # 0e00
    CONFIG_FPGA_GEN_CMD = b'\x5a\xa5\x03\x00\x06\x00\x01\x02\x01\x02\x03\x1e\xaa\xee'
    DELAY_CMD = b'\x5a\xa5\x0b\x00\x06\x00\xc0\x05\x12\x7a\x00\x00\xaa\xee' #includes the delay between packets, now 3d09=125us, 7a12=250us, F424=500us, 0C35=25us

    SUCCESS_RSP = b'\x5a\xa5\x0a\x00\x00\x01\xaa\xee'

    def __str__(self):
        return str(self.value)

    def __bytes__(self):
        return self.value


# MESSAGE = codecs.decode(b'5aa509000000aaee', 'hex')
CONFIG_HEADER = '5aa5'
CONFIG_STATUS = '0000'
CONFIG_FOOTER = 'aaee'
ADC_PARAMS = {'chirps': 128,  # 32
              'rx': 4,
              'tx': 1,
              'samples': 256,
              'IQ': 2,
              'bytes': 2}
# STATIC
MAX_PACKET_SIZE = 4096
BYTES_IN_PACKET = 1456
# DYNAMIC
BYTES_IN_FRAME = (ADC_PARAMS['chirps'] * ADC_PARAMS['rx'] *
                  ADC_PARAMS['IQ'] * ADC_PARAMS['samples'] * ADC_PARAMS['bytes'])
BYTES_IN_FRAME_CLIPPED = (BYTES_IN_FRAME // BYTES_IN_PACKET) * BYTES_IN_PACKET
PACKETS_IN_FRAME = BYTES_IN_FRAME / BYTES_IN_PACKET
PACKETS_IN_FRAME_CLIPPED = BYTES_IN_FRAME // BYTES_IN_PACKET
INT16_IN_PACKET = BYTES_IN_PACKET // 2
INT16_IN_FRAME = BYTES_IN_FRAME // 2


class DCA1000:
    """Software interface to the DCA1000 EVM board via ethernet.

    Attributes:
        static_ip (str): IP to receive data from the FPGA
        adc_ip (str): IP to send configuration commands to the FPGA
        data_port (int): Port that the FPGA is using to send data
        config_port (int): Port that the FPGA is using to read configuration commands from


    General steps are as follows:
        1. Power cycle DCA1000 and XWR1xxx sensor
        2. Open mmWaveStudio and setup normally until tab SensorConfig or use lua script
        3. Make sure to connect mmWaveStudio to the board via ethernet
        4. Start streaming data
        5. Read in frames using class

    Examples:
        >>> dca = DCA1000()
        >>> adc_data = dca.read(timeout=.1)
        >>> frame = dca.organize(adc_data, 128, 4, 256)

    """
    
    def __init__(self, static_ip='192.168.33.30', adc_ip='192.168.33.180',
                 data_port=4098, config_port=4096):
        # Save network data
        # self.static_ip = static_ip
        # self.adc_ip = adc_ip
        # self.data_port = data_port
        # self.config_port = config_port

        # Create configuration and data destinations
        self.cfg_dest = (adc_ip, config_port)
        self.cfg_recv = (static_ip, config_port)
        self.data_recv = (static_ip, data_port)

        # Create sockets
        self.config_socket = socket.socket(socket.AF_INET,
                                           socket.SOCK_DGRAM,
                                           socket.IPPROTO_UDP)
        self.data_socket = socket.socket(socket.AF_INET,
                                         socket.SOCK_DGRAM,
                                         socket.IPPROTO_UDP)

        # Bind data socket to fpga
        self.data_socket.bind(self.data_recv)

        # Bind config socket to fpga
        self.config_socket.bind(self.cfg_recv)

        self.frame_buff = []
        self.start_buff = np.zeros(INT16_IN_PACKET, dtype=np.int16)
        self.overshoot = 0
        self.prev_packet_num = 0
        self.total_packets_lost = 0
        self.total_packets = 0
        self.curr_frame_end_byte = None
        self.curr_frame_packet_idx = 0

        self.packet_length = 0

        
    def configure(self):
        """Initializes and connects to the FPGA

        Returns:
            None

        """
        

        #sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        #sock.bind(("192.168.33.30",4096))

        ## host IP is "192.168.33.30" 
        ## DCA IP is "192.168.33.180"


        cmd_list = [CMD.RECORD_STOP_CMD, CMD.SYSTEM_CONNECT_CMD, CMD.READ_FPGA_VERSION_CMD, CMD.CONFIG_FPGA_GEN_CMD, CMD.DELAY_CMD, CMD.RECORD_START_CMD]
        
        for cmd in cmd_list:
            self.config_socket.sendto(cmd.__bytes__(), ('192.168.33.180',4096))
            msg, server = self.config_socket.recvfrom(2048)
            print("REQ", cmd)
            print("RES",msg)
            time.sleep(0.5)
            
        #ASSUMPTION - The measurement starts here.
        print ("Trigger Start Frame")

        
    def close(self):
        """Closes the sockets that are used for receiving and sending data

        Returns:
            None

        """
        self.data_socket.close()
        self.config_socket.close()

    def debug(self):
        print("end of current frame byte index:", self.curr_frame_end_byte)
        print("overshoot from previous frame:", self.overshoot)
        print("arrived packets:", self.total_packets)
        print("packets lost total:", self.total_packets_lost)

    def read(self, timeout=1):
        """ Read in a whole frame via UDP

        Args:
            timeout (float): Time to wait for packet before moving on

        Returns:
            Full frame as array if successful, else None

        """
        # Configure
        self.data_socket.settimeout(timeout)

        # Frame buffer
        # is 1 packet longer to catch packets containing parts of next frame
        self.frame_buff = np.zeros(INT16_IN_FRAME + INT16_IN_PACKET, dtype=np.int16)
        self.frame_buff[: self.overshoot // 2] = self.start_buff[: self.overshoot // 2] # this seems to still be in bytes
        # safes the next frames first data if packet creates overshoot
        self.start_buff = np.zeros(INT16_IN_PACKET, dtype=np.int16) 
        
        
        #packet info
        #Seq: 1 Bytes: 0 Lost packets: 0
        #1 0
        #Seq: 2 Bytes: 1456 Lost packets: 0
        #2 0
        #Seq: 3 Bytes: 2912 Lost packets: 0
        #3 0
        
        # Wait for start of first frame
        #determine the packet and the byte/number offset into the packet where the next frame will start+
        while True:
            packet_num, byte_count, data = self._read_data_packet()
            byte_count_curr = byte_count + BYTES_IN_PACKET
            self.total_packets += 1
            self.total_packets_lost += packet_num - (self.prev_packet_num + 1) # maybe also current packet loss
            #print("Packets lost total:", self.total_packets_lost)
            self.prev_packet_num = packet_num # muss auch beim n??chsten call da sein
            #print("packet number:", packet_num)
            # packet position in current frame funktioniert jetzt nur mit >>int16<<
            # independet of frame packet loss
            curr_frame_packet_idx = (byte_count % BYTES_IN_FRAME) // 2
            self.packet_length = len(data)
            #TODO: watch out for lost packets and offset calculation, especially at the end of a frame, packet loss == bad 

            # catch the first packet received in the whole transmission process
            # notice, that this is still not really safe, if the frame gets chatched in the middle
            # self.total_packets == 0
            if not (self.total_packets - 1):
                self.curr_frame_end_byte = BYTES_IN_FRAME * (byte_count // BYTES_IN_FRAME + 1)  # watch out for frames catched in the middle, this works for bytes but can be done for int16 as well
                self.frame_buff[curr_frame_packet_idx : curr_frame_packet_idx + INT16_IN_PACKET] = data # unsafe part, problematic when added packet contains parts of two frames
                print("first packet arrived was: ", packet_num)

            elif (byte_count + BYTES_IN_PACKET >= self.curr_frame_end_byte):
                #frame has ended at this packet's end or inside packet
                #return completed frame, safe beginning of next frame if ended inside packet
                #curr_frame_end_byte =  wie oben, eigentlich
                print("#### frame captured ####")
                self.debug()
                print("last packet number of frame:", packet_num, ", byte count:", byte_count, ", byte count after this packet:", byte_count + BYTES_IN_PACKET)
                print("current frame packet index:", curr_frame_packet_idx, ", in bytes:", curr_frame_packet_idx * 2)
                # when the capturing begins somewhere in the transmission some ethernet controllers will buffer the first transmitted packets but drop everything afterwards
                # e.g. the Jetson TX2: buffers 1..78 and drops everything afterwards till packages are removed from the buffer
                # this is only a problem if whole frames were skipped due to packet loss
                # --> determine if a frame was skipped
                last_packet_in_frame_missed = self.curr_frame_end_byte <= byte_count
                if last_packet_in_frame_missed:
                    self.curr_frame_end_byte = BYTES_IN_FRAME * (byte_count // BYTES_IN_FRAME + 1)
                    # use the start buffer to safe the current packet and insert it into the next frame on the beginning of the next call
                    # OR drop the whole packet --> faster
                    # TODO: NOT SAFE, corner case: the last packet of the transmission gets buffered and there is a length missmatch
                    # consider self.start_buff[: len(data)]
                    self.start_buff[: INT16_IN_PACKET] = data
                    # safe the packet index of the next frame to insert the data on the next call or drop the whole packet
                    self.curr_frame_packet_idx = curr_frame_packet_idx
                    self.overshoot = 0
                    return self.frame_buff[0 : INT16_IN_FRAME]
                try:
                    # raises exception if data is too small to fit in the normal packet size
                    self.frame_buff[curr_frame_packet_idx : curr_frame_packet_idx + INT16_IN_PACKET] = data
                except:
                    # catches the possible length mismatch of the last packet of the whole transmission
                    print("FFFFFFFFFFFFFFFFFFFF Last scheduled packet may be reached or error occured, pipe last frame FFFFFFFFFFFFFFFFFFFF")
                    print("Check transmitted frames:", (byte_count + BYTES_IN_PACKET) // BYTES_IN_FRAME)
                    print("Check number of last packet:", packet_num)
                    # append the last packet and return, should work
                    self.frame_buff[curr_frame_packet_idx : curr_frame_packet_idx + len(data)] = data
                    print("length last packet", len(data))
                    return self.frame_buff[0 : INT16_IN_FRAME]
                
                self.overshoot = ((byte_count + BYTES_IN_PACKET) % BYTES_IN_FRAME)
                # bytes/ints belonging to next frame, still in bytes
                # ((INT16_IN_FRAME - curr_frame_packet_idx) <= INT16_IN_PACKET) == True
                self.start_buff[0 : self.overshoot // 2] = data[INT16_IN_PACKET - self.overshoot // 2 :] # cannot do data[-self.overshoot // 2 :] because of case self.overshoot == 0
                #TODO: hier unbedingt noch beschreiben, wann der n??chste frame aufh??rt!!!! curr_frame_end_byte neu setzen
                print("#### frame captured ####")
                # simple calculation if next frame's end byte index, as the last packet in the frame was catched, this can be dependent on the last value
                self.curr_frame_end_byte = self.curr_frame_end_byte + BYTES_IN_FRAME
                return self.frame_buff[0 : INT16_IN_FRAME]
            
            else:
                # kommt hier zum ??berschreiben, wenn frameende erreicht -->nein, da frameende im oberen Fall abgeckt, auch verlorene pakete st??ren nicht
                self.frame_buff[curr_frame_packet_idx : curr_frame_packet_idx + INT16_IN_PACKET] = data
        
    def _send_command(self, cmd, length='0000', body='', timeout=1):
        """Helper function to send a single commmand to the FPGA

        Args:
            cmd (CMD): Command code to send to the FPGA
            length (str): Length of the body of the command (if any)
            body (str): Body information of the command
            timeout (int): Time in seconds to wait for socket data until timeout

        Returns:
            str: Response message

        """
        # Create timeout exception
        self.config_socket.settimeout(timeout)

        # Create and send message
        resp = ''
        msg = codecs.decode(''.join((CONFIG_HEADER, str(cmd), length, body, CONFIG_FOOTER)), 'hex')
        try:
            self.config_socket.sendto(msg, self.cfg_dest)
            resp, addr = self.config_socket.recvfrom(MAX_PACKET_SIZE)
        except socket.timeout as e:
            print(e)
        return resp

    def _read_data_packet(self):
        """Helper function to read in a single ADC packet via UDP

        Returns:
            int: Current packet number, byte count of data that has already been read, raw ADC data in current packet

        """
        data, addr = self.data_socket.recvfrom(MAX_PACKET_SIZE)
        #print("packet")
        packet_num = struct.unpack('<1l', data[:4])[0]
        byte_count = struct.unpack('>Q', b'\x00\x00' + data[4:10][::-1])[0]
        packet_data = np.frombuffer(data[10:], dtype=np.int16)
        return packet_num, byte_count, packet_data

    def _listen_for_error(self):
        """Helper function to try and read in for an error message from the FPGA

        Returns:
            None

        """
        self.config_socket.settimeout(None)
        msg = self.config_socket.recvfrom(MAX_PACKET_SIZE)
        if msg == b'5aa50a000300aaee':
            print('stopped:', msg)

    def _stop_stream(self):
        """Helper function to send the stop command to the FPGA

        Returns:
            str: Response Message

        """
        return self._send_command(CMD.RECORD_STOP_CMD)

    
    @staticmethod
    def organize(raw_frame, num_chirps, num_rx, num_samples):
        """Reorganizes raw ADC data into a full frame

        Args:
            raw_frame (ndarray): Data to format
            num_chirps: Number of chirps included in the frame
            num_rx: Number of receivers used in the frame
            num_samples: Number of ADC samples included in each chirp

        Returns:
            ndarray: Reformatted frame of raw data of shape (num_chirps, num_rx, num_samples)

        """
        ret = np.zeros(len(raw_frame) // 2, dtype=np.complex64)

        # Separate IQ data
        ret[0::2] = raw_frame[0::4] + 1j * raw_frame[2::4]
        ret[1::2] = raw_frame[1::4] + 1j * raw_frame[3::4]
        return ret.reshape((num_chirps, num_rx, num_samples))

    @staticmethod
    def organize_ros(raw_frame, num_chirps, num_rx, num_samples):
        """Reorganizes raw ADC data into a full frame

        Args:
            raw_frame (ndarray): Data to format
            num_chirps: Number of chirps included in the frame
            num_rx: Number of receivers used in the frame
            num_samples: Number of ADC samples included in each chirp

        Returns:
            ndarray: Reformatted frame of raw data of shape (num_chirps, num_rx, num_samples)

        """
        ret = np.zeros(len(raw_frame) // 2, dtype=complex)
        real = np.zeros(len(raw_frame) // 2, dtype=int16)
        im = np.zeros(len(raw_frame) // 2, dtype=int16)

        # Separate IQ data
        
        real[0::2] = raw_frame[0::4]
        real[1::2] = raw_frame[1::4]
        im[0::2] = raw_frame[2::4]
        im[1::2] = raw_frame[3::4]
        return real, im, (num_chirps, num_rx, num_samples)
