# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 09:58:59 2019

@author: prasath, hagonzalezd
"""
import socket
import time
import threading
import struct
#7a12
connect_req1 = b'\x5a\xa5\x09\x00\x00\x00\xaa\xee'
connect_req2 = b'\x5a\xa5\x0e\x00\x00\x00\xaa\xee'
connect_req3 = b'\x5a\xa5\x03\x00\x06\x00\x01\x02\x01\x02\x03\x1e\xaa\xee'
connect_req4 = b'\x5a\xa5\x0b\x00\x06\x00\xc0\x05\x12\x7a\x00\x00\xaa\xee'
dca_req = b'\x5a\xa5\x05\x00\x00\x00\xaa\xee'

resp_after_packets = b'\x5a\xa5\x0a\x00\x00\x01\xaa\xee'

final_req = b'\x5a\xa5\x06\x00\x00\x00\xaa\xee'

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("192.168.33.30",4096))

a=[]
b=[]
i=0
j=0
end_time = 0

#Packets to capture
start_pack = 0
end_pack = 0

## host IP is "192.168.33.30" 
## DCA IP is "192.168.33.180"

#Thread 1 - The loop runs for infinite number of times.#
def thrd_1():
    thread1 = threading.currentThread()
    while getattr(thread1, "do_run", True):
        data_1, addr_1 = sock1.recvfrom(1508) # buffer size is 1024 bytes
        a.append((data_1))

#Thread 2 - The loop runs for infinite number of times.#
def thrd_2():
    thread2 = threading.currentThread()
    while getattr(thread2, "do_run", True):
        data_2, addr_2 = sock1.recvfrom(1508) # buffer size is 1024 bytes
        b.append((data_2))

#Thread 3 - It determines the end time.#
def thrd_3():
    global end_time
    time.sleep(6) # Determines the final interval of recording
    end_time = 1

#Reset Packet count
sock.sendto(final_req,('192.168.33.180',4096))
msg, server = sock.recvfrom(2048)
#print ("msg.hex",msg.hex()) # It seems the correct response is 8203. msg.hex 5aa50e008203aaee
print ("REQ",final_req)
print ("RES",msg)
time.sleep(1)

sock.sendto(connect_req1,('192.168.33.180',4096))
msg, server = sock.recvfrom(2048)
#print ("msg.hex",msg.hex()) # It seems the correct response is 8203. msg.hex 5aa50e008203aaee
print ("REQ",connect_req1)
print ("RES",msg)
time.sleep(1)

sock.sendto(connect_req2,('192.168.33.180',4096))
msg, server = sock.recvfrom(2048)
#print ("msg.hex",msg.hex()) # It seems the correct response is 8203. msg.hex 5aa50e008203aaee
print ("REQ",connect_req2)
print ("RES",msg)
time.sleep(1)

sock.sendto(connect_req3,('192.168.33.180',4096))
msg, server = sock.recvfrom(2048)
#print ("msg.hex",msg.hex()) # It seems the correct response is 8203. msg.hex 5aa50e008203aaee
print ("REQ",connect_req3)
print ("RES",msg)
time.sleep(1)

sock.sendto(connect_req4,('192.168.33.180',4096))
msg, server = sock.recvfrom(2048)
#print ("msg.hex",msg.hex()) # It seems the correct response is 8203. msg.hex 5aa50e008203aaee
print ("REQ",connect_req4)
print ("RES",msg)
time.sleep(1)

sock.sendto(dca_req,('192.168.33.180',4096))
msg, server = sock.recvfrom(2048)
#print ("msg.hex",msg.hex()) # It seems the correct response is 8203. msg.hex 5aa50e008203aaee
print ("REQ",dca_req)
print ("RES",msg)
time.sleep(1)

#ASSUMPTION - The measurement starts here.
print ("Trigger Start Frame")
time.sleep(5) # Decides measurment start time

sock1 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock1.bind(("192.168.33.30",4098))

#Initiating two threads. Assumption - 2 threads should run in two different cores#
try:
    thread1 = threading.Thread(target = thrd_1)
    thread2 = threading.Thread(target = thrd_2)
    thread3 = threading.Thread(target = thrd_3)
    thread1.start()
    thread2.start()
    thread3.start()
except:
    import traceback
    print (traceback.format_exc())
    
time.sleep(1)

"""Sorting the received packets based on the updated lists 'a' and 'b'."""
chirp_size = 128
samples = 256
channels = 4
byte_numbers = 4
per_frame_size = (chirp_size*samples*channels*byte_numbers)

maincount = 0
reached_end = 0
flag_a_done = 0
flag_b_done = 0
netcount = 0
zeroarr = struct.pack('=QQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQH',0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)

srta1 = 0
srtb1 = 0

wbx = 0 #Wait bit for x (data based on list 'a')
wby = 0 #Wait bit for y (data based on list 'b')
netlist = [] #Final sorted list
missed_packets = 0
firsttime = 1

#Main sorting loops
while (maincount<3):
        c = len(a)
        d = len(b)
        subcount = 0
        if(firsttime != 1):
            maincount = maincount+1
        while (subcount<5):
            if(((c == 0) and (d == 0) and (wbx == 0) and (wby == 0)) or ((reached_end == 1) and (flag_a_done == 1) and (flag_b_done == 1))): #No data in the list
                time.sleep(0.1)
                subcount = subcount+1 #Sub counter to check the idle time
            else:
                subcount = 0
                maincount = 0
                firsttime = 0
                if ((c > 0)): #packets available for sorting list 'a'
                    if((wbx == 0)and(flag_a_done == 0)):
                        x = a.pop(0) #remove index 0 (FIFO) recived packet from list 'a'
                        c = c-1 #reduce the measured size for list 'a'
                        srta2 = srta1
                        srta1 = x[0]+(x[1]<<8)+(x[2]<<16)+(x[3]<<24) #identify the packet number
                        if((srta1>=start_pack)&(start_pack != 0)&((srta1 <= end_pack)or(end_time != 2))):
                            srta = srta1 + 1 - start_pack
                        else:
                            srta = 0
                            if(start_pack == 0):
                                wbx = 1
                    
                    if((srta1 > end_pack) and (end_time == 2)):
                        flag_a_done = 1
                    
                    if(srta == netcount+1): #the particular packet is received in proper order
                        netlist.append(x) #append the packet to the final sorted list
                        netcount = netcount+1
                        wbx = 0
                    elif((srta <= netcount)&(srta != 0)): #the particular packet is received late and the index is already filled with zeros in the final sorted list
                        netlist[srta-1] = x #add packet to the proper index
                        wbx = 0
                        missed_packets -= 1
                    elif(srta != 0): #expected packet is not recieved in this list or recieved in different order
                        netlist.append(zeroarr) #Pad zero in the final list for the next index
                        netcount = netcount+1
                        missed_packets += 1
                        wbx = 1 #set Wait bit for list 'a', so that during the consecutive iteration data available in 'x' is maintained until it is stored in the sorted location
                elif(wbx == 1): #consecutive iteration where the wait bit is active for list 'a'
                    if(srta == netcount+1): #the particular packet is in correct order to get stored
                        netlist.append(x) #append the packet to the final sorted list
                        netcount = netcount+1
                        wbx = 0 #Reset wait bit 'x' to zero
                    elif((srta <= netcount)&(srta != 0)): #the particular packet is received late or index is already filled with zeros in the final sorted list
                        netlist[srta-1] = x #add packet to the proper index
                        wbx = 0 #Reset wait bit 'x' to zero
                        missed_packets -= 1
                    elif(srta != 0): #expected packet is not recieved in this list or recieved in different order
                        netlist.append(zeroarr) #Pad zero in the final list for the next index
                        missed_packets += 1
                        netcount = netcount+1
                
                if ((d > 0)): #packets available for sorting list 'b'
                    if((wby == 0)and(flag_b_done == 0)):
                        y = b.pop(0) #remove index 0 (FIFO) recived packet from list 'b'
                        d = d-1
                        srtb2 = srtb1
                        srtb1 = y[0]+(y[1]<<8)+(y[2]<<16)+(y[3]<<24) #identify the packet number
                        if((srtb1>=start_pack)&(start_pack != 0)&((srtb1 <= end_pack)or(end_time != 2))):
                            srtb = srtb1 + 1 - start_pack
                        else:
                            srtb = 0
                            if(start_pack == 0):
                                wby = 1
                    
                    if((srtb1 > end_pack) and (end_time == 2)):
                        flag_b_done = 1
                    
                    if(srtb == netcount+1): #the particular packet is received in proper order
                        netlist.append(y) #append the packet to the final sorted list
                        netcount = netcount+1
                        wby = 0
                    elif((srtb <= netcount)&(srtb != 0)): #the particular packet is received late and the index is already filled with zeros in the final sorted list
                        netlist[srtb-1] = y #add packet to the proper index
                        wby = 0
                        missed_packets -= 1
                    elif(srtb != 0): #expected packet is not recieved in this list or recieved in different order
                        netlist.append(zeroarr) #Pad zero in the final list for the next index
                        netcount = netcount+1
                        missed_packets += 1
                        wby = 1 #set Wait bit for list 'b', so that during the consecutive iteration data available in 'y' is maintained until it is stored in the sorted location
                elif(wby == 1): #consecutive iteration where the wait bit is active for list 'b'
                    if(srtb == netcount+1): #the particular packet is in correct order to get stored
                        netlist.append(y) #append the packet to the final sorted list
                        netcount = netcount+1
                        wby = 0 #Reset wait bit 'y' to zero
                    elif((srtb <= netcount)&(srtb != 0)): #the particular packet is received late or index is already filled with zeros in the final sorted list
                        netlist[srtb-1] = y #add packet to the proper index
                        wby = 0 #Reset wait bit 'y' to zero
                        missed_packets -= 1
                    elif(srtb != 0): #expected packet is not recieved in this list or recieved in different order
                        netlist.append(zeroarr) #Pad zero in the final list for the next index
                        missed_packets += 1
                        netcount = netcount+1
                if(start_pack == 0):
                    if((srta1 != 0)&(srtb1 != 0)):
                        start_pack = min(srta1, srtb1)
                        #srta = srta1 + 1 - start_pack
                        #srtb = srtb1 + 1 - start_pack
                    elif(srta1 != 0):
                        start_pack = srta1
                        #srta = srta1 + 1 - start_pack
                    else:
                        start_pack = srtb1
                        #srtb = srtb1 + 1 - start_pack
                    if(start_pack != 0):
                        #print ("start_pack before",start_pack)
                        so_far_bytes = ((start_pack - 1) * 1456)
                        so_far_frame = (so_far_bytes / per_frame_size)
                        #print ("so_far_frame",so_far_frame)
                        current_frame_so_far_bytes = (so_far_bytes % per_frame_size)
                        bytes_yet_to_start_new_frame = per_frame_size - current_frame_so_far_bytes
                        #print ("bytes_yet_to_start_new_frame",bytes_yet_to_start_new_frame)
                        required_no_of_packets = bytes_yet_to_start_new_frame // 1456
                        #print ("required_no_of_packets",required_no_of_packets)
                        start_pack = start_pack + required_no_of_packets
                        start_position_start_pack = (bytes_yet_to_start_new_frame % 1456) + 10
                        if(start_position_start_pack == 10):
                            start_pack = start_pack - 1
                        if((srta1 != 0)&(srta1 >= start_pack)):
                            srta = srta1 + 1 - start_pack
                        else:
                            wbx = 0
                        if((srtb1 != 0)&(srtb1 >= start_pack)):
                            srtb = srtb1 + 1 - start_pack
                        else:
                            wby = 0
                        #print ("start_pack",start_pack)
                        bytes_calculation = start_pack * 1456
                        frame_on_start_pack = (bytes_calculation // per_frame_size)+1
                        #print ("Start Frame:",frame_on_start_pack)
                        frames_until_end_pack = frame_on_start_pack
                        bytes_for_end_pack = frames_until_end_pack * per_frame_size
                        if((bytes_for_end_pack % 1456) != 0):
                            additional_packet = 1
                        else:
                            additional_packet = 0
                        end_pack = (bytes_for_end_pack // 1456) + additional_packet
                    
                if(end_time == 0):
                    if((not(srta1 < end_pack))or(not(srtb1 < end_pack))):
                        frames_until_end_pack = frames_until_end_pack + 1
                        bytes_for_end_pack = frames_until_end_pack * per_frame_size
                        if((bytes_for_end_pack % 1456) != 0):
                            additional_packet = 1
                        else:
                            additional_packet = 0
                        end_pack = (bytes_for_end_pack // 1456) + additional_packet
                
                if(end_time == 1):
                    end_time = 2
                    end_pack_waste_data = (end_pack*1456) % per_frame_size
                    end_position_end_pack = 1466 - end_pack_waste_data
                    reached_end = 1
                    thread1.do_run = False
                    thread2.do_run = False
                    thread1.join()
                    #print ("Thread1 done")
                    thread2.join()
                    #print ("Thread2 done")
                    thread3.join()
                    #print ("Thread3 done")

while(len(netlist)<(end_pack+1-start_pack)):
    netlist.append(zeroarr) #Pad zero in the final list for the next index
    netcount = netcount+1
    missed_packets += 1

total_frames = frames_until_end_pack-(frame_on_start_pack-1)

print("*********************************")

print("Frame Size                :", per_frame_size)

print("Recorded First Packet     :", start_pack)

print("Requested Last Packet     :", end_pack)

print("Recorded Last Packet      :", max(srta2, srtb2))

print("Started Frame             :", frame_on_start_pack)

print("Ended Frame               :", frames_until_end_pack)

print("Number of captured frames :", total_frames)

print("Number of missed packets  :", missed_packets)

print("Size of the sorted list   :", len(netlist))

print("*********************************")

file1 = open('adc_raw_0.bin','wb')
packet_count = 0
file1.write(struct.pack('=I', total_frames))
file1.write(netlist[packet_count][:4])
file1.write(struct.pack('=I', (1466-start_position_start_pack)))
file1.write(netlist[packet_count][4:10])
file1.write(netlist[packet_count][start_position_start_pack:])
regular_data_size = struct.pack('=I', 1456)
packet_count += 1
total_length = len(netlist) - 1
while(packet_count < total_length):
    file1.write(netlist[packet_count][:4])
    file1.write(regular_data_size)
    file1.write(netlist[packet_count][4:])
    packet_count += 1

file1.write(netlist[packet_count][:4])
file1.write(struct.pack('=I', (end_position_end_pack-10)))
file1.write(netlist[packet_count][4:end_position_end_pack])
file1.flush()
file1.close()

print("adc_raw_bin --> Done!!!")

print("ALL DONE")
