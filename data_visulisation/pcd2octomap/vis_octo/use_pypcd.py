#!/usr/bin/python
import numpy as np
import sys 
import pypcd


if len(sys.argv) < 2:
        print("Provide the PCD full path!")
else:
        data_path = sys.argv[1]


pc = pypcd.PointCloud.from_path(data_path)
print(pc.count)
print(pc.fields)
print(pc.pc_data['rgb'][:100])
rgbs_list = pc.pc_data['rgb'][6000:6100]

#  the format are  R G B A
new_list = [[int('{:032b}'.format(data)[:8],2),int('{:032b}'.format(data)[8:16],2),int('{:032b}'.format(data)[16:24],2),int('{:032b}'.format(data)[24:],2)] for data in rgbs_list]
print(new_list)






# pc.pc_data has the data as a structured array
# pc.fields, pc.count, etc have the metadata

# center the x field
#pc.pc_data['x'] -= pc.pc_data['x'].mean()

# save as binary compressed
#pc.save_pcd('bar.pcd', compression='binary_compressed')
