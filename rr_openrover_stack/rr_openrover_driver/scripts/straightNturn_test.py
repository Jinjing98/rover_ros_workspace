#!/usr/bin/env python
 
from movement import MovementManager

max_velocity_linear  = 0.5
max_velocity_angular  = 0.628
manager = MovementManager()


manager.set_max_linear_velocity(max_velocity_linear)
manager.set_max_angular_velocity(max_velocity_linear)
 

manager.move_straight(2)  #3 meters away from you, therefore we drive with 0.3m/s for 10s
manager.turn(0.05)      # the angle you will turn...the upper limit 0.1  is restricted by the max linear velocity.  bigger linear velocity, the upper limit is bigger.  also some other restricition by the parasms in move_straight?  or by nature the car can not rotate that much?

print "the car has run for ",manager.accumulated_time," seconds"   #   when check the actul vel , the ang vel is positive some time negative some other time
 
 
