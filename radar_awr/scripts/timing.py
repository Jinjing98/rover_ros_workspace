import numpy as np
import time

class Timer:
    def __init__(self, name=None, measurements=1):
        self.measurements = measurements
        if not measurements:
            self.timings = 1
        else:
            self.timings = np.zeros(measurements)
        if not name:
            self.name = "timer"
        else:
            self.name = name
        self.index = 0
        self.start_time = time.time()
        print("Timer", name, "initiated")

        
    def start(self):
        self.start_time = time.time()


    def stop(self):
        try:
            self.timings[self.index] = time.time() - self.start_time
        except:
            self.timings = np.append(timings, time.time() - self.start_time)
        self.index += 1

    def times(self):
        return self.timings()

    def get_name(self):
        if not self.index == self.measurements:
            return self.name + ": " + str(self.index) + " measurements done but " + str(self.measurements) + " where planned"
        else:
            return self.name + ": " + str(self.measurements) + " measurements done"

    @staticmethod
    def evaluate(timers):
        """
        timers: arrary of Timer objects
        """
        idx = 0
        for timer in timers:
            mean = 0
            max = 0
            min = 1000
            for t in timer.timings:
                if (t>max):
                    max = t
                elif (t<min):
                    min = t
                mean += t
                idx += 1
        
            mean = mean / timer.index
            print("TTTTT Timer", timer.get_name(), "  TTTTT")
            print("max=", max, "min=", min, "mean=", mean)

