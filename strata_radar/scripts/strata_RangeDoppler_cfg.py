# **************************************************************************
#                        IMPORTANT NOTICE
# **************************************************************************
# THIS SOFTWARE IS PROVIDED "AS IS". NO WARRANTIES, WHETHER EXPRESS,
# IMPLIED OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE APPLY TO THIS
# SOFTWARE. INFINEON SHALL NOT, IN ANY CIRCUMSTANCES, BE LIABLE FOR SPECIAL,
# INCIDENTAL, OR CONSEQUENTIAL DAMAGES, FOR ANY REASON WHATSOEVER.
# **************************************************************************

import strata


def getConfigs():
	##MMIC configuration
	mmicConfig = strata.IfxRfe_MmicConfig()
	mmicConfig.enableMonitoring = False
	mmicConfig.sampleWidth = 14                     # [bits]
	mmicConfig.sampleRate = (50e6 / 8)              # [Hz]
	mmicConfig.txPower = 80.0                       # [%] TX channel output power {0.0: 100.0}
	mmicConfig.lpGain = 14                          # [dB] AFE low-pass filter gain; RXS: {-16:6:56}
	mmicConfig.mixerGain = 0                        # [dB] Mixer gain; RXS: {0;6}
	mmicConfig.dcocEnable = False                   # enable DC offset compensation (DCOC)
	mmicConfig.dcocShift = 0                        # RXS: {0:16} (default 3) N factor for DFE DCOC

	## Sequence configuration
	#time values have to be multiples of 1/(sample rate after decimation)
	#total number of ramps should be a power of 2, otherwise automatic zero padding
	sequenceConfig = strata.IfxRfe_Sequence()
	sequenceConfig.tRampStartDelay      = 0.0e-6       # [s]       delay before starting LVDS transmission
	sequenceConfig.tRampStopDelay       = 0.0e-6       # [s]       delay before stopping LVDS transmission

	samples                             = 256
	tPayload                            = samples / mmicConfig.sampleRate   # [s]   sampled time period
	sequenceConfig.tRamp                = sequenceConfig.tRampStartDelay + tPayload + sequenceConfig.tRampStopDelay    #[s]  total ramp duration including start/stop delay
	sequenceConfig.tJump                = 9.6e-6        # [s]       duration of flyback segment
	sequenceConfig.tWait                = 1.6e-6        # [s]       duration of wait segment

	sequenceConfig.rxMask               = 0b1111        # [1]       RX channels enable bitmask (32 bit): 0b rx31 ... rx3 rx2 rx1 rx0
	sequenceConfig.loops                = 8             # [1]       number of sequence repetitions; (RXS: [1 1023])

	sequenceConfig.ramps = [strata.IfxRfe_Ramp()] * 1
	sequenceConfig.ramps[0].fStart      = 76.1e9        # [Hz]      start frequency of tRamp segment; (RXS: [76.0 77.0])
	sequenceConfig.ramps[0].fDelta      =  0.8e9        # [Hz]      delta frequency of tRamp segment; (RXS: [0.0 1.0e9])

	sequenceConfig.ramps[0].loops       = 1             # [1]       number of ramp repitions; (RXS: [1 1023])
	sequenceConfig.ramps[0].muxOut      = 0             # [1]       signal output bitmask during tRamp segment; (RXS: 0b DMUX_D DMUX_C DMUX_B DMUX_A)
	sequenceConfig.ramps[0].txMask      = 0b001         # [1]       TX channels enable bitmask (16bit): 0b tx15 ... tx2 tx1 tx0
	sequenceConfig.ramps[0].txPhases    = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # [deg[16]] TX phase values array for enabled channels in .txMask (values for non-active channels are ignored); [0:360]
	
	## Processing configuration
	processingConfig = strata.IfxRsp_Stages()
	processingConfig.fftSteps      = 2                                    # [1]       number of FFT stages enabled; [0: time data, 1: Range FFT, 2: Doppler]
	processingConfig.format        = strata.IfxRsp.DataFormat.Complex32   # output data format for time data or FFT; (only Real?? or Complex??, Real16 would be recommended for timedata, but it is only supported on Aurix B-step)
	processingConfig.nciFormat     = strata.IfxRsp.DataFormat.Disabled    # output data format for NCI (only Real32)
	processingConfig.virtualChannels = strata.IfxRsp.VirtualChannel.RawData        # [1]   channels bitmask to enable/disable data output

	#range FFT
	processingConfig.fftSettings[0].size           = 0         # [1]   length of FFT input vector after zero padding (multiple interger of power of 2); 0 - applying no zero padding. 
	processingConfig.fftSettings[0].acceptedBins   = 0         # [1]   0 = disable rejection or depends on the fftFlags
	processingConfig.fftSettings[0].window         = strata.IfxRsp.FftWindow.Hann  # [1]   selecting a window function applying to time data before performing 1st fft; [.FftWindow_NoWindow,.FftWindow_Hann,.FftWindow_Hamming,.FftWindow_BlackmanHarris]
	processingConfig.fftSettings[0].windowFormat   = strata.IfxRsp.DataFormat.Real32
	processingConfig.fftSettings[0].exponent       = 0          # Number of shift left at the result (Gain)
	processingConfig.fftSettings[0].flags          = strata.IfxRsp.FFT_FLAGS.DISCARD_HALF | strata.IfxRsp.FFT_FLAGS.INPLACE # option for discarding (negative) half of the fft output ; [FFT_FLAGS_DISCARD_HALF, FFT_FLAGS_INPLACE].

	# velocity FFT
	processingConfig.fftSettings[1].size           = 0         # [1]   length of FFT input vector after zero padding (multiple interger of power of 2); 0 - applying no zero padding. 
	processingConfig.fftSettings[1].acceptedBins   = 0         # [1]   0 = disable rejection or depends on the fftFlags
	processingConfig.fftSettings[1].window         = strata.IfxRsp.FftWindow.Hann  # [1]   selecting a window function applying to time data before performing 1st fft; [.FftWindow_NoWindow,.FftWindow_Hann,.FftWindow_Hamming,.FftWindow_BlackmanHarris]
	processingConfig.fftSettings[1].windowFormat   = strata.IfxRsp.DataFormat.Real32 
	processingConfig.fftSettings[1].exponent       = 0         # Number of shift left at the result (Gain)
	processingConfig.fftSettings[1].flags          = strata.IfxRsp.FFT_FLAGS.INPLACE # option for discarding (negative) half of the fft output ; [FFT_FLAGS_DISCARD_HALF, FFT_FLAGS_INPLACE].

	#Detections
	processingConfig.detectionSettings.maxDetections = 0

	processingConfig.dbfSetting[0].angles = 0
	processingConfig.dbfSetting[1].angles = 0

	return mmicConfig, sequenceConfig, processingConfig
