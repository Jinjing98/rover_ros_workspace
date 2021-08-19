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
	mmicConfig.enableMonitoring         = False;        # [bool]    enabling/ disabling monitoring (also includes low-power mode between sequences)

	mmicConfig.sampleWidth              = 14;           # [bits]    ADC sample width; (RXS: [12, 14])
	mmicConfig.sampleRate               = 50.0e6/8;     # [Hz]      ADC sample rate after decimation; (RXS: 50MSps/[2, 3, 4, 5, 6, 8, 10, 12, 20])
	mmicConfig.txPower                  = 80.0;         # [%]       TX channel output power; [0.0 100.0]

	mmicConfig.lpGain                   = 14;           # [dB]      AFE setting (low-pass filter gain); (RXS: [-16:6:44])
	mmicConfig.mixerGain                = 0;            # [dB]      Mixer gain setting; (RXS: [0, 6])
	mmicConfig.dcocEnable               = False;        # [bool]    enabling/disabling DC offset compensation (DCOC) (RXS)
	mmicConfig.dcocShift                = 0;            # [1]       setting the N factor used as the coefficient in the DFE DCOC; (RXS: [0:1:16])


	##Sequence configuration
	# time values have to be multiples of 1/(sample rate after decimation)
	# total number of ramps should be a power of 2, otherwise automatic zero padding
	sequenceConfig = strata.IfxRfe_Sequence()
	sequenceConfig.tRampStartDelay      = 0.0e-6        # [s]       delay before starting LVDS transmission
	sequenceConfig.tRampStopDelay       = 0.0e-6        # [s]       delay before stopping LVDS transmission

	samples                             = 256 
	tPayload                            = samples / mmicConfig.sampleRate   # [s]   sampled time period
	sequenceConfig.tRamp                = sequenceConfig.tRampStartDelay + tPayload + sequenceConfig.tRampStopDelay    # [s]  total ramp duration including start/stop delay
	sequenceConfig.tJump                = 9.6e-6        # [s]       duration of flyback segment
	sequenceConfig.tWait                = 1.6e-6        # [s]       duration of wait segment

	sequenceConfig.rxMask               = 0b1111         # [1]       RX channels enable bitmask (32 bit): 0b rx31 ... rx3 rx2 rx1 rx0
	sequenceConfig.loops                = 8             # [1]       number of sequence repitions; (RXS: [1 1023])
	
	sequenceConfig.ramps = [strata.IfxRfe_Ramp()] * 1
	sequenceConfig.ramps[0].fStart      = 76.1e9        # [Hz]      start frequency of tRamp segment; (RXS: [76.0 77.0])
	sequenceConfig.ramps[0].fDelta      =  0.8e9        # [Hz]      delta frequency of tRamp segment; (RXS: [0.0 1.0e9])
    
	sequenceConfig.ramps[0].loops       = 1             # [1]       number of ramp repitions; (RXS: [1 1023])
	sequenceConfig.ramps[0].muxOut      = 0             # [1]       signal output bitmask during tRamp segment; (RXS: 0b DMUX_D DMUX_C DMUX_B DMUX_A)
	sequenceConfig.ramps[0].txMask      = 0b001         # [1]       TX channels enable bitmask (16bit): 0b tx15 ... tx2 tx1 tx0
	sequenceConfig.ramps[0].txPhases    = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # .txPhases[16] phase values for TX channels

	#Processing configuration
	processingConfig = strata.IfxRsp_Stages()
	processingConfig.fftSteps      = 0                  # [1]       number of FFT stages enabled; [0: time data, 1: Range FFT, 2: Doppler]
	processingConfig.format        = strata.IfxRsp.DataFormat.Complex16     # output data format for time data or FFT; (only Real?? or Complex??, Real16 would be recommended for timedata, but it is only supported on Aurix B-step)
	processingConfig.virtualChannels = strata.IfxRsp.VirtualChannel.RawData          # [1]   channels bitmask to enable/disable data output


	return mmicConfig, sequenceConfig, processingConfig
