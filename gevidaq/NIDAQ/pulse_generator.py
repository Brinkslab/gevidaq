import time

import nidaqmx
import numpy as np


class PulseGenerator:
    def __init__(
        self,
        input_channel,
        output_channel,
        threshold_voltage=0.005,
        pulse_voltage=1.2,
    ):
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.threshold_voltage = threshold_voltage
        self.pulse_voltage = pulse_voltage

        self.task_in = nidaqmx.Task()
        self.task_out = nidaqmx.Task()

        # Configure analog input channel
        self.task_in.ai_channels.add_ai_voltage_chan(self.input_channel)

        # Configure analog output channel with 1.2 volt pulse
        self.task_out.ao_channels.add_ao_voltage_chan(self.output_channel)

    def detect_rising_edge(self, duration_sec):
        start_time = time.time()
        while (time.time() - start_time) < duration_sec:
            previous_value = False
            current_voltage = self.task_in.read()
            if (
                current_voltage > self.threshold_voltage
                and previous_value <= self.threshold_voltage
            ):
                self.generate_analog_pulse()
            previous_value = current_voltage > self.threshold_voltage

    def generate_analog_pulse(self):
        pulse_width = 2e-6  # 2 micros pulse width
        sample_rate = 1e6  # 1 MHz sample rate
        num_samples = int(
            pulse_width * sample_rate
        )  # Number of samples for 1 micros pulse at 1 MHz

        pulse_data = np.full(num_samples, self.pulse_voltage)

        self.task_out.timing.cfg_samp_clk_timing(
            rate=sample_rate, samps_per_chan=num_samples
        )
        self.task_out.write(pulse_data, auto_start=True)

    def close(self):
        self.task_in.close()
        self.task_out.close()
