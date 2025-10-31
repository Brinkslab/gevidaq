import ctypes
import logging
import sys
import warnings
from time import sleep


class Mock:
    """mock class, replaces object with dummy that returns self

    logs accesses at debug level
    """

    def __init__(self, name):
        self._name = name
        self._logger = logging.getLogger(name)
        self._logger.debug(f"init Mock({name})")

    def __getattr__(self, attr):
        self._logger.debug(f"attr .{attr}")
        return self

    def __call__(self, *args, **kwargs):
        self._logger.debug(f"call ({args}, {kwargs})")
        return self

    def __gt__(self, _):
        return False

    def __lt__(self, _):
        return False


seq = 0
logger = logging.getLogger("test")


def _daq_read(*args, **kwargs):
    global seq
    logger.debug(f"_daq_read ({args}, {kwargs})")
    sleep(0.1)
    a = kwargs["data"]
    for i, items in enumerate(a):
        for j, _ in enumerate(items):
            add = (j + seq) % 100
            items[j] = ((1000 + i * 1000) + add) * 100000

    seq = (seq + 1) % 100
    return 0


def _daq_chan_type(arg, aarg, val):
    val._obj.value = analog_output
    logger.debug(f"_daq_chan_type ({arg}, {aarg}, {val})")
    sleep(0.1)
    return 0


def _daq_task_attribute(task, attribute, val, temp_size):
    if temp_size != 6:
        return 6

    val.value = b"mocked"
    logger.debug(
        f"_daq_task_attribute ({task}, {attribute}, {val}, {temp_size})"
    )
    sleep(0.1)
    return 0


def _daq_chan_attribute(task, channel, attribute, val):
    val._obj.value = analog_output

    logger.debug(
        f"_daq_chan_attribute ({task}, {channel}, {attribute}, {val})"
    )
    sleep(0.1)
    return 0


def run_test():
    from gevidaq import __main__ as main

    # warnings are errors
    warnings.filterwarnings("error")
    warnings.filterwarnings("default", category=DeprecationWarning)

    # lower logging level
    logging.getLogger().setLevel(logging.DEBUG)

    # use mocks for ctypes dlls
    sys.modules["gevidaq.CoordinatesManager.DMDActuator"] = Mock("DMDActuator")
    ctypes.WinDLL = Mock("WinDLL")

    # import nidaq for monkeypatching
    import nidaqmx._lib as daq_lib
    import nidaqmx.constants
    import nidaqmx.stream_readers as daq_streams

    # create mock objects for nidaq
    nidaq_dll = Mock("nidaq_dll")
    global analog_output
    analog_output = nidaqmx.constants.ChannelType.ANALOG_OUTPUT.value
    nidaq_dll.DAQmxGetChanType = _daq_chan_type
    nidaq_dll.DAQmxGetChanType.argtypes = True
    nidaq_dll.DAQmxGetChanAttribute = _daq_chan_attribute
    nidaq_dll.DAQmxGetChanAttribute.argtypes = True
    nidaq_dll.DAQmxGetTaskAttribute = _daq_task_attribute
    nidaq_dll.DAQmxGetTaskAttribute.argtypes = True

    # apply nidaq monkeypatches
    daq_streams.AnalogMultiChannelReader.read_many_sample = _daq_read
    daq_lib.DaqLibImporter.windll = nidaq_dll
    daq_lib.DaqLibImporter.cdll = nidaq_dll
    daq_lib.DaqLibImporter._windll = nidaq_dll
    daq_lib.DaqLibImporter.encoding = "utf-8"

    # run fiumchino
    main.run()


if __name__ == "__main__":
    run_test()
