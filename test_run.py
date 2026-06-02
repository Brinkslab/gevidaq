import ctypes
import logging
import sys
import warnings
from time import sleep

from gevidaq import __main__ as main


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


logger = logging.getLogger("test")


def _make_daq_read():
    seq = 0

    def _daq_read(*args, **kwargs):
        nonlocal seq
        logger.debug(f"_daq_read ({args}, {kwargs})")
        sleep(0.1)
        a = kwargs["data"]
        for i, items in enumerate(a):
            for j, _ in enumerate(items):
                add = (j + seq) % 100
                items[j] = ((1000 + i * 1000) + add) * 100000

        seq = (seq + 1) % 100
        return 0

    return _daq_read


def _daq_write(*args, **kwargs):
    logger.debug(f"_daq_write ({args}, {kwargs})")


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


def _make_BMC_GetNextMessage():
    count = 0

    def BMC_GetNextMessage(
        serial, device, message_type, message_id, message_data
    ):
        nonlocal count
        logger.debug(
            f"BMC_GetNextMessage ({serial}, {device}, {message_type}, "
            f"{message_id}, {message_data}) count = {count}"
        )

        if count == 0:
            message_id._obj.value = 0
        else:
            message_id._obj.value = 1

        message_type._obj.value = 2
        count += 1

    return BMC_GetNextMessage


def _dcamapi_init(paraminit):
    paraminit._obj.iDeviceCount = 1
    return 1  # DCAMERR_NOERROR


PROPTABLE = [
    "binning",
    "buffer_framebytes",
    "defect_correct_mode",
    "exposure_time",
    "image_framebytes",
    "image_height",
    "image_width",
    "internal_frame_rate",
    "output_trigger_kind[0]",
    "readout_speed",
    "subarray_hpos",
    "subarray_hsize",
    "subarray_mode",
    "subarray_vpos",
    "subarray_vsize",
    "timing_readout_time",
    "trigger_active",
    "trigger_source",
]


def _make_dcamprop_getnextid():
    i = -1

    def _dcamprop_getnextid(cam, prop_id, opion):
        nonlocal i
        if i < len(PROPTABLE) - 1:
            i += 1

        prop_id._obj.value = i
        return 1  # DCAMERR_NOERROR

    return _dcamprop_getnextid


def _dcamprop_getname(cam, prop_id, buf, buflen):
    buf.value = PROPTABLE[prop_id.value].encode()
    return 1  # DCAMERR_NOERROR


def _dcamprop_getattr(cam, attr):
    attr._obj.attribute = 2  # DCAMPROP_TYPE_LONG
    return 1  # DCAMERR_NOERROR


def _dcamprop_getvalue(cam, prop_id, value):
    value._obj.value = 1
    return 1  # DCAMERR_NOERROR


def _dcamcap_status(handle, status):
    sleep(1)
    return 1  # DCAMERR_NOERROR


def _dcamcap_transferinfo(handle, transfer):
    transfer._obj.nFrameCount = 1
    return 1  # DCAMERR_NOERROR


def run_test():
    # warnings are errors
    warnings.filterwarnings("error")
    warnings.filterwarnings("default", category=DeprecationWarning)

    # lower logging level
    logging.getLogger().setLevel(logging.DEBUG)

    # use mocks for ctypes dlls
    sys.modules["gevidaq.CoordinatesManager.DMDActuator"] = Mock("DMDActuator")
    ctypes.WinDLL = Mock("WinDLL")
    ctypes.cdll = Mock("cdll")

    # mock thorlabs dll functions
    ctypes.cdll.BMC_GetNextMessage = _make_BMC_GetNextMessage()

    # mock hamamatsu dcam dll functions
    ctypes.WinDLL.dcamapi_init = _dcamapi_init
    ctypes.WinDLL.dcamprop_getnextid = _make_dcamprop_getnextid()
    ctypes.WinDLL.dcamprop_getname = _dcamprop_getname
    ctypes.WinDLL.dcamprop_getattr = _dcamprop_getattr
    ctypes.WinDLL.dcamprop_getvalue = _dcamprop_getvalue
    ctypes.WinDLL.dcamcap_status = _dcamcap_status

    # import nidaq for monkeypatching
    import nidaqmx._lib as daq_lib
    import nidaqmx.constants
    import nidaqmx.stream_readers as daq_rstreams
    import nidaqmx.stream_writers as daq_wstreams

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
    daq_rstreams.AnalogMultiChannelReader.read_many_sample = _make_daq_read()
    daq_wstreams.AnalogMultiChannelWriter.write_many_sample = _daq_write
    daq_lib.DaqLibImporter.windll = nidaq_dll
    daq_lib.DaqLibImporter.cdll = nidaq_dll
    daq_lib.DaqLibImporter._windll = nidaq_dll
    daq_lib.DaqLibImporter.encoding = "utf-8"

    # disable PW
    from gevidaq.NIDAQ import WaveformWidget

    WaveformWidget.DISABLE_PW = True
    # TODO: fix this crash so workaround is no longer needed

    # run fiumchino
    main.run()


if __name__ == "__main__":
    main.set_up_logging()
    run_test()
