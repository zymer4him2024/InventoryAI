"""Quick Hailo-8 inference test — run on RPi5 to verify model loading and output shape."""

from hailo_platform import (
    HailoStreamInterface,
    ConfigureParams,
    HEF,
    VDevice,
    InputVStreamParams,
    OutputVStreamParams,
    InferVStreams,
    FormatType,
)
import numpy as np

hef = HEF("/usr/share/hailo-models/yolov8s_h8.hef")
vd = VDevice()
p = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
ng = vd.configure(hef, p)[0]
ip = InputVStreamParams.make(ng, format_type=FormatType.FLOAT32)
op = OutputVStreamParams.make(ng, format_type=FormatType.FLOAT32)
img = np.random.rand(1, 640, 640, 3).astype(np.float32)
input_name = hef.get_input_vstream_infos()[0].name

with ng.activate():
    with InferVStreams(ng, ip, op) as vs:
        r = vs.infer({input_name: img})

k = list(r.keys())[0]
print("shape:", r[k].shape)
d = r[k][0]
m = d[:, :, 4] > 0.5
print("dets>0.5:", m.sum())
print("sample:", d[m][:3])
