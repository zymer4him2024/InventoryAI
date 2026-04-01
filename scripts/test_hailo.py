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

for name, val in r.items():
    print(f"output: {name}")
    print(f"  type: {type(val)}")
    if isinstance(val, np.ndarray):
        print(f"  shape: {val.shape}")
        print(f"  sample: {val.flat[:10]}")
    elif isinstance(val, list):
        print(f"  len: {len(val)}")
        for i, item in enumerate(val[:2]):
            arr = np.array(item) if not isinstance(item, np.ndarray) else item
            print(f"  [{i}] type={type(item)} shape={arr.shape} sample={arr.flat[:10]}")
