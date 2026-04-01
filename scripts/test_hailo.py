"""Quick Hailo-8 inference test — run on RPi5 to verify model loading and output shape."""

import numpy as np
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

hef = HEF("/usr/share/hailo-models/yolov8s_h8.hef")

with VDevice() as vdevice:
    params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
    ng = vdevice.configure(hef, params)[0]
    ng_params = ng.create_params()

    input_info = hef.get_input_vstream_infos()[0]
    print(f"input: {input_info.name} shape={input_info.shape}")

    ip = InputVStreamParams.make(ng, format_type=FormatType.FLOAT32)
    op = OutputVStreamParams.make(ng, format_type=FormatType.FLOAT32)

    image = np.random.rand(*input_info.shape).astype(np.float32)
    input_data = {input_info.name: np.expand_dims(image, axis=0)}
    print(f"input_data shape={input_data[input_info.name].shape} dtype={input_data[input_info.name].dtype}")

    with ng.activate(ng_params):
        with InferVStreams(ng, ip, op) as pipeline:
            r = pipeline.infer(input_data)

    for name, val in r.items():
        print(f"output: {name}")
        print(f"  type: {type(val)}")
        if isinstance(val, np.ndarray):
            print(f"  shape: {val.shape}")
        elif isinstance(val, list):
            print(f"  len: {len(val)}")
            for i, item in enumerate(val[:2]):
                arr = np.array(item) if not isinstance(item, np.ndarray) else item
                print(f"  [{i}] shape={arr.shape}")

print("DONE")
