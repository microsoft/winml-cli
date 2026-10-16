"""Process-isolated OpenVINO GPU allocator diagnostic; no system changes."""
import argparse
import ctypes
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys


def worker(args):
    import numpy as np
    import onnx
    import onnxruntime as ort
    import psutil
    from onnx import helper, numpy_helper, TensorProto
    from winml.modelkit.session import EPDeviceTarget, WinMLEPRegistry
    from winml.modelkit.sysinfo import get_ep_device_luid

    session_id = ctypes.c_ulong()
    ctypes.windll.kernel32.ProcessIdToSessionId(os.getpid(), ctypes.byref(session_id))
    selected = WinMLEPRegistry.instance().auto_device(EPDeviceTarget(ep='openvino', device='gpu'))
    devices = [d for d in selected.ep.devices if d.device_type == 'GPU']
    rows = []
    for device in devices:
        handle = device.ort_handle
        memory = handle.memory_info(ort.OrtDeviceMemoryType.HOST_ACCESSIBLE)
        rows.append({'ep': handle.ep_name, 'luid': get_ep_device_luid(handle), 'hardware': dict(handle.device.metadata),
                     'ov_luid': handle.ep_metadata.get('DEVICE_LUID'),
                     'ov_device': handle.ep_metadata.get('ov_device'),
                     'allocator_id': memory.device_id if memory else None,
                     'allocator_name': memory.name if memory else None})
    report = {'pid': os.getpid(), 'session_id': session_id.value,
              'elevated': bool(ctypes.windll.shell32.IsUserAnAdmin()),
              'user': subprocess.check_output(['whoami'], text=True).strip(),
              'python': sys.executable, 'ort': ort.__version__,
              'packages': {p: importlib.metadata.version(p) for p in ['winml-cli', 'onnxruntime-windowsml', 'torch', 'transformers']},
              'default_luid': get_ep_device_luid(selected.device.ort_handle),
              'devices': rows, 'index': args.index, 'env_allocators': args.env_allocators}
    if args.index is not None:
        device = devices[args.index]
        report['selected'] = rows[args.index]
        weights = np.arange(256, dtype=np.float32).reshape(16, 16) / 256
        graph = helper.make_graph([helper.make_node('MatMul', ['X', 'W'], ['Y'])], 'allocator-probe',
            [helper.make_tensor_value_info('X', TensorProto.FLOAT, [1,16])],
            [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1,16])],
            [numpy_helper.from_array(weights, 'W')])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)], ir_version=9)
        model_path = Path(args.output).with_suffix('.onnx')
        onnx.save(model, model_path)
        try:
            options = ort.SessionOptions()
            if args.env_allocators:
                options.add_session_config_entry('session.use_env_allocators', '1')
            options.add_provider_for_devices([device.ort_handle], {})
            session = ort.InferenceSession(str(model_path), sess_options=options)
            x = np.ones((1,16), dtype=np.float32)
            for _ in range(3):
                actual = session.run(None, {'X': x})
                np.testing.assert_allclose(actual[0], x @ weights, rtol=1e-3, atol=1e-3)
            report['result'] = 'PASS'
        except Exception as exc:
            report['result'] = 'FAIL'
            report['error'] = str(exc)
    report['loaded_dlls'] = sorted({m.path for m in psutil.Process().memory_maps() if any(s in m.path.lower() for s in ['openvino', 'onnxruntime.dll'])})
    Path(args.output).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--index', type=int)
    parser.add_argument('--env-allocators', action='store_true')
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    def run(name, extra):
        target = root / (name + '.json')
        with (root / (name + '.log')).open('w', encoding='utf-8') as log:
            completed = subprocess.run([sys.executable, __file__, '--worker', '--output', str(target), *extra],
                stdout=log, stderr=subprocess.STDOUT, timeout=180, check=False)
        if completed.returncode or not target.exists():
            raise RuntimeError(f'{name} did not finish; see log in {root}')
        row = json.loads(target.read_text(encoding='utf-8'))
        print(json.dumps({'probe': name, 'default_luid': row['default_luid'],
                          'selected': row.get('selected'), 'result': row.get('result'), 'error': row.get('error')}), flush=True)
        return row
    inventory = run('inventory', [])
    print('OV_DIAG_INVENTORY ' + json.dumps(inventory), flush=True)
    for index in range(len(inventory['devices'])):
        run(f'gpu-{index}-default', ['--index', str(index)])
        run(f'gpu-{index}-env-allocators', ['--index', str(index), '--env-allocators'])


if __name__ == '__main__':
    main()
