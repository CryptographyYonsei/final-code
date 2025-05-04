# inspect_h5.py

import h5py

for path in ["./Data/ASCAD.h5",
             "./Data/ASCAD_desync50.h5",
             "./Data/ASCAD_desync100.h5",
             "./Data/ATMega8515_raw_traces.h5"]:
    try:
        with h5py.File(path, 'r') as f:
            print(f"\n>>> {path} keys:\n  {list(f.keys())}")
    except Exception as e:
        print(f"\n>>> {path} 열기 실패: {e}")
