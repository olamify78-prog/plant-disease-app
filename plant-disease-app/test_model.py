import h5py

try:
    with h5py.File("plant_disease_model.h5", "r") as f:
        print("HDF5 file is valid and readable! Keys:", list(f.keys()))
except Exception as e:
    print(f"File error: {e}")