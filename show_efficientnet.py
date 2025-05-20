import h5py
with h5py.File("modelos_efficientnet/efficientnet_Face2Face.h5", 'r') as f:
    print(f.attrs['model_config'])