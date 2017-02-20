import numpy as np

from vispy import io, plot as vp

fig = vp.Fig(bgcolor='k', size=(800, 800), show=False)

#vol_data = np.load(io.load_data_file('brain/mri.npz'))['data']
vol_data = np.load('/home/yunfanz/Data/Kaggle/LungCan/stage1_processed/sp2_waterseg/train/0c0de3749d4fe175b7a5098b060982a1.npy')
vol_data = np.flipud(np.rollaxis(vol_data, 1))

clim = [0, 1]
vol_pw = fig[0, 0]
vol_pw.volume(vol_data, clim=None)
vol_pw.view.camera.elevation = 30
vol_pw.view.camera.azimuth = 30
vol_pw.view.camera.scale_factor /= 1.5

shape = vol_data.shape
fig[1, 0].image(vol_data[:, :, shape[2] // 2], cmap='grays', clim=clim,
                fg_color=(0.5, 0.5, 0.5, 1))
fig[0, 1].image(vol_data[:, shape[1] // 2, :], cmap='grays', clim=clim,
                fg_color=(0.5, 0.5, 0.5, 1))
fig[1, 1].image(vol_data[shape[0] // 2, :, :].T, cmap='grays', clim=clim,
                fg_color=(0.5, 0.5, 0.5, 1))

if __name__ == '__main__':
    fig.show(run=True)

