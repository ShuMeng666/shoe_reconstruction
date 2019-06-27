from __future__ import division
import numpy as np
from os.path import join
from imageio import get_writer,imread,imwrite
import astra
import matplotlib.pyplot as plt
from skimage import measure,data,filters
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.tri as mtri
import functions as func
import edge_smooth as smooth


# Configuration.
distance_source_origin = 606  # Distance between point source and object
distance_origin_detector = 260  # Distance between the detector and object
detector_pixel_size = 1.6  # pixel size on the detector, downscale from 0.2 tp 1.6
detector_rows = 256  # Vertical size of detector [pixels].
detector_cols = 256  # Horizontal size of detector [pixels].
num_of_projections = 36  #number of projection angles
angles = np.linspace(0, 2 * np.pi, num=num_of_projections, endpoint=False)



# Load project images from folder

input_dir = '22L_360_10 degree pitch_turntable turn'
projections = np.zeros((detector_rows, num_of_projections, detector_cols))
for i in range(num_of_projections):
    im = imread(join(input_dir, '25L_%03d.tif' % (i*10))).astype(float)
    im_downscale = im[0:2048:8, 0:2048:8]
    im_downscale = 65534-im_downscale
    # im /= 65534
    projections[:, i, :] = im_downscale


proj_geom = astra.create_proj_geom('cone', 1, 1, detector_rows, detector_cols, angles,
                           (distance_source_origin + distance_origin_detector) /
                           detector_pixel_size, 0)

# Copy projection images into ASTRA Toolbox.
projections_id = astra.data3d.create('-sino', proj_geom, projections)

# Create reconstruction. Volume is created to store reconstructed image stack
vol_geom = astra.creators.create_vol_geom(detector_cols, detector_cols, detector_rows)

reconstruction_id = astra.data3d.create('-vol', vol_geom, data=0)
alg_cfg = astra.astra_dict('FDK_CUDA') # use FDK_CUDA or SIRT3D_CUDA reconstruction method
alg_cfg['ProjectionDataId'] = projections_id
alg_cfg['ReconstructionDataId'] = reconstruction_id
algorithm_id = astra.algorithm.create(alg_cfg)
astra.algorithm.run(algorithm_id)
reconstruction = astra.data3d.get(reconstruction_id)

# Limit and scale reconstruction.
reconstruction[reconstruction < 0] = 0
reconstruction /= np.max(reconstruction)
reconstruction = np.round(reconstruction * 255).astype(np.uint8)


output_dir = 'reconstruction_shoe'

# Save reconstruction.
for i in range(0,140):
    im = reconstruction[:,:,i]
    im = np.flipud(im)
    imwrite(join(output_dir, 'reco%04d.jpg' % i), im)

fig = plt.figure()
for i in range(110, 130):
    fig.suptitle('layer: ' + str(i + 1))
    plt.imshow(reconstruction[:,:,i], cmap='gray')
    plt.pause(.001)


# print(reconstruction.shape)




# reconstruction = np.round(reconstruction * 255).astype(np.uint8)


# # use Otsu thresholding method to enhance blurring boundary
# '''to solve: 2d image smooth to get rid of boundary oscillation, edge smooth? '''
# enhanced_reconstruction = np.zeros_like(reconstruction)
# val = filters.threshold_otsu(reconstruction)
# mask = reconstruction < val
# enhanced_reconstruction = 1 - mask.astype(int)


# extract mesh from the volumeric image stack
# verts_rec, faces_rec, normals_rec, values_rec = measure.marching_cubes_lewiner(enhanced_reconstruction, level=None, spacing=(1.0, 1.0, 1.0), gradient_direction='descent', step_size=3, allow_degenerate=True, use_classic=False)
# verts_ori, faces_ori, normals_ori, values_ori = measure.marching_cubes_lewiner(phantom, level=None, spacing=(1.0, 1.0, 1.0), gradient_direction='descent', step_size=3, allow_degenerate=True, use_classic=False)




# fig = plt.figure()
# fig.suptitle('angle number: ' + str(angles.shape))
# ax = fig.add_subplot(1, 2, 1, projection='3d')
# ax.plot_trisurf(verts_rec[:,0],verts_rec[:,1],verts_rec[:,2],triangles=faces_rec, edgecolor='k',alpha=0)
# # ax.set_xlim(0, reconstruction.shape[0])
# # ax.set_ylim(0, reconstruction.shape[1])
# # ax.set_zlim(0, reconstruction.shape[2])
# # plt.show()
# ax.title.set_text('reconstructed')
#
# ax = fig.add_subplot(1, 2, 2, projection='3d')
# ax.plot_trisurf(verts_ori[:,0],verts_ori[:,1],verts_ori[:,2],triangles=faces_ori, edgecolor='r',alpha=0)
# # ax.set_xlim(0, reconstruction.shape[0])
# # ax.set_ylim(0, reconstruction.shape[1])
# # ax.set_zlim(0, reconstruction.shape[2])
# plt.show()
# ax.title.set_text('ground truth')


# fig2 = plt.figure()
# for i in range(89, 90):
#     fig2.suptitle('layer: ' + str(i + 1))
#     plt.subplot(121)
#     plt.imshow(phantom[:,i,:], cmap='gray')
#
#     plt.subplot(122)
#     plt.imshow(enhanced_reconstruction[:,i,:], cmap='gray')
#     plt.pause(.001)

# Cleanup GPU
astra.algorithm.delete(algorithm_id)
astra.data3d.delete(reconstruction_id)
astra.data3d.delete(projections_id)