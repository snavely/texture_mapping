# Script for testing rendering by rendering a COLMAP point cloud.

import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
import json
import re
import numpy as np
import argparse
import trimesh
import pyrender
import png
import time
import cv2
from pyquaternion import Quaternion
from satellite_stereo.lib import latlon_utm_converter
from satellite_stereo.lib import latlonalt_enu_converter
from satellite_stereo.lib.plyfile import PlyData, PlyElement

# Compute the dimensions of a new image resized such that the max
# dimension (width or height) is at most max_dim. Returns a tuple
# (resized_width, resized_height).
def resized_image_dims_for_max_dim(imwidth, imheight, max_dim):
    if imwidth <= max_dim and imheight <= max_dim:
        return (imwidth, imheight)

    if float(imwidth) / max_dim > float(imheight) / max_dim:
        resized_dims = (max_dim,
                        int(round(float(imheight) * max_dim / imwidth)))
    else:
        resized_dims = (int(round(float(imwidth) * max_dim / imheight)),
                        max_dim)

    return resized_dims

# Resize the provided color buffer to the provided maximum size (on
# either dimension), and save to a png file called 'image_name'.
def resize_and_save_color_buffer_to_png(image, max_dim, image_name):
    height = np.shape(image)[0]
    width = np.shape(image)[1]

    if width <= max_dim and height <= max_dim:
        png.from_array(image, 'RGB').save(image_name)
    else:
        resized_dims = resized_image_dims_for_max_dim(width, height, max_dim)
        resized = cv2.resize(image, dsize=resized_dims,
                             interpolation=cv2.INTER_AREA)
        png.from_array(resized, 'RGB').save(image_name)

# Transform a depth map to the range [0,255].
def normalize_and_discretize_depth_buffer(depth):
    # Depth of zero is a sentinel value.
    depth_masked = np.ma.masked_equal(depth, 0.0)
    depth_min = depth_masked.min(axis=0).min(axis=0)
    depth_max = depth_masked.max(axis=0).max(axis=0)
    depth_normalized = (255 * (depth_masked - depth_min) /
                        (depth_max - depth_min)).filled(0).astype(np.uint8)
    return depth_normalized

# Normalize the values of and resize the provided depth buffer to the
# provided maximum size (on either dimension), and save to a png file
# called 'image_name'.
def resize_and_save_depth_buffer_to_png(depth, max_dim, image_name):
    depth_normalized = normalize_and_discretize_depth_buffer(depth)

    height = np.shape(depth_normalized)[0]
    width = np.shape(depth_normalized)[1]

    if width <= max_dim and height <= max_dim:
        png.from_array(depth_normalized, 'L').save(image_name)
    else:
        resized_dims = resized_image_dims_for_max_dim(width, height, max_dim)
        resized = cv2.resize(depth_normalized, dsize=resized_dims,
                             interpolation=cv2.INTER_AREA)
        png.from_array(resized, 'L').save(image_name)


class PerspectiveCamera(object):
    def __init__(self, image_name, camera_spec):
        self.image_name = image_name
        self.width = camera_spec[0]
        self.height = camera_spec[1]
        self.K = np.array([[camera_spec[2],            0.0, camera_spec[4]],
                           [           0.0, camera_spec[3], camera_spec[5]],
                           [           0.0,            0.0,            1.0]])
        quat = Quaternion(camera_spec[6], camera_spec[7],
                          camera_spec[8], camera_spec[9])
        self.R = quat.rotation_matrix
        self.t = np.array([camera_spec[10],
                           camera_spec[11],
                           camera_spec[12]]).transpose()

        # Convert pose from Y-Down to Y-Up ("OpenGL") coordinates.
        X180 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        self.R = np.dot(X180, self.R)
        self.t = np.dot(X180, self.t)
        
        self.pose = np.concatenate(
            (np.concatenate((self.R, np.expand_dims(self.t, axis=1)), axis=1),
             np.array([[0, 0, 0, 1]])), axis=0)
        self.pose = np.linalg.inv(self.pose)

        # Compute a reasonable zNear and zFar, based on the projection
        # of the camera location on the (negative) viewing direction,
        # assuming that the scene is located near the origin.
        camera_pos = -np.dot(np.transpose(self.R), self.t)
        view_dir = np.dot(np.transpose(self.R),
                          np.array([[0.0], [0.0], [-1.0]]))
        scene_distance = -np.dot(np.transpose(camera_pos), view_dir)

        znear = max(scene_distance - 1e5, 1.0)
        zfar = scene_distance + 1e5

        self.pyrender_camera = pyrender.IntrinsicsCamera(
            fx=camera_spec[2], fy=camera_spec[3],
            cx=camera_spec[4], cy=camera_spec[5],
            znear=znear, zfar=zfar, name=image_name)

    def project(self, point):
        proj3 = np.dot(self.K, np.dot(self.R, np.transpose(point)) + self.t)
        proj = np.array([-proj3[0] / proj3[2],
                         -proj3[1] / proj3[2]]).transpose()
        return proj

class Reconstruction(object):
    def __init__(self, recon_path):
        if not os.path.isabs(recon_path):
            fpath = os.path.abspath(recon_path)
        self.recon_path = recon_path

        # Read the camera data.
        with open(
            os.path.join(
            recon_path,
            'colmap/sfm_pinhole/debug/kai_cameras.json')) as fp:
            # 'colmap/skew_correct/pinhole_dict.json')) as fp:
            camera_data = json.load(fp)
            self.cameras = {}
            for image, camera in camera_data.items():
                self.cameras[image] = PerspectiveCamera(image, camera)

    def write_meta(self, fname):
        with open(fname, 'w') as fp:
            json.dump(self.meta, fp, indent=2)


class TestCameras(object):
    def __init__(self, ply_path, recon_path):
        self.reconstruction = Reconstruction(recon_path)

        self.points = np.loadtxt(
            '/phoenix/S2/snavely/code/CORE3D/texture_mapping/points.txt')
        num_points, _ = np.shape(self.points)

        print 'num_points:', num_points
        colors = np.zeros((num_points, 4), dtype=np.uint8)
        for i in xrange(0, num_points):
            colors[i,:] = trimesh.visual.random_color()

        self.mesh = pyrender.Mesh.from_points(self.points, colors=colors)

        self.scene = pyrender.Scene(ambient_light=(1.0, 1.0, 1.0))
        self.scene.add(self.mesh)

        self.ply_textured = None
        # self.texture_ply()

    def test_rendering(self):
        width = 2000
        height = 2000
        renderer = pyrender.OffscreenRenderer(width, height, point_size=10.0)
        test_camera = pyrender.IntrinsicsCamera(
            fx=866.0 * 1000.0, fy=866.0 * 1000.0, cx=1000.0, cy=1000.0,
            znear=1000.0, zfar=1.0e8)
        test_camera_pose = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0],
                                     [0, 0, 1, 1.0e6],
                                     [0, 0, 0, 1]])

        self.scene.add(test_camera, pose=test_camera_pose)
        t = time.time()
        color, depth = renderer.render(self.scene)
        elapsed = time.time() - t
        print 'Time to render:', elapsed

        resize_and_save_color_buffer_to_png(color, 1024, 'test_render.png')
        resize_and_save_depth_buffer_to_png(depth, 1024, 'test_depth.png')

    def test_rendering_on_real_camera(self):
        image, camera = (self.reconstruction.cameras.items())[0]
        print 'rendering image', image

        color, depth = self.render_from_camera(camera)

        # png.from_array(color, 'RGB').save(image + '_render.png')
        resize_and_save_color_buffer_to_png(color, 1e6, image + '_render.png')
        resize_and_save_depth_buffer_to_png(depth, 1e6, image + '_depth.png')

    # Render the loaded scene from the provided camera. Returns color
    # and depth buffers.
    def render_from_camera(self, camera):
        renderer = pyrender.OffscreenRenderer(
            camera.width, camera.height, point_size=10.0)

        print 'camera.K:'
        print camera.K
        print 'camera.pose (inverted):'
        print camera.pose
        print 'projection_matrix:'
        print camera.pyrender_camera.get_projection_matrix(camera.width,
                                                           camera.height)
        node = self.scene.add(camera.pyrender_camera, pose=camera.pose)

        t = time.time()
        color, depth = renderer.render(self.scene)
        elapsed = time.time() - t
        print 'Time to render:', elapsed

        self.scene.remove_node(node)

        return color, depth

    def render_all_cameras(self):
        for image, camera in self.reconstruction.cameras.items():
            print 'rendering image', image
            print 'camera.K:'
            print camera.K
            print 'camera.pose:'
            print camera.pose

            color, depth = self.render_from_camera(camera)

            resize_and_save_color_buffer_to_png(color, 1e6, # 1024,
                                                image + '_render.png')
            # resize_and_save_depth_buffer_to_png(depth, 1e6, # 1024,
            #                                     image + '_depth.png')


def test():
    # Base path for the reconstruction (cameras and images) to be used
    # in texture mapping.
    recon_path = 'testdata'

    # Location of the ply file to be texture mapped.
    ply_path = 'testdata/aoi.ply'
    test_cameras = TestCameras(ply_path, recon_path)

    # test_cameras.test_rendering()
    test_cameras.test_rendering_on_real_camera()
    # test_cameras.render_all_cameras()

if __name__ == '__main__':
    test()
