# Script for generating a "full" texture map for a primitive model,
# including sidewalls, given a collection of images with camera poses.

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
            fpath = os.path.abspath(recon_Path)
        self.recon_path = recon_path

        # Get metadata from aoi.json file.
        with open(os.path.join(recon_path, 'aoi.json')) as fp:
            self.bbox = json.load(fp)
            self.lat0 = (self.bbox['lat_min'] + self.bbox['lat_max']) / 2.0
            self.lon0 = (self.bbox['lon_min'] + self.bbox['lon_max']) / 2.0
            self.alt0 = self.bbox['alt_min']

        # Read the UTM zone information.
        self.utm_zone = self.bbox['zone_number']
        self.hemisphere = self.bbox['hemisphere']

        # Read the camera data.
        with open(
            os.path.join(
            recon_path,
            'colmap/skew_correct/pinhole_dict.json')) as fp:
            camera_data = json.load(fp)
            self.cameras = {}
            for image, camera in camera_data.items():
                self.cameras[image] = PerspectiveCamera(image, camera)
                # print self.pyrender_cameras[image].get_projection_matrix(camera[0], camera[1])

    def write_meta(self, fname):
        with open(fname, 'w') as fp:
            json.dump(self.meta, fp, indent=2)

    def utm_to_enu(self, point):
        # Convert a point in UTM coordinates to ENU.
        lat, lon = latlon_utm_converter.eastnorth_to_latlon(point[:, 0:1],
                                                            point[:, 1:2],
                                                            self.utm_zone,
                                                            self.hemisphere)
        alt = point[:, 2:3]
        x, y, z = latlonalt_enu_converter.latlonalt_to_enu(lat, lon, alt,
                                                           self.lat0,
                                                           self.lon0,
                                                           self.alt0)
        return np.concatenate((x, y, z), axis=1)

    def norm_coord(self, point):
        # the point is in utm coordinate
        # (easting, northing, elevation)
        (x, y, z) = point

        u = (x - self.ll[0]) / (self.lr[0] - self.ll[0])
        v = (y - self.ll[1]) / (self.ul[1] - self.ll[1])

        return u, v


class FullTextureMapper(object):
    def __init__(self, ply_path, recon_path):
        self.reconstruction = Reconstruction(recon_path)
        self.ply_data = PlyData.read(ply_path)
        self.vertices = self.ply_data.elements[0]
        self.faces = self.ply_data.elements[1]

        self.tmesh = trimesh.load(ply_path)

        # Transform vertices from UTM to ENU.
        vertices_enu = self.reconstruction.utm_to_enu(self.tmesh.vertices)
        print 'tmesh.vertices_enu:', vertices_enu[0:2, :]
        self.tmesh.vertices = vertices_enu

        # Recolor the facets.
        num_facets = self.tmesh.facets.size
        print 'number of facets:', num_facets
        color_index = long(1)

        # TODO(snavely): Why are some facets showing up as gray? Are
        # they somehow facing the wrong direction? Do those faces not
        # show up in the list of facets?
        for facet in self.tmesh.facets:
            # Random trimesh colors have random hue but nearly full
            # saturation and value. Useful for visualization and
            # debugging.

            # tmesh.visual.face_colors[facet] = trimesh.visual.random_color()
            r, g, b = self.color_index_to_color(color_index)
            # Last 255 is for alpha channel (fully opaque).
            self.tmesh.visual.face_colors[facet] = np.array((r, g, b, 255))
            color_index = color_index + 1
        
        self.mesh = pyrender.Mesh.from_trimesh(self.tmesh, smooth=False)
        self.scene = pyrender.Scene(ambient_light=(1.0, 1.0, 1.0))
        self.scene.add(self.mesh)

        self.ply_textured = None
        # self.texture_ply()

    def color_index_to_color(self, color_index):
        # red is the lower 8-bits, then green, then blue.
        r = color_index & 0xff
        g = (color_index >> 8) & 0xff
        b = (color_index >> 16) & 0xff
        return r, g, b

    def color_buffer_to_color_indices(self, color):
        # red is the lower 8-bits, then green, then blue.
        color_indices = (
            color[:,:,0] + 0xff * color[:,:,1] + 0xffff * color[:,:,2])
        return color_indices

    def test_rendering(self):
        width = 2000
        height = 2000
        renderer = pyrender.OffscreenRenderer(width, height)
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
        image, camera = (self.reconstruction.cameras.items())[1]
        print 'rendering image', image

        color, depth = self.render_from_camera(camera)

        # png.from_array(color, 'RGB').save(image + '_render.png')
        resize_and_save_color_buffer_to_png(color, 1024, image + '_render.png')
        resize_and_save_depth_buffer_to_png(depth, 1024, image + '_depth.png')

    # Render the loaded scene from the provided camera. Returns color
    # and depth buffers.
    def render_from_camera(self, camera):
        renderer = pyrender.OffscreenRenderer(camera.width, camera.height)

        node = self.scene.add(camera.pyrender_camera, pose=camera.pose)

        t = time.time()
        color, depth = renderer.render(self.scene)
        elapsed = time.time() - t
        print 'Time to render:', elapsed

        self.scene.remove_node(node)

        return color, depth

    def create_textures(self):
        num_cameras = len(self.reconstruction.cameras)
        num_facets = self.tmesh.facets.size

        print 'num_cameras:', num_cameras
        print 'num_facets:', num_facets
        
        # Num cameras by num facets matrix counting the visibility of
        # each facet in each image
        facet_pixel_counts = np.zeros((num_cameras, num_facets),
                                      dtype=np.int16)

        camera_index = 0
        for image, camera in self.reconstruction.cameras.items():
            print 'rendering image', image

            color, depth = self.render_from_camera(camera)

            # Count number of times each color appears.
            color_indices = self.color_buffer_to_color_indices(color)
            elems, counts = np.unique(color_indices, return_counts=True)
            print 'unique colors:', elems.size

            for elem, count in zip(elems, counts):
                if elem > 0 and elem <= num_facets:
                    facet_index = elem - 1
                    facet_pixel_counts[camera_index, facet_index] = count

            # resize_and_save_color_buffer_to_png(color, 1024,
            #                                     image + '_render.png')
            # resize_and_save_depth_buffer_to_png(depth, 1024,
            #                                     image + '_depth.png')

    # write texture coordinate to vertex
    def texture_ply(self):
        # drop the RGB properties, and add two new properties (u, v)
        # vert_list = []
        # for vert in self.vertices.data:
        #     vert = vert.tolist()   # convert to tuple
        # vertices_utm = np.reshape(self.vertices.data
        vertices_utm = np.stack((self.vertices['x'],
                                 self.vertices['y'],
                                 self.vertices['z']), axis=1)
        vertices_enu = self.reconstruction.utm_to_enu(vertices_utm)

        # vert_list.append(xyz)

        # vert_list.append(vert[0:3]+(u, v))
        # vertices = np.array(vert_list,
        #                     dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4')])
        #                           ('u', '<f4'), ('v', '<f4')])
        # vert_el = PlyElement.describe(vertices, 'vertex',
        #                                comments=['point coordinate, texture coordinate'])
        # self.ply_textured = PlyData([vert_el, self.faces], text=True)
        print 'ply_vertices:', vertices_enu[0:2,:]

        renderer = pyrender.OffscreenRenderer(1000, 1000)
        for image, camera in self.reconstruction.cameras.items():
            print camera.project(vertices_enu[0,:])
            test_camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
            test_camera_pose = np.array([[1, 0, 0, 0],
                                         [0, 1, 0, 0],
                                         [0, 0, 1, 1000.0],
                                         [0, 0, 0, 1]])
            # self.scene.add(camera.pyrender_camera, camera.pose)
            self.scene.add(test_camera, pose=test_camera_pose)
            light = pyrender.SpotLight(color=np.ones(3),
                                       intensity=3.0,
                                       innerConeAngle=np.pi/16.0)
            # self.scene.add(light, pose=camera.pose)
            # self.scene.add(light, pose=test_camera_pose)
            # renderer = pyrender.OffscreenRenderer(camera.width, camera.height)
            color, depth = renderer.render(self.scene)
            # png.from_array(color, 'RGB').save(image + '_render.png')
            png.from_array(color, 'RGB').save('test_render.png')


    # fname should not come with a file extension
    def save_texture(self, fname):
        # convert tiff to jpg
        os.system('gdal_translate -ot Byte -of jpeg {} {}.jpg'.format(self.tiff.fpath, fname))
        # remove the intermediate file
        os.remove(fname + '.jpg.aux.xml')

    # fname and texture_fname should not come with a file extension
    def save_ply(self, fname, texture_fname):
        name = texture_fname[texture_fname.rfind('/')+1:]
        self.ply_textured.comments = ['TextureFile {}.jpg'.format(name), ]   # add texture file into the comment
        self.ply_textured.write('{}.ply'.format(fname))
        TextureMapper.insert_uv_to_face('{}.ply'.format(fname))

    def save(self, fname):
        # convert tiff to jpg
        os.system('gdal_translate -ot Byte -of jpeg {} {}.jpg'.format(self.tiff.fpath, fname))
        # remove the intermediate file
        os.remove(fname + '.jpg.aux.xml')
        # save ply
        name = fname[fname.rfind('/')+1:]
        self.ply_textured.comments = ['TextureFile {}.jpg'.format(name), ]   # add texture file into the comment
        self.ply_textured.write('{}.ply'.format(fname))
        TextureMapper.insert_uv_to_face('{}.ply'.format(fname))

    # write texture coordinate to face
    @staticmethod
    def insert_uv_to_face(ply_path):
        ply = PlyData.read(ply_path)
        uv_coord = ply['vertex'][['u', 'v']]
        vert_cnt = ply['vertex'].count

        with open(ply_path) as fp:
            all_lines = fp.readlines()
        modified = []
        flag = False; cnt = 0
        for line in all_lines:
            line = line.strip()
            if cnt < vert_cnt:
                modified.append(line)
            if line == 'property list uchar int vertex_indices':
                modified.append('property list uchar float texcoord')
            if flag:
                cnt += 1
            if line == 'end_header':
                flag = True
            if cnt > vert_cnt: # start modify faces
                face = [int(x) for x in line.split(' ')]
                face_vert_cnt = face[0]
                line += ' {}'.format(face_vert_cnt * 2)
                for i in range(1, face_vert_cnt + 1):
                    idx = face[i]
                    line += ' {} {}'.format(uv_coord[idx]['u'],  uv_coord[idx]['v'])
                modified.append(line)
        with open(ply_path, 'w') as fp:
            fp.writelines([line + '\n' for line in modified])


def test():
    print('entering test')
    img = TifImg('/home/kai/satellite_project/sync_folder/true_ortho.tif')
    img.write_meta('true_ortho_meta.json')
    print('all tests passed!')


def test2():
    # Base path for the reconstruction (cameras and images) to be used
    # in texture mapping.
    recon_path = '/phoenix/S7/kz298/core3d_result/aoi-d4-jacksonville'
    # recon_path = '/phoenix/S7/kz298/core3d_result/aoi-d5-san_fernando'

    # Location of the ply file to be texture mapped.
    ply_path = '/phoenix/S2/snavely/data/CORE3D/aws/data/wdixon/jul_test1/jacksonville_d4/buildings_prim/fitting/scores/aoi.ply'
    # ply_path = '/phoenix/S2/snavely/data/CORE3D/aws/data/wdixon/jul_test1/argentina_d5/buildings_prim/fitting/scores/aoi.ply'
    texture_mapper = FullTextureMapper(ply_path, recon_path)

    # texture_mapper.test_rendering()
    # texture_mapper.test_rendering_on_real_camera()
    texture_mapper.create_textures()

    # texture_mapper.save('/home/kai/satellite_project/d2_texture_result/012_5_nonBox_textured')


def deploy():
    parser = argparse.ArgumentParser(description='texture-map a .ply to a .tif ')
    parser.add_argument('mesh', help='path/to/.ply/file')
    parser.add_argument('orthophoto', help='path/to/.tif/file')
    parser.add_argument('filename', help='filename for the output files. will output '
                                       '{filename}.ply and {filename}.jpg')
    args = parser.parse_args()

    texture_mapper = TextureMapper(args.mesh, args.orthophoto)
    texture_mapper.save(args.filename)


if __name__ == '__main__':
    # test()
    test2()
    # deploy()
