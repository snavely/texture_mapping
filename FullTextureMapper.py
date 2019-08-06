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
import imageio
import subprocess
from pyquaternion import Quaternion
from satellite_stereo.lib import latlon_utm_converter
from satellite_stereo.lib import latlonalt_enu_converter
from satellite_stereo.lib.plyfile import PlyData, PlyElement
import shutil

# Compute the dimensions of a new image resized such that the max
# dimension (width or height) is at most max_dim. Returns a tuple
# (resized_width, resized_height) and an optional scale factor.
def resized_image_dims_for_max_dim(imwidth, imheight, max_dim):
    if imwidth <= max_dim and imheight <= max_dim:
        return (imwidth, imheight), 1.0

    if float(imwidth) / max_dim > float(imheight) / max_dim:
        scale = max_dim / imwidth
        resized_dims = (max_dim,
                        max(int(round(float(imheight) * max_dim / imwidth)), 1))
    else:
        scale = max_dim / imheight
        resized_dims = (max(int(round(float(imwidth) * max_dim / imheight)), 1),
                        max_dim)

    return resized_dims, scale

# Resize the provided color buffer to the provided maximum size (on
# either dimension), and save to a png file called 'image_name'.
def resize_and_save_color_buffer_to_png(image, max_dim, image_name):
    height = np.shape(image)[0]
    width = np.shape(image)[1]

    if width <= max_dim and height <= max_dim:
        png.from_array(image, 'RGB').save(image_name)
    else:
        resized_dims, _ = resized_image_dims_for_max_dim(width, height, max_dim)
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
        resized_dims, _ = resized_image_dims_for_max_dim(width, height, max_dim)
        resized = cv2.resize(depth_normalized, dsize=resized_dims,
                             interpolation=cv2.INTER_AREA)
        png.from_array(resized, 'L').save(image_name)

def unit_projection_onto_plane(vector, normal):
    projection = vector - np.dot(vector, normal) * normal
    return projection / np.linalg.norm(projection)


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

        # Save the "standard" y-down pose.
        self.pose_ydown = np.concatenate(
            (np.concatenate((self.R, np.expand_dims(self.t, axis=1)), axis=1),
             np.array([[0, 0, 0, 1]])), axis=0)

        # Convert pose from Y-Down to Y-Up ("OpenGL") coordinates.
        X180 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        self.R = np.dot(X180, self.R)
        self.t = np.dot(X180, self.t)

        self.pose = np.concatenate(
            (np.concatenate((self.R, np.expand_dims(self.t, axis=1)), axis=1),
             np.array([[0, 0, 0, 1]])), axis=0)
        # pyrender expects us to provide a camera-to-world transform, so
        # invert the pose.
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

    def project_ydown(self, points):
        num_points = np.shape(points)[0]

        # Convert points to homogeneous coordinates.
        points_homogeneous = np.concatenate(
            (points, np.ones((num_points, 1))), axis=1)
        proj3 = np.dot(self.K,
                       np.dot(self.pose_ydown,
                              np.transpose(points_homogeneous))[0:3])
        proj = np.array([proj3[0] / proj3[2],
                         proj3[1] / proj3[2]]).transpose()
        return proj

class Reconstruction(object):
    def __init__(self, recon_path):
        if not os.path.isabs(recon_path):
            fpath = os.path.abspath(recon_path)
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
            'colmap/sfm_pinhole/debug/kai_cameras.json')) as fp:
            # 'colmap/skew_correct/pinhole_dict.json')) as fp:
            camera_data = json.load(fp)
            self.cameras = {}
            for image, camera in camera_data.items():
                self.cameras[image] = PerspectiveCamera(image, camera)

    def write_meta(self, fname):
        with open(fname, 'w') as fp:
            json.dump(self.meta, fp, indent=2)

    def utm_to_enu(self, points):
        # Convert points in UTM coordinates to ENU.
        lat, lon = latlon_utm_converter.eastnorth_to_latlon(points[:, 0:1],
                                                            points[:, 1:2],
                                                            self.utm_zone,
                                                            self.hemisphere)
        alt = points[:, 2:3]
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
    def __init__(self, ply_path, recon_path, local_texture_path):
        self.reconstruction = Reconstruction(recon_path)
        self.local_texture_path = local_texture_path

        # Remove previous texture images.
        # TODO: make this a safer operation by creating a temporary directory.
        if os.path.exists(self.local_texture_path):
            shutil.rmtree(self.local_texture_path)
        os.makedirs(self.local_texture_path)
        
        # Kai says: no longer needed
        # self.ply_data = PlyData.read(ply_path)
        # self.vertices = self.ply_data.elements[0]
        # self.faces = self.ply_data.elements[1]

        trimesh.tol.merge = 1.0e-3
        self.tmesh = trimesh.load(ply_path)
        print('num_vertices: {}'.format(np.shape(self.tmesh.vertices)))
        
        # Transform vertices from UTM to ENU.
        vertices_enu = self.reconstruction.utm_to_enu(self.tmesh.vertices)
        # print('tmesh.vertices_enu: {}'.format(vertices_enu[0:2, :]))
        self.tmesh.vertices = vertices_enu
        # self.tmesh.export('./mesh_enu.ply')

        # Recolor the facets.
        num_facets = self.tmesh.facets.size
        print('number of facets: {}'.format(num_facets))
        # facet_index = long(0)

        # TODO(snavely): Why are some facets showing up as gray? Are they
        # somehow facing the wrong direction? Do those faces not show up in the
        # list of facets?
        for facet_index, facet in enumerate(self.tmesh.facets):
            # Random trimesh colors have random hue but nearly full saturation
            # and value. Useful for visualization and debugging.

            # tmesh.visual.face_colors[facet] = trimesh.visual.random_color()
            r, g, b = self.color_index_to_color(facet_index)
            # Last 255 is for alpha channel (fully opaque).
            self.tmesh.visual.face_colors[facet] = np.array([r, g, b, 255],
                                                            dtype=np.uint8)
            # self.tmesh.visual.face_colors[facet] = np.array([20, 20, 0, 255], dtype=np.uint8)
            # facet_index = facet_index + 1
        
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
        # red is the lowest 8-bits, then green, then blue.
        color_indices = (
            color[:,:,0] + 256 * color[:,:,1] + 65536 * color[:,:,2])
        return color_indices

    def test_rendering(self):
        width = 2000
        height = 2000
        renderer = pyrender.OffscreenRenderer(width, height)
        test_camera = pyrender.IntrinsicsCamera(
            fx=866.0 * 1000.0, fy=866.0 * 1000.0, cx=1000.0, cy=0.0, #cy=1000.0,
            znear=1000.0, zfar=1.0e8)
        test_camera_pose = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0],
                                     [0, 0, 1, 1.0e6],
                                     [0, 0, 0, 1]])

        self.scene.add(test_camera, pose=test_camera_pose)
        t = time.time()
        color, depth = renderer.render(self.scene)
        elapsed = time.time() - t
        print('Time to render: {}'.format(elapsed))

        resize_and_save_color_buffer_to_png(color, 2048, 'test_render.png')
        resize_and_save_depth_buffer_to_png(depth, 2048, 'test_depth.png')

    def test_rendering_on_real_camera(self):
        image_name, camera = (self.reconstruction.cameras.items())[0]
        print('Rendering image {}'.format(image_name))

        color, depth = self.render_from_camera(camera)
        image = imageio.imread(image_name)

        self.create_local_texture(camera, 0, image)

        # resize_and_save_color_buffer_to_png(color, 1024,
        #                                     image_name + '_render.png')
        # resize_and_save_depth_buffer_to_png(depth, 1024,
        #                                     image_name + '_depth.png')

    # Render the loaded scene from the provided camera. Returns color and depth
    # buffers.
    def render_from_camera(self, camera):
        renderer = pyrender.OffscreenRenderer(camera.width, camera.height)

        node = self.scene.add(camera.pyrender_camera, pose=camera.pose)

        t = time.time()
        color, depth = renderer.render(self.scene)
        elapsed = time.time() - t
        print('Time to render: {}'.format(elapsed))

        self.scene.remove_node(node)

        return color, depth

    def create_textures(self):
        num_cameras = len(self.reconstruction.cameras)
        num_facets = self.tmesh.facets.size

        # Num cameras by num facets matrix counting the visibility of
        # each facet in each image
        facet_pixel_counts = np.zeros((num_cameras, num_facets),
                                      dtype=np.int16)

        camera_index = 0
        # for image, camera in self.reconstruction.cameras.items():
        facet_uv_coords = {}

        for image_name, camera in self.reconstruction.cameras.items()[0:1]:
            print('rendering image {}'.format(image_name))

            image = imageio.imread(image_name)
            color, depth = self.render_from_camera(camera)

            # Count number of times each color appears.
            color_indices = self.color_buffer_to_color_indices(color)
            elems, counts = np.unique(color_indices, return_counts=True)
            print('Visible facets: {}'.format(elems.size))

            for elem, count in zip(elems, counts):
                if elem >= 0 and elem < num_facets:
                    facet_index = elem
                    # print('Creating local texture for facet {}'.
                    #       format(facet_index))
                    facet_pixel_counts[camera_index, facet_index] = count
                    uv_coords = self.create_local_texture(
                        camera, facet_index, image)
                    facet_uv_coords[facet_index] = uv_coords
                else:
                    print('Skipping out-of-range facet_index {}'.format(elem))

            facet_bboxes = self.generate_texture_atlas('texture.png')

            for facet_index, bbox in facet_bboxes.items():
                # For each facet, apply the offset into the global texture map.
                # u is along the column axis, while v is along the row axis.
                facet_uv_coords[facet_index] = (
                    facet_uv_coords[facet_index] + np.array([bbox[0], bbox[1]]))

            print('Assigning per-face texture...')
            FullTextureMapper.write_textured_trimesh(self.tmesh, facet_uv_coords,
                                                     'texture.png',
                                                     'textured.ply')
            # Clean up local texture data.
            # TODO: make this a safer operation by creating a temporary directory.
            shutil.rmtree(self.local_texture_path)
            
            # Debugging output.
            # resize_and_save_color_buffer_to_png(color, 1024,
            #                                     image_name + '_render.png')
            # resize_and_save_depth_buffer_to_png(depth, 1024,
            #                                     image_name + '_depth.png')

    @staticmethod
    def write_textured_trimesh(trimesh_obj, facet_uv_coords, texture_img, out_ply):
        # load texture image
        img = imageio.imread(texture_img)
        img_height, img_width, _ = img.shape

        # build face-->facet mapping
        face2facet_mapping = []
        for facet_idx, facet in enumerate(trimesh_obj.facets):
            for idx, face_idx in enumerate(np.sort(facet)):
                face2facet_mapping.append([int(face_idx), (int(facet_idx), int(idx))])
        face2facet_mapping = dict(face2facet_mapping)
        
        # write to ply file
        with open(out_ply, 'w') as fp:
            # write header
            header = ('ply\nformat ascii 1.0\n'
                      'comment TextureFile {}\n'
                      'element vertex {}\n'
                      'property float x\n'
                      'property float y\n'
                      'property float z\n'
                      'element face {}\n'               
                      'property list uint8 int32 vertex_index\n'
                      'property list uint8 float texcoord\n'
                      'end_header\n')
            header = header.format(texture_img,
                                   len(trimesh_obj.vertices),
                                   len(trimesh_obj.faces))
            fp.write(header)

            # start writing vertices
            for idx, vertex in enumerate(trimesh_obj.vertices):
                vertex = np.float32(vertex)
                fp.write('{} {} {}\n'.format(vertex[0], vertex[1], vertex[2]))

            # start writing faces and textures
            cnt = 0
            for face_idx, face in enumerate(trimesh_obj.faces):
                if face_idx not in face2facet_mapping:
                    print('Face: {} has no corresponding facet'.format(face_idx))
                    uv_coords = np.zeros((3, 2))
                    cnt += 1
                else:
                    facet_idx, idx = face2facet_mapping[face_idx]

                    if facet_idx not in facet_uv_coords: 
                        # print('Face: {}, corresponding facet: {} has no texture'.format(
                        #     face_idx, facet_idx))
                        uv_coords = np.zeros((3, 2))
                        cnt += 1
                    else:
                        uv_coords = facet_uv_coords[facet_idx][idx*3:(idx+1)*3, :]

                face_str = '{}'.format(np.uint8(len(face)))
                for vertex_idx in face:
                    face_str += ' {}'.format(np.int32(vertex_idx))
                face_str += '\n'

                texcoord_str = '{}'.format(np.uint8(len(face) * 2))
                for uv in uv_coords:
                    # needs to convert the origin from upper-left to lower-left expected by OpenGL
                    # https://community.khronos.org/t/texture-mapping-upper-left-hand-origin/61068/5
                    u = np.float32(uv[0]) / img_width
                    v = 1.0 - np.float32(uv[1]) / img_height
                    texcoord_str += ' {} {}'.format(u, v)
                texcoord_str += '\n'

                fp.write(face_str + texcoord_str)

            print('{}/{} ({}%) faces untextured'.format(cnt, len(trimesh_obj.faces), 100.0 * float(cnt)/len(trimesh_obj.faces)))

    def create_local_texture(self, camera, facet_index, image,
                             max_side_length=256):
        # Gather the vertices for this facet.
        # vertices = self.tmesh.vertices[
        #     self.tmesh.faces[self.tmesh.facets[facet_index]]]
        # vertices_shape = np.shape(vertices)
        # assert vertices_shape[2] == 3
        # vertices = np.reshape(vertices,
        #                       (vertices_shape[0] * vertices_shape[1], 3))

        # @kai access the faces inside a facet with increasing face_idx
        #  access the vertices inside a face with increasing vertex_idx
        #  the same access order is used in write_textured_trimesh
        vertices = []
        for face_idx in np.sort(self.tmesh.facets[facet_index]):
            face = self.tmesh.faces[face_idx]
            if len(face) != 3:  # only collect triangular face
                continue
            for vertex_idx in face:
                vertex = self.tmesh.vertices[vertex_idx]
                vertices.append([vertex[0], vertex[1], vertex[2]])
        vertices = np.array(vertices)

        # Project the vertices into the image.
        projections = camera.project_ydown(vertices)

        # Clamp projections to image boundaries.
        # TODO(snavely): Consider more intelligent handling of
        # out-of-bounds issues.
        projections = np.clip(projections,
                              np.array([0.0, 0.0]),
                              np.array([camera.width-1, camera.height-1]))

        # Compute the bounding box in image space.
        # Compute 2D bounding box of projected uv coordinates.
        proj_bbox = np.stack((np.amin(projections, axis=0),
                              np.amax(projections, axis=0)))


        # Snap the bounding box to integer coordinates via floor and
        # ceiling operations.
        proj_bbox = np.array(
            [[np.floor(proj_bbox[0,0]), np.floor(proj_bbox[0,1])],
             [np.ceil(proj_bbox[1,0]), np.ceil(proj_bbox[1,1])]],
            dtype=np.int32)

        # TODO(snavely): optionally pad the bounding box to avoid
        # boundary effects.
        
        local_uv_coords = projections - proj_bbox[0,:]

        # Crop the image and save to a file.
        image_cropped = (
            image[proj_bbox[0,1]:proj_bbox[1,1]+1,
                  proj_bbox[0,0]:proj_bbox[1,0]+1])

        crop_height, crop_width = np.shape(image_cropped)

        # Check the size against the requested max size.
        resized_dims, scale = resized_image_dims_for_max_dim(
            crop_width, crop_height, max_side_length)

        if scale < 1.0:
            # Scale the cropped patch and the uv_coords accordingly.
            local_uv_coords = scale * local_uv_coords
            image_cropped = cv2.resize(image_cropped, dsize=resized_dims,
                                       interpolation=cv2.INTER_AREA)

        image_name = os.path.join(self.local_texture_path,
                                  'texture_%05d.png' % facet_index)
        png.from_array(image_cropped, 'L').save(image_name)

        return local_uv_coords


    # Generate a texture atlas by running the 'atlas' program, and returning a
    # bounding box for each facet's local texture within the atlas.
    def generate_texture_atlas(self, output_image_name):
        print('Generating texture atlas')
        atlas_bin='/phoenix/S2/snavely/code/atlas/atlas'
        subprocess.call(
            '{} {} -o {} -t {}'.format(atlas_bin,
                                       self.local_texture_path,
                                       output_image_name,
                                       '/tmp/atlas.txt'), shell=True)

        with open('/tmp/atlas.txt') as fp:
            atlas_lines = fp.readlines()

        facet_bboxes = {}
        for line in atlas_lines:
            fields = line.split()
            tex_name = fields[0]
            tex_name_fields = tex_name.split('_')
            facet_index = int(tex_name_fields[1])

            x = int(fields[1])
            y = int(fields[2])
            w = int(fields[3])
            h = int(fields[4])
            
            facet_bboxes[facet_index] = (x, y, w, h)

        # Clean up.
        os.remove('/tmp/atlas.txt')

        return facet_bboxes


    # Given an assignment from facets to images, create a texture map.
    # Returns a texture image and a list of uv-coordinates per vertex
    # per face.
    # 
    # Inputs:
    #   facet_assignments: array of length num_facets, containing
    #     string identifying image to be used for texturing.
    #
    # Outputs:
    #   image: texture atlas
    #   uv_coords: per-vertex per-face list of texture coordinates
    def create_textures_from_facet_assignments(facet_assignments):
        pass

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
        # print('ply_vertices: {}'.format(vertices_enu[0:2,:]))

        renderer = pyrender.OffscreenRenderer(1000, 1000)
        for image, camera in self.reconstruction.cameras.items():
            # print('{}'.format(camera.project(vertices_enu[0,:])))
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
    # Base path for the reconstruction (cameras and images) to be used
    # in texture mapping.
    recon_path = 'testdata'

    # Location of the ply file to be texture mapped.
    ply_path = 'testdata/aoi.ply'
    texture_mapper = FullTextureMapper(ply_path, recon_path, '/tmp/textures')

    # texture_mapper.test_rendering()
    # texture_mapper.test_rendering_on_real_camera()
    texture_mapper.create_textures()

    # texture_mapper.save('testdata/textured')


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
    test()
    # deploy()
