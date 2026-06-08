import torch
import os.path as osp
from pytorch3d.io import load_obj
from .base_renderer import BaseMeshRenderer
from utils.helper import face_vertices
from utils.graphics_utils import GS_BaseMeshRenderer

import numpy as np
from pytorch3d.renderer import ( RasterizationSettings, PointLights, MeshRenderer, 
                                MeshRasterizer, TexturesVertex, SoftPhongShader, 
                                look_at_view_transform, BlendParams, OrthographicCameras, AmbientLights)
from pytorch3d.structures import Meshes
from torch import nn
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix
class Renderer(BaseMeshRenderer):
    ''' visualizer
    '''

    def __init__(self, assets_dir, image_size=1024, device='cuda', focal_length=12):
        super().__init__(assets_dir, image_size, device, focal_length=focal_length)
        topology_path = osp.join(assets_dir, 'smplx_tex.obj')
        self.focal_length=focal_length
        verts, faces, aux = load_obj(topology_path)
        uvcoords = aux.verts_uvs[None, ...]      # (N, V, 2)
        uvfaces = faces.textures_idx[None, ...]  # (N, F, 3)
        faces = faces.verts_idx[None, ...]
        self.register_buffer('faces', faces)
        self.register_buffer('raw_uvcoords', uvcoords)

        # uv coords
        uvcoords = torch.cat(
            [uvcoords, uvcoords[:, :, 0:1]*0.+1.], -1)  # [bz, ntv, 3]
        uvcoords = uvcoords*2 - 1
        uvcoords[..., 1] = -uvcoords[..., 1]
        face_uvcoords = face_vertices(uvcoords, uvfaces)
        self.register_buffer('uvcoords', uvcoords)
        self.register_buffer('uvfaces', uvfaces)
        self.register_buffer('face_uvcoords', face_uvcoords)

        self.setup()
    def setup(self):
        # Use naive rasterization (bin_size=0) to avoid coarse-rasterization overflow
        raster_settings = RasterizationSettings(
            image_size=self.image_size,
            faces_per_pixel=1,
            cull_backfaces=True,
            perspective_correct=True,
            bin_size=0,
        )

        self.lights = PointLights(
            device=self.device,
            location=((0.0, 0.0, 5.0), ),
            ambient_color=((0.5, 0.5, 0.5), ),
            diffuse_color=((0.5, 0.5, 0.5), ),
            specular_color=((0.01, 0.01, 0.01), )
        )

        self.mesh_rasterizer = MeshRasterizer(raster_settings=raster_settings)
        self.debug_renderer = MeshRenderer(
            rasterizer=self.mesh_rasterizer,
            shader=SoftPhongShader(device=self.device, lights=self.lights)
        )

        R, T = look_at_view_transform(dist=10)
        self.principal_point = nn.Parameter(torch.zeros(1, 2).float().to(self.device))
        self.R = nn.Parameter(matrix_to_rotation_6d(R).to(self.device))
        self.t = nn.Parameter(T.to(self.device))

        self.debug_renderer.rasterizer.raster_settings.image_size = self.image_size

        self.cameras = OrthographicCameras(device=self.device, focal_length=1, R=rotation_6d_to_matrix(self.R), T=self.t,)

    def render_image(self, transformed_vertices, skin_color=[252, 224, 203], bg_color=[50, 50, 50]):   # (RGB) boy skin color: [252, 224, 203], girl skin color:  [254, 242, 240]
        B = transformed_vertices.shape[0]
      
        faces = self.faces.repeat(B, 1, 1)

        V = transformed_vertices.shape[1]

        points = transformed_vertices.clone()

        ## rasterizer near 0 far 100. move mesh so minz larger than 0
        points[:, :, 1:] = -points[:, :, 1:]

        if isinstance(skin_color, (list, tuple)):
            verts_rgb = torch.from_numpy(np.array(skin_color)/255.).cuda().float()[None, None, :].repeat(B, V, 1)
        else:
            verts_rgb = torch.from_numpy(skin_color).cuda().float()

        textures = TexturesVertex(verts_features=verts_rgb.cuda())
        meshes_world = Meshes(verts=[points[i] for i in range(B)], faces=[faces[i] for i in range(B)], textures=textures)

        blend = BlendParams(background_color=np.array(bg_color)/225.)

        fragments = self.mesh_rasterizer(meshes_world, cameras=self.cameras)
        rendering = self.debug_renderer.shader(fragments, meshes_world, cameras=self.cameras, blend_params=blend)

        return {'rendered_img': rendering[..., :3]}

    def forward(self, vertices, faces=None, landmarks={}, cameras=None, transform_matrix=None, focal_length=None, is_weak_cam=False, ret_image=True):
        if faces is None:
            faces = self.faces.squeeze(0)
        return super().forward(vertices, faces, landmarks, cameras, transform_matrix, focal_length, is_weak_cam, ret_image)
    
    
class Renderer2(GS_BaseMeshRenderer):
    def __init__(self, assets_dir, image_size=1024, device='cuda', focal_length=24):
        super().__init__( image_size, focal_length=focal_length,inverse_light=True)
        topology_path = osp.join(assets_dir, 'smplx_tex.obj')
        self.focal_length=focal_length
        verts, faces, aux = load_obj(topology_path)
        uvcoords = aux.verts_uvs[None, ...]      # (N, V, 2)
        uvfaces = faces.textures_idx[None, ...]  # (N, F, 3)
        faces = faces.verts_idx[None, ...]
        self.register_buffer('faces', faces)
        self.register_buffer('raw_uvcoords', uvcoords)

        # uv coords
        uvcoords = torch.cat(
            [uvcoords, uvcoords[:, :, 0:1]*0.+1.], -1)  # [bz, ntv, 3]
        uvcoords = uvcoords*2 - 1
        uvcoords[..., 1] = -uvcoords[..., 1]
        face_uvcoords = face_vertices(uvcoords, uvfaces)
        self.register_buffer('uvcoords', uvcoords)
        self.register_buffer('uvfaces', uvfaces)
        self.register_buffer('face_uvcoords', face_uvcoords)
        
    def forward(self, vertices, faces=None, landmarks={}, cameras=None, transform_matrix=None, focal_length=None, is_weak_cam=False, ret_image=True):
        if faces is None:
            faces = self.faces.squeeze(0)
        return super().forward(vertices, faces, landmarks, cameras, transform_matrix, focal_length, ret_image)