
import torch
import math
import numpy as np
from pytorch3d.structures import (
    Meshes, Pointclouds
)
from pytorch3d.renderer import (
    PerspectiveCameras,CamerasBase,
    look_at_view_transform, RasterizationSettings, 
    PointLights, TexturesVertex, BlendParams,TexturesUV,
    SoftPhongShader, MeshRasterizer, MeshRenderer,
)
from pytorch3d.transforms.transform3d import _broadcast_bmm
from pytorch3d.renderer.mesh.rasterizer import Fragments,rasterize_meshes
import lightning as L
import torch
import matplotlib.pyplot as plt
import trimesh

def show_image(image_tensor, name):
    # 转为 [H, W, C] 并归一化
    image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    # 保存为 PNG
    plt.imsave(f"test_images/{name}.png", image_np)

def get_view_matrix(R, t):
    device=R.device
    Rt = torch.cat((R, t.view(3,1)),1)
    b_row=torch.tensor([0, 0, 0, 1], dtype=torch.float32, device=device).view(1, 4)
    #torch.FloatTensor([0,0,0,1],device=device).view(1,4)
    view_matrix = torch.cat((Rt, b_row))
    return view_matrix

def get_batch_view_matrix(R, t):
    """
    R: [B, 3, 3] - batch of rotation matrices
    t: [B, 3] or [B, 1, 3] - batch of translation vectors
    Returns:
        view_matrix: [B, 4, 4]
    """
    device = R.device
    B = R.shape[0]
    # Ensure t is shape [B, 3, 1]
    if t.ndim == 2:
        t = t.unsqueeze(-1)  # [B, 3, 1]
    # Concatenate R and t → [B, 3, 4]
    Rt = torch.cat((R, t), dim=2)
    # Create [0, 0, 0, 1] row for each batch → [B, 1, 4]
    b_row = torch.tensor([0, 0, 0, 1], dtype=torch.float32, device=device).view(1, 1, 4).expand(B, 1, 4)
    # Concatenate to form full view matrix → [B, 4, 4]
    view_matrix = torch.cat((Rt, b_row), dim=1)

    return view_matrix


def get_proj_matrix( tanfov,device, z_near=0.01, z_far=100, z_sign=1.0,):

    tanHalfFovY = tanfov
    tanHalfFovX = tanfov

    top = tanHalfFovY * z_near
    bottom = -top
    right = tanHalfFovX * z_near
    left = -right
    z_sign = 1.0

    proj_matrix = torch.zeros(4, 4).float().to(device)
    proj_matrix[0, 0] = 2.0 * z_near / (right - left)
    proj_matrix[1, 1] = 2.0 * z_near / (top - bottom)
    proj_matrix[0, 2] = (right + left) / (right - left)
    proj_matrix[1, 2] = (top + bottom) / (top - bottom)
    proj_matrix[3, 2] = z_sign
    proj_matrix[2, 2] = z_sign * z_far / (z_far - z_near)
    proj_matrix[2, 3] = -(z_far * z_near) / (z_far - z_near)
    return proj_matrix

def get_full_proj_matrix(w2c_cam,tanfov):
    assert len(w2c_cam.shape)==2 
    view_matrix=get_view_matrix(w2c_cam[:3,:3],w2c_cam[:3,3]).transpose(0,1).contiguous()
    proj_matrix=get_proj_matrix(tanfov,device=w2c_cam.device,z_near=0.01, z_far=100, z_sign=1.0).transpose(0,1).contiguous()
    full_proj_matrix = (view_matrix.unsqueeze(0).bmm(proj_matrix.unsqueeze(0))).squeeze(0)#torch.mm(view_matrix, proj_matrix)
    
    return view_matrix,full_proj_matrix


def get_batch_full_proj_matrix(w2c_cam, tanfov):
    """
    Args:
        w2c_cam: [B, 4, 4] - world-to-camera matrix
        tanfov: float or [B] - tangent of half FOV
    Returns:
        view_matrix: [B, 4, 4]
        full_proj_matrix: [B, 4, 4]
    """
    device = w2c_cam.device
    B = w2c_cam.shape[0]

    # Get view matrix from R and t
    R = w2c_cam[:, :3, :3]
    t = w2c_cam[:, :3, 3]
    view_matrix = get_batch_view_matrix(R, t)  # [B, 4, 4]

    # Get projection matrix per batch
    proj_matrix = get_proj_matrix(
        tanfov, device=device, z_near=0.01, z_far=100.0, z_sign=1.0
    )  # Should return [B, 4, 4]

    # Ensure shape: [B, 4, 4]
    if proj_matrix.ndim == 2:  # single matrix
        proj_matrix = proj_matrix.unsqueeze(0).expand(B, -1, -1)

    # Matrix multiplication in batch
    full_proj_matrix = proj_matrix.bmm(view_matrix)

    return view_matrix, full_proj_matrix



def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x*y, -1, keepdim=True)

def length(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return torch.sqrt(torch.clamp(dot(x,x), min=eps)) # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN

def safe_normalize(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return x / length(x, eps)
# 计算三角面片的局部坐标系（方向/姿态），即每个三角面片的三个正交轴向量
def compute_face_orientation(verts, faces, return_scale=False):
    i0 = faces[..., 0].long()
    i1 = faces[..., 1].long()
    i2 = faces[..., 2].long()

    v0 = verts[..., i0, :]
    v1 = verts[..., i1, :]
    v2 = verts[..., i2, :]

    a0 = safe_normalize(v1 - v0)
    a1 = safe_normalize(torch.cross(a0, v2 - v0, dim=-1))
    a2 = -safe_normalize(torch.cross(a1, a0, dim=-1)) 

    orientation = torch.cat([a0[..., None], a1[..., None], a2[..., None]], dim=-1)

    if return_scale:
        s0 = length(v1 - v0)
        s1 = dot(a2, (v2 - v0)).abs()
        scale = (s0 + s1) / 2
    return orientation, scale

class VertexPositionShader(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, fragments, meshes, **kwargs):
        """
        :param fragments: Fragments of the meshes that are rasterized.
        :param meshes: Meshes to render.
        :param kwargs: Additional arguments passed by the renderer.
        :return: The output colors, which in this case will be the vertex positions.
        """
        pixel_positions = fragments.pix_to_face  # shape (num_pixels, 3)
        batch_size, H, W = pixel_positions.shape[0], pixel_positions.shape[1], pixel_positions.shape[2]
        bary_coords=fragments.bary_coords.squeeze(-2)
        
        alpha = (pixel_positions!=-1)*1.0
        vertex_faces = meshes.faces_packed()[pixel_positions.squeeze(-1)]#  # shape (num_pixels, 3)
        vertex_positions=(meshes.verts_packed()[vertex_faces]*bary_coords[...,None]).sum(dim=-2)#.mean(dim=-2)
        results=torch.cat([vertex_positions,alpha],dim=-1)
        extra_result={"vertex_faces":vertex_faces,"bary_coords":bary_coords}
        return results,extra_result

class GS_Camera(CamerasBase):
    #still obey pytorch 3d coordinate system, adapting to gaussian splatting's projection method 
    def __init__(
        self,
        focal_length=1.0,
        R: torch.Tensor = torch.eye(3)[None],
        T: torch.Tensor = torch.zeros(1, 3),
        principal_point=((0.0, 0.0),),#useless
        device = "cpu",
        in_ndc: bool = True,
        image_size = None,
    ) -> None:

        kwargs = {"image_size": image_size} if image_size is not None else {}
        super().__init__(
            device=device,
            focal_length=focal_length,
            principal_point=((0.0, 0.0),),
            R=R,
            T=T,
            K=None,
            _in_ndc=in_ndc,
            **kwargs,  # pyre-ignore
        )
        if image_size is not None:
            if (self.image_size < 1).any():  # pyre-ignore
                raise ValueError("Image_size provided has invalid values")
        else:
            self.image_size = None

        # When focal length is provided as one value, expand to
        # create (N, 2) shape tensor
        
        if self.focal_length.ndim == 1:  # (N,)
            self.focal_length = self.focal_length[:, None]  # (N, 1)
        self.focal_length = self.focal_length.expand(-1, 2)  # (N, 2)
        self.proj_mats=None

        
    def transform_points_to_view(self, points, eps = None, **kwargs):
        #from wold to view
        R: torch.Tensor = kwargs.get("R", self.R)
        T: torch.Tensor = kwargs.get("T", self.T)
        self.R = R
        self.T = T
        if R.dim() == 2 :
            Tmat=torch.eye(4,device=R.device)[None]
            Tmat[:,:3,:3] = R
            Tmat[:,:3,3] = T
        else:
            
            Tmat=torch.eye(4,device=R.device)[None].repeat(R.shape[0],1,1)
            Tmat[:,:3,:3] = R
            Tmat[:,:3,3] = T
            
        points_batch = points.clone()
        if points_batch.dim() == 2:
            points_batch = points_batch[None]  # (P, 3) -> (1, P, 3)
        if points_batch.dim() != 3:
            msg = "Expected points to have dim = 2 or dim = 3: got shape %r"
            raise ValueError(msg % repr(points.shape))
        N, P, _3 = points_batch.shape
        ones = torch.ones(N, P, 1, dtype=points.dtype, device=points.device)
        points_batch = torch.cat([points_batch, ones], dim=2)
        # points_out=_broadcast_bmm(points_batch,Tmat)
        points_out=torch.einsum('bij,bnj->bni',Tmat,points_batch)
        return points_out[:,:,:3]
    
    def get_projection_transform(self,device):
        if self.proj_mats is None:  # 构造内参，中心点为 00， focal 为 24
            proj_mats=[]
            if  torch.unique(self.focal_length).numel()==1:
                invtanfov=self.focal_length[0,0]
                proj_mat=get_proj_matrix(1/invtanfov,device)
                proj_mats=proj_mat[None].repeat(self.focal_length.shape[0],1,1)
            else:
                for invtanfov in self.focal_length:
                    invtanfov=invtanfov[0]; assert invtanfov[0]==invtanfov[1]
                    proj_mat=get_proj_matrix(1/invtanfov,device)
                    proj_mats.append(proj_mat[None])
                proj_mats=torch.cat(proj_mats,dim=0)
            self.proj_mats=proj_mats
        else:
            proj_mats=self.proj_mats
        return proj_mats
    
    def transform_points_to_ndc(self, points, eps = None, **kwargs):
        #from wold to ndc
        R: torch.Tensor = kwargs.get("R", self.R)
        T: torch.Tensor = kwargs.get("T", self.T)
        self.R = R
        self.T = T
        if R.dim() == 2 :
            Tmat=torch.eye(4,device=R.device)[None]
            Tmat[:,:3,:3] = R.clone()
            Tmat[:,:3,3] = T.clone()
        else:
            
            Tmat=torch.eye(4,device=R.device)[None].repeat(R.shape[0],1,1) # true
            Tmat[:,:3,:3] = R.clone()
            Tmat[:,:3,3] = T.clone()
            
        #points_view=self.transform_points_to_view(points)
        N, P, _3 = points.shape
        ones = torch.ones(N, P, 1, dtype=points.dtype, device=points.device)
        points_h = torch.cat([points, ones], dim=2)
        proj_mat=self.get_projection_transform(points.device)#[None].expand(N,-1,-1)
        proj_mat=proj_mat.to(R.device)
        
        B = Tmat.shape[0]
        if B > self.focal_length.shape[0]:  # 第二阶段使用的，concat在一起
            proj_mat = proj_mat.repeat(2,1,1)
            
        full_mat=torch.bmm(proj_mat[:Tmat.shape[0]],Tmat)  # 防止最后一个batch
        points_ndc=torch.einsum('bij,bnj->bni',full_mat,points_h)

        points_ndc_xyz=points_ndc[:,:,:3]/(points_ndc[:,:,3:]+1e-7)
        # points_ndc_xyz[:,:,2]=points_ndc[:,:,3] #retain z range
        return points_ndc_xyz
    
    def transform_points_view_to_ndc(self, points, eps = None, **kwargs):
        #from view to ndc
        points_view=points.clone()
        N, P, _3 = points_view.shape
        ones = torch.ones(N, P, 1, dtype=points.dtype, device=points_view.device)
        points_view = torch.cat([points_view, ones], dim=2)
        
        proj_mat=self.get_projection_transform(points.device)#[None].expand(N,-1,-1)
        # points_ndc=_broadcast_bmm(points_view,proj_mat)
        points_ndc=torch.einsum('bij,bnj->bni',proj_mat,points_view)
        points_ndc_xyz=points_ndc[:,:,:3]/(points_ndc[:,:,3:]+1e-7)
        
        return points_ndc_xyz
    
    def transform_points_to_screen(self, points, with_xyflip = True, **kwargs):
        #from wold to screen
        'with_xyflip: obey pytroch 3d coordinate'
        R: torch.Tensor = kwargs.get("R", self.R)
        T: torch.Tensor = kwargs.get("T", self.T)
        self.R = R
        self.T = T
        
        points_ndc=self.transform_points_to_ndc(points)
        N, P, _3 = points_ndc.shape
        image_size=self.image_size[:N]
        if not torch.is_tensor(image_size):
            image_size = torch.tensor(image_size, device=R.device)
        if image_size.dim()==2:
            image_size = image_size[:,None]
        image_size=image_size[:,:,[1,0]]#width height
        
        points_screen=points_ndc.clone()
        # points_ndc[...,:2] 以 (0,0) 为 光心
        # points_screen[...,:2]=points_ndc[...,:2] * image_size/2-image_size/2  # x,y  in  [-1024,0]
        points_screen[...,:2]= (points_ndc[...,:2] - 1)* image_size/2   # x,y  in  [-1024,0]
        if with_xyflip:  # true
            points_screen[...,:2]=points_screen[:,:,:2]*-1  # 转化到 [0,1024]，似乎可以前面+1，这里就不用取反了
        
        return points_screen
    
    def transform_points_screen(self, points, with_xyflip = True, **kwargs):
        return self.transform_points_to_screen(points, with_xyflip, **kwargs)
    

    def perspective_projection(self, points, with_xyflip = True, **kwargs):
        
        '''
        Computes the perspective projection of a set of 3D points.
        https://github.com/shubham-goel/4D-Humans/blob/6ec79656a23c33237c724742ca2a0ec00b398b53/hmr2/utils/geometry.py#L64-L102
        '''
        # from wold to screen
        'with_xyflip: obey pytroch 3d coordinate'
        rotation: torch.Tensor = kwargs.get("R", self.R)
        translation: torch.Tensor = kwargs.get("T", self.T)

        B = points.shape[0]
        if rotation is None:
            rotation = torch.tensor([
                [-1.0,  0.0,  0.0],
                [ 0.0, -1.0,  0.0],
                [ 0.0,  0.0,  1.0]
            ], device=points.device, dtype=points.dtype).unsqueeze(0).expand(B, -1, -1)

        # rotation = torch.eye(3, device=points.device, dtype=points.dtype).unsqueeze(0).expand(B, -1, -1)
        camera_center = torch.zeros(B, 2, device=points.device, dtype=points.dtype)
        # Populate intrinsic camera matrix K.
        K = torch.zeros([B, 3, 3], device=points.device, dtype=points.dtype)
        K[:,   0,  0] = 24
        K[:,   1,  1] = 24
        K[:,   2,  2] = 1.
        K[:, :-1, -1] = camera_center

        # Transform points
        points = torch.einsum('bij, bkj -> bki', rotation, points)
        points = points + translation.unsqueeze(1)

        # Apply perspective distortion
        projected_points = points / points[:, :, -1].unsqueeze(-1)

        # Apply camera intrinsics
        projected_points = torch.einsum('bij, bkj -> bki', K, projected_points)


        points_screen = projected_points.clone()
        # points_ndc[...,:2] 以 (0,0) 为 光心
        # points_screen[...,:2]=points_ndc[...,:2] * image_size/2-image_size/2  # x,y  in  [-1024,0]
        points_screen[...,:2]= (projected_points[...,:2] - 1) * 1024 / 2   # x,y  in  [-1024,0]
        if with_xyflip:  # true
            points_screen[...,:2] = points_screen[:,:,:2]*-1  # 转化到 [0,1024]，似乎可以前面+1，这里就不用取反了

        return points_screen # [B,N,3]
        
class GS_MeshRasterizer(MeshRasterizer):
    """
    adapted to GS_camera
    This class implements methods for rasterizing a batch of heterogeneous
    Meshes.
    """

    def __init__(self, cameras:GS_Camera=None, raster_settings=None) -> None:
        """
        Args:
            cameras: A cameras object which has a  `transform_points` method
                which returns the transformed points after applying the
                world-to-view and view-to-ndc transformations.
            raster_settings: the parameters for rasterization. This should be a
                named tuple.

        All these initial settings can be overridden by passing keyword
        arguments to the forward function.
        """
        super().__init__()
        if raster_settings is None:
            # default to naive rasterization to avoid coarse-rasterization overflow
            raster_settings = RasterizationSettings(bin_size=0)

        self.cameras = cameras
        self.raster_settings = raster_settings

    def to(self, device):
        # Manually move to device cameras as it is not a subclass of nn.Module
        if self.cameras is not None:
            self.cameras = self.cameras.to(device)
        return self

    def transform(self, meshes_world, **kwargs) -> torch.Tensor:
        #adapted to GS_camera
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of MeshRasterizer"
            raise ValueError(msg)

        n_cameras = len(cameras)
        if n_cameras != 1 and n_cameras != len(meshes_world):
            msg = "Wrong number (%r) of cameras for %r meshes"
            raise ValueError(msg % (n_cameras, len(meshes_world)))

        verts_world = meshes_world.verts_padded()

        # NOTE: Retaining view space z coordinate for now.
        # TODO: Revisit whether or not to transform z coordinate to [-1, 1] or
        # [0, 1] range.
        eps = kwargs.get("eps", None)
        verts_view = cameras.transform_points_to_view(verts_world, eps=eps,**kwargs)
        verts_ndc =  cameras.transform_points_view_to_ndc(verts_view, eps=eps,**kwargs)

        verts_ndc[..., 2] = verts_view[..., 2]
        meshes_ndc = meshes_world.update_padded(new_verts_padded=verts_ndc)
        return meshes_ndc

    def forward(self, meshes_world, **kwargs) -> Fragments:
        """
        Args:
            meshes_world: a Meshes object representing a batch of meshes with
                          coordinates in world space.
        Returns:
            Fragments: Rasterization outputs as a named tuple.
        """
        meshes_proj = self.transform(meshes_world, **kwargs)
        raster_settings = kwargs.get("raster_settings", self.raster_settings)

        # By default, turn on clip_barycentric_coords if blur_radius > 0.
        # When blur_radius > 0, a face can be matched to a pixel that is outside the
        # face, resulting in negative barycentric coordinates.
        clip_barycentric_coords = raster_settings.clip_barycentric_coords
        if clip_barycentric_coords is None:
            clip_barycentric_coords = raster_settings.blur_radius > 0.0

        # If not specified, infer perspective_correct and z_clip_value from the camera
        cameras = kwargs.get("cameras", self.cameras)
        perspective_correct=False 
        z_clip = None 
        if raster_settings.perspective_correct is not None: 
            perspective_correct = raster_settings.perspective_correct 
        else:
            perspective_correct = True 
        # if raster_settings.z_clip_value is not None:
        #     z_clip = raster_settings.z_clip_value
        # else:
        #     znear = cameras.get_znear()
        #     if isinstance(znear, torch.Tensor):
        #         znear = znear.min().item()
        #     z_clip = None if not perspective_correct or znear is None else znear / 2

        # By default, turn on clip_barycentric_coords if blur_radius > 0.
        # When blur_radius > 0, a face can be matched to a pixel that is outside the
        # face, resulting in negative barycentric coordinates.

        pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(
            meshes_proj,
            image_size=raster_settings.image_size,  # 512
            blur_radius=raster_settings.blur_radius,  # 0
            faces_per_pixel=raster_settings.faces_per_pixel,  # 1
            bin_size=raster_settings.bin_size, # None  Bin size was too small in the coarse rasterization phase.
            max_faces_per_bin=raster_settings.max_faces_per_bin, # None 
            clip_barycentric_coords=clip_barycentric_coords,  # False
            perspective_correct=perspective_correct,  # True
            cull_backfaces=raster_settings.cull_backfaces,  # false
            z_clip_value=z_clip,  # None
            cull_to_frustum=raster_settings.cull_to_frustum, # flase
        )

        return Fragments(
            pix_to_face=pix_to_face, # 	每个像素最近的面片索引（-1 表示无）
            zbuf=zbuf,  # 	每个像素对应的深度值
            bary_coords=bary_coords,  # 每个像素相对于三角形的重心坐标
            dists=dists,  # 像素距离面片的距离（用于模糊/soft raster）
        )


class BaseMeshRenderer(L.LightningModule):
    def __init__(self, faces,image_size=512,lbs_weights=None, skin_color=[252, 224, 203], bg_color=[0, 0, 0], 
                 faces_uvs=None,verts_uvs=None,focal_length=24,inverse_light=False):
        super(BaseMeshRenderer, self).__init__()
        
        self.image_size = image_size

        self.skin_color = np.array(skin_color)
        self.bg_color = bg_color
        self.focal_length = focal_length
        # use naive rasterization to avoid coarse-rasterization overflow in some PyTorch3D builds
        # set bin_size=0 to force naive rasterization; this avoids "Bin size was too small" warnings
        bin_size = 0
        self.raster_settings = RasterizationSettings(image_size=image_size, blur_radius=0.0, faces_per_pixel=1,
                                bin_size=bin_size)
        if inverse_light:
            self.lights = PointLights( location=[[0.0, -1.0, -10.0]])
        else:
            self.lights = PointLights( location=[[0.0, 1.0, 10.0]])
        self.manual_lights = PointLights(
            location=((0.0, 0.0, 5.0), ),
            ambient_color=((0.5, 0.5, 0.5), ),
            diffuse_color=((0.5, 0.5, 0.5), ),
            specular_color=((0.01, 0.01, 0.01), )
        )
        self.blend = BlendParams(background_color=np.array(bg_color)/225.)
        self.faces = torch.nn.Parameter(faces, requires_grad=False)
        if faces_uvs is not None:
            self.faces_uvs = torch.nn.Parameter(faces_uvs, requires_grad=False)
        if verts_uvs is not None:
            self.verts_uvs = torch.nn.Parameter(verts_uvs.clone(), requires_grad=False)
            #SoftPhongShader will flip the v, so we need to flip it back
            self.verts_uvs[:,1]=1-self.verts_uvs[:,1]
        self.lbs_weights = None
        if lbs_weights is not None: self.lbs_weights = torch.nn.Parameter(lbs_weights, requires_grad=False)
        
    def _build_cameras(self, transform_matrix, focal_length):
        device = transform_matrix.device    
        batch_size = transform_matrix.shape[0]
        screen_size = torch.tensor(
            [self.image_size, self.image_size], device=device
        ).float()[None].repeat(batch_size, 1)
        cameras_kwargs = {
            'principal_point': torch.zeros(batch_size, 2, device=device).float(), 'focal_length': focal_length, 
            'image_size': screen_size, 'device': device,
        }
        #cameras = PerspectiveCameras(**cameras_kwargs, R=transform_matrix[:, :3, :3], T=transform_matrix[:, :3, 3])
        cameras = GS_Camera(**cameras_kwargs, R=transform_matrix[:, :3, :3], T=transform_matrix[:, :3, 3])
        return cameras
    
    def forward(self, vertices, faces=None, landmarks={}, cameras=None, transform_matrix=None, focal_length=None, ret_image=True):
        B, V = vertices.shape[:2]
        focal_length = self.focal_length if focal_length is None else focal_length
        if isinstance(cameras, torch.Tensor):
            cameras = cameras.clone()
        elif cameras is None:
            cameras = self._build_cameras(transform_matrix, focal_length)
        
        t_faces = faces[None].repeat(B, 1, 1)
        
        ret_vertices = cameras.transform_points_screen(vertices)
        ret_landmarks = {k: cameras.transform_points_screen(v) for k,v in landmarks.items()}

        images = None
        if ret_image:
            # Initialize each vertex to be white in color.
            verts_rgb = torch.from_numpy(self.skin_color/255).float().to(self.device)[None, None, :].repeat(B, V, 1)
            textures = TexturesVertex(verts_features=verts_rgb)
            mesh = Meshes(
                verts=vertices.to(self.device),
                faces=t_faces.to(self.device),
                textures=textures
            )
            renderer = MeshRenderer(#GS_MeshRasterizer MeshRasterizer
                rasterizer=GS_MeshRasterizer(cameras=cameras, raster_settings=self.raster_settings),
                shader=SoftPhongShader(cameras=cameras, lights=self.lights.to(vertices.device), device=self.device, blend_params=self.blend)
            )
            render_results = renderer(mesh).permute(0, 3, 1, 2)
            images = render_results[:, :3]
            alpha_images = render_results[:, 3:]
            images[alpha_images.expand(-1, 3, -1, -1)<0.5] = 0.0
            images = images * 255
        
        return ret_vertices, ret_landmarks, images

    def render_mesh(self, vertices,cameras=None,transform_matrix=None, faces=None,lights=None,reverse_camera=True):
        #render mesh vertices value and lbs weights
        device = vertices.device
        B, V = vertices.shape[:2]
        
        if faces is None:
            faces = self.faces
        if cameras is None:
            transform_matrix=transform_matrix.clone()
            if reverse_camera:
                tf_mat=torch.tensor([[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]],dtype=torch.float32).to(device)
                tf_mat=tf_mat[None].expand(B,-1,-1)
                transform_matrix = torch.bmm(tf_mat,transform_matrix)
            cameras = self._build_cameras(transform_matrix, self.focal_length)
        t_faces = faces[None].repeat(B, 1, 1)
        # self.lights=lights
        # if lights is None:
        #     self.lights = PointLights(device=device, location=[[0.0, 1.0, 10.0]])
        
        # Initialize each vertex to be white in color.
        # verts_rgb = vertices.clone()
        # textures = TexturesVertex(verts_features=verts_rgb)
        # mesh = Meshes(
        #     verts=vertices.to(device),
        #     faces=t_faces.to(device),
        #     textures=textures
        # )

        # renderer = MeshRenderer(
        #     rasterizer=MeshRasterizer(cameras=cameras, raster_settings=self.raster_settings),
        #     shader=SoftPhongShader(cameras=cameras, lights=self.lights, device=device, blend_params=self.blend)
        # )
        mesh = Meshes(
            verts=vertices.to(device),
            faces=t_faces.to(device),
        )
        shader = VertexPositionShader().to(device)
        rasterizer=GS_MeshRasterizer(cameras=cameras, raster_settings=self.raster_settings)
        #GS_MeshRasterizer MeshRasterizer
        renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)
        render_results,extra_result = renderer(mesh)
        render_lbs_weights=None
        if self.lbs_weights is not None:
            vertex_faces=extra_result['vertex_faces']
            bary_coords=extra_result['bary_coords']
            lbs_weights=self.lbs_weights[None].expand(B, -1, -1).reshape(-1,55) 
            render_lbs_weights=(lbs_weights[vertex_faces]*bary_coords[...,None]).sum(dim=-2)
        # images = render_results[:, :3]
        # alpha_images = render_results[:, 3:]
        return render_results,render_lbs_weights

    def render_fragments(self, vertices,cameras=None,transform_matrix=None, faces=None,reverse_camera=True):
        device = vertices.device
        B, V = vertices.shape[:2]
        
        if faces is None:
            faces = self.faces
        if cameras is None:
            transform_matrix=transform_matrix.clone()
            if reverse_camera:
                tf_mat=torch.tensor([[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]],dtype=torch.float32).to(device)
                tf_mat=tf_mat[None].expand(B,-1,-1)
                transform_matrix = torch.bmm(tf_mat,transform_matrix)
            cameras = self._build_cameras(transform_matrix, self.focal_length)
        t_faces = faces[None].repeat(B, 1, 1)

        mesh = Meshes( # 顶点和面构成 mesh
            verts=vertices.to(device),
            faces=t_faces.to(device),
        )
        rasterizer=GS_MeshRasterizer(cameras=cameras, raster_settings=self.raster_settings)
        fragments = rasterizer(mesh) # 把 mesh 渲染到平面
        
        #return visble faces
        return fragments.pix_to_face,fragments
    
    def render_textured_mesh(self,vertices,uvmap,fragments=None,faces_uvs=None,verts_uvs=None,faces=None,cameras=None,transform_matrix=None,reverse_camera=True):
        device = vertices.device
        B, V = vertices.shape[:2]
        if faces is None:
            faces = self.faces
        if faces_uvs is None:
            faces_uvs = self.faces_uvs
        if verts_uvs is None:
            verts_uvs = self.verts_uvs
            
        if cameras is None:
            transform_matrix=transform_matrix.clone()
            if reverse_camera:
                tf_mat=torch.tensor([[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]],dtype=torch.float32).to(device)
                tf_mat=tf_mat[None].expand(B,-1,-1)
                transform_matrix = torch.bmm(tf_mat,transform_matrix)
            cameras = self._build_cameras(transform_matrix, self.focal_length)
        
        t_faces = faces[None].repeat(B, 1, 1)
        t_faces_uvs = faces_uvs[None].repeat(B, 1, 1)
        t_verts_uvs = verts_uvs[None].repeat(B, 1, 1)
        
        textures = TexturesUV(maps=uvmap,faces_uvs=t_faces_uvs,verts_uvs=t_verts_uvs)
        mesh = Meshes(
            verts=vertices.to(self.device),
            faces=t_faces.to(self.device),
            textures=textures
        )
        lights = PointLights( location=[[0.0, 0.0, 1000.0]])
        if fragments is None:
            rasterizer=GS_MeshRasterizer(cameras=cameras, raster_settings=self.raster_settings)
            fragments = rasterizer(mesh)
        shader=SoftPhongShader(cameras=cameras, lights=lights, device=device, blend_params=self.blend).to(device)
        images=shader(fragments, mesh)

        return images


class GS_BaseMeshRenderer(torch.nn.Module):
    #RENDERING IN GS PROJECTION METHOD
    def __init__(self,image_size=512, skin_color=[252, 224, 203], bg_color=[0, 0, 0], focal_length=24,inverse_light=False):
        super(GS_BaseMeshRenderer, self).__init__()
        
        self.image_size = image_size

        self.skin_color = np.array(skin_color)
        self.bg_color = bg_color
        self.focal_length = focal_length

        # default to naive rasterization to avoid coarse-rasterization overflow
        self.raster_settings = RasterizationSettings(image_size=image_size, blur_radius=0.0, faces_per_pixel=1, bin_size=0)
        if inverse_light:
            self.lights = PointLights( location=[[0.0, -1.0, -10.0]])
        else:
             self.lights = PointLights( location=[[0.0, 1.0, 10.0]])
             
        self.manual_lights = PointLights(
            location=((0.0, 0.0, 5.0), ),
            ambient_color=((0.5, 0.5, 0.5), ),
            diffuse_color=((0.5, 0.5, 0.5), ),
            specular_color=((0.01, 0.01, 0.01), )
        )
        self.blend = BlendParams(background_color=np.array(bg_color)/225.)
        # self.faces = torch.nn.Parameter(faces, requires_grad=False)
        # self.faces=None
        self.head_color=np.array([236,248,254])
        #np.array([222,235,247])
        
    def _build_cameras(self, transform_matrix, focal_length):
        device = transform_matrix.device    
        batch_size = transform_matrix.shape[0]
        screen_size = torch.tensor(
            [self.image_size, self.image_size], device=device
        ).float()[None].repeat(batch_size, 1)
        cameras_kwargs = {
            'principal_point': torch.zeros(batch_size, 2, device=device).float(), 'focal_length': focal_length, 
            'image_size': screen_size, 'device': device,
        }
        #cameras = PerspectiveCameras(**cameras_kwargs, R=transform_matrix[:, :3, :3], T=transform_matrix[:, :3, 3])
        cameras = GS_Camera(**cameras_kwargs, R=transform_matrix[:, :3, :3], T=transform_matrix[:, :3, 3])
        return cameras
    
    def forward(self, vertices, faces=None, landmarks={}, cameras=None, transform_matrix=None, focal_length=None, ret_image=True):
        B, V = vertices.shape[:2]
        device=vertices.device
        focal_length = self.focal_length if focal_length is None else focal_length
        if isinstance(cameras, torch.Tensor):
            cameras = cameras.clone()
        elif cameras is None:
            cameras = self._build_cameras(transform_matrix, focal_length) # 24
        
        t_faces = faces[None].repeat(B, 1, 1)
        
        ret_vertices = cameras.transform_points_screen(vertices)
        ret_landmarks = {k: cameras.transform_points_screen(v) for k,v in landmarks.items()}
        images = None
        
        if ret_image:
            # Initialize each vertex to be white in color.
            verts_rgb = torch.from_numpy(self.skin_color/255).float().to(self.device)[None, None, :].repeat(B, V, 1)
            textures = TexturesVertex(verts_features=verts_rgb)
            mesh = Meshes(
                verts=vertices.to(self.device),
                faces=t_faces.to(self.device),
                textures=textures
            )
            renderer = MeshRenderer(#GS_MeshRasterizer MeshRasterizer
                rasterizer=GS_MeshRasterizer(cameras=cameras, raster_settings=self.raster_settings),
                shader=SoftPhongShader(cameras=cameras, lights=self.lights.to(device), device=device, blend_params=self.blend)
            )
            render_results = renderer(mesh).permute(0, 3, 1, 2)
            images = render_results[:, :3]
            alpha_images = render_results[:, 3:]
            images[alpha_images.expand(-1, 3, -1, -1)<0.5] = 0.0
            images = images * 255
        
        return ret_vertices, ret_landmarks, images

    def render_mesh(self, vertices,cameras=None,transform_matrix=None, faces=None,lights=None,skin_color=None,smplx2flame_ind=None):
        device = vertices.device
        B, V = vertices.shape[:2]
        
        if faces is None:
            faces = self.faces
            assert faces is not None
        if cameras is None:
            transform_matrix=transform_matrix.clone()
            cameras = self._build_cameras(transform_matrix, self.focal_length)
                
        
        if lights is None:
            self.lights = self.lights
        else:
            self.lights=lights
            
        if faces.dim() == 2:
            faces = faces[None]
        t_faces = faces.repeat(B, 1, 1)
        # Initialize each vertex to be white in color.
        if skin_color is None:
            skin_color=self.skin_color
        if isinstance(skin_color, (list, tuple)):
            verts_rgb = torch.from_numpy(np.array(skin_color)/255.).to(vertices.device).float()[None, None, :].repeat(B, V, 1)
        else:
            verts_rgb = torch.from_numpy(skin_color/255).to(vertices.device).float()[None, None, :].repeat(B, V, 1) # here
            
        if smplx2flame_ind is not None:
            head_rgb = torch.from_numpy(self.head_color/255).to(vertices.device).float()[None, None, :].repeat(B, V, 1)
            verts_rgb[:,smplx2flame_ind] = head_rgb[:,smplx2flame_ind]
            
        textures = TexturesVertex(verts_features=verts_rgb)
        mesh = Meshes(
            verts=vertices.to(device),
            faces=t_faces.to(device),
            textures=textures
        )



        # rot = torch.tensor(
        #     trimesh.transformations.rotation_matrix(
        #         np.radians(-60), [0, 1, 0]
        #     )[:3, :3], dtype=torch.float32, device=device
        # )
        # verts_rot = torch.matmul(mesh.verts_packed(), rot.T)[None]
        # # 替换顶点
        # mesh = Meshes(
        #     verts=verts_rot,
        #     faces=t_faces.to(device),
        #     textures=textures
        # )


        renderer = MeshRenderer(  #   GS_MeshRasterizer MeshRasterizer
            rasterizer=GS_MeshRasterizer(cameras=cameras, raster_settings=self.raster_settings),
            shader=SoftPhongShader(cameras=cameras, lights=self.lights.to(device), device=device, blend_params=self.blend)
        )

        render_results = renderer(mesh).permute(0, 3, 1, 2)
        images = render_results[:, :3]
        alpha_images = render_results[:, 3:]
        alpha = alpha_images.expand(-1, 3, -1, -1) < 0.5
        # White background instead of black: set background pixels to 1.0
        images[alpha] = 1.0
        images = torch.cat([images, 1 - alpha[:, :1] * 1.0], dim=1)
        images = images * 255
        
        return images


if __name__=="__main__":
    pass
    import pdb;pdb.set_trace()