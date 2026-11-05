"""Project the original target's world region without relabeling another object."""

import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np


@lru_cache(maxsize=16)
def source_region(source_bddl, target):
    from libero.libero.envs.bddl_utils import robosuite_parse_problem
    parsed=robosuite_parse_problem(source_bddl)
    placement=next(state for state in parsed["initial_state"] if state[0].lower()=="on" and state[1]==target)
    region=placement[2]
    definition=parsed["regions"][region]
    if definition["target"]!="floor" or len(definition["ranges"])!=1:
        raise ValueError("Only a single native floor placement region is supported")
    return {"name":region,"xy_bounds":definition["ranges"][0],"source_bddl":source_bddl,
            "source_bddl_sha256":hashlib.sha256(Path(source_bddl).read_bytes()).hexdigest(),"target":target}


def convex_hull(points):
    points=sorted(set(map(tuple,np.asarray(points,float))))
    if len(points)<3:return np.asarray(points)
    def cross(o,a,b):return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    lower=[]
    for p in points:
        while len(lower)>=2 and cross(lower[-2],lower[-1],p)<=0:lower.pop()
        lower.append(p)
    upper=[]
    for p in reversed(points):
        while len(upper)>=2 and cross(upper[-2],upper[-1],p)<=0:upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1]+upper[:-1])


def project_pixels(points,transform,convention):
    """OpenCV camera coordinates -> exact LIBERO rotation/resize pixel convention."""
    homogeneous=np.column_stack((points,np.ones(len(points))))
    camera=homogeneous@transform.T
    if (camera[:,2]<=0).any():raise ValueError("Projected region crosses or lies behind the camera")
    xy=camera[:,:2]/camera[:,2:3]
    # get_camera_transform_matrix uses OpenCV (top-left) coordinates. Native
    # sim.render is bottom-left; robosuite convention then OpenPI rotates180°.
    xy[:,0]=255-xy[:,0]
    if convention==-1:xy[:,1]=255-xy[:,1]
    elif convention!=1:raise ValueError("Unknown robosuite image convention")
    return (xy+.5)*(224/256)-.5


def camera_transform(sim,name,height=256,width=256):
    """Native robosuite pinhole formula without camera_utils' unrelated h5py import."""
    camera=sim.model.camera_name2id(name)
    f=.5*height/np.tan(float(sim.model.cam_fovy[camera])*np.pi/360)
    intrinsic=np.eye(4);intrinsic[:3,:3]=[[f,0,width/2],[0,f,height/2],[0,0,1]]
    rotation=np.asarray(sim.data.cam_xmat[camera]).reshape(3,3)@np.diag([1.,-1.,-1.])
    extrinsic=np.eye(4);extrinsic[:3,:3]=rotation.T
    extrinsic[:3,3]=-rotation.T@np.asarray(sim.data.cam_xpos[camera])
    return intrinsic@extrinsic


def project_old_target_region(env, *, source_bddl, target, allow_invisible=False, camera_transforms=None):
    from PIL import Image, ImageDraw
    from robosuite import macros
    from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING

    backend=env.backend.env if hasattr(env.backend,"env") else env.backend
    sim=backend.sim
    region=source_region(str(Path(source_bddl).resolve()),target)
    x0,y0,x1,y1=map(float,region["xy_bounds"])
    site=sim.model.site_name2id(region["name"])
    site_center=np.asarray(sim.data.site_xpos[site])
    if not np.allclose(site_center[:2],[(x0+x1)/2,(y0+y1)/2],atol=1e-6):
        raise ValueError("Original placement region differs from current native region coordinates")
    floor_geom=sim.model.geom_name2id("floor")
    if int(sim.model.geom_type[floor_geom])!=0:
        raise ValueError("Native floor geometry is not a plane")
    normal=np.asarray(sim.data.geom_xmat[floor_geom]).reshape(3,3)[:,2]
    if not np.allclose(normal,[0,0,1],atol=1e-8):
        raise ValueError("Tilted support planes need an explicit plane projection")
    z=float(sim.data.geom_xpos[floor_geom][2])
    obj=backend.objects_dict[target]
    height=float(obj.top_offset[2]-obj.bottom_offset[2])
    if not 0<height<.5:raise ValueError("Invalid native target geometry height")
    floor=np.array([[x0,y0,z],[x1,y0,z],[x1,y1,z],[x0,y1,z]])
    prism=np.concatenate([floor,floor+[0,0,height]])
    convention=IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]
    masks={};cameras={}
    for camera,native in (("image","agentview"),("wrist_image","robot0_eye_in_hand")):
        transform=camera_transform(sim,native,256,256) if camera_transforms is None else camera_transforms[camera]
        try:
            polygons={"floor":project_pixels(floor,transform,convention),
                      "prism":convex_hull(project_pixels(prism,transform,convention))}
        except ValueError as exc:
            if not allow_invisible:raise
            for kind in ("floor","prism"):masks[f"{camera}__old_target_{kind}"]=np.zeros((224,224),bool)
            cameras[camera]={"valid_projection":False,"reason":str(exc),"floor_pixels":0,"prism_pixels":0}
            continue
        for kind,polygon in polygons.items():
            image=Image.new("1",(224,224));ImageDraw.Draw(image).polygon([tuple(p) for p in polygon],fill=1)
            masks[f"{camera}__old_target_{kind}"]=np.asarray(image,bool)
        cameras[camera]={"valid_projection":True,"world_to_opencv_pixel_homography":transform.tolist(),
                         "floor_polygon_xy":polygons["floor"].tolist(),"prism_silhouette_polygon_xy":polygons["prism"].tolist(),
                         "floor_pixels":int(masks[f"{camera}__old_target_floor"].sum()),"prism_pixels":int(masks[f"{camera}__old_target_prism"].sum())}
    metadata={**region,"floor_plane_z":z,"native_visualization_site_z":float(site_center[2]),
              "floor_plane_source":"native MuJoCo floor plane geom; excludes the visualization site's2cm z offset",
              "prism_height_m":height,"world_floor_vertices":floor.tolist(),
              "world_prism_vertices":prism.tolist(),"cameras":cameras,"robosuite_image_convention":convention,
              "scope":"Geometric prior-location region, not an object-instance mask; prism includes occluded volume. Floor polygon is a separate plane-only control.",
              "primary_mask":"old_target_prism","pixel_mapping":"native OpenCV projection, convention-aware rotation180,224/256 pixel-center resize",
              "camera_sampling":"current simulator camera cache" if camera_transforms is None else "native camera sensor sampling instant"}
    return masks,metadata
