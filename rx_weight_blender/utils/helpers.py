import sys
import time
from functools import wraps

from functools import partial

import maya.OpenMaya as om
import maya.OpenMayaAnim as oma
from maya import cmds

string_types = str if sys.version_info[0] == 3 else basestring  # noqa


def preserveSelection(function):
    @wraps(function)
    def result(*args, **kwargs):
        selection = cmds.ls(sl=True, fl=True)
        highlight = cmds.ls(hl=True, fl=True)
        try:
            return function(*args, **kwargs)
        finally:
            if selection:
                cmds.select(selection)
            else:
                cmds.select(clear=True)
            if highlight is not None:
                cmds.hilite(highlight)

    return result


def undoable(function):
    @wraps(function)
    def result(*args, **kwargs):
        cmds.undoInfo(openChunk=True)
        try:
            return function(*args, **kwargs)
        finally:
            cmds.undoInfo(closeChunk=True)

    return result


def timing(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        ts = time.time()
        result = f(*args, **kwargs)
        te = time.time()
        print('func:%r args:[%r, %r]' % (f.__name__, args, kwargs))
        print('func:%r took: %2.4f sec' % (f.__name__, te - ts))
        return result

    return wrap


def __assert_transform_type(obj=None, shape_type=None, sl=False):
    """
    :param obj: -> union[string,node,None]
    :param shape_type: -> string, e.g."mesh", "nurbsCurve", "joint", "locator"
                          ** special: "group"
    :return: node of the obj, None if assert is false.
    """

    if sl:
        sel = cmds.ls(sl=True, tr=True)
        if not sel:
            return cmds.warning("Nothing selected")
        obj = sel[0]

    if not obj:  # in case empty string passed in by UI element
        return

    if isinstance(obj, str) and cmds.objExists(obj):
        obj = obj
    elif isinstance(obj, om.MObjectHandle):
        obj = obj.object()
    else:
        return

    if cmds.nodeType(obj) == shape_type:
        return cmds.listRelatives(obj, parent=True, fullPath=True)[0]

    if cmds.nodeType(obj) != "transform":
        if sl:
            cmds.warning(f"Please select a {shape_type}")
        return

    shapes = cmds.listRelatives(obj, shapes=True, fullPath=True) or []

    if not shapes and shape_type == "group":
        return obj

    if shapes and cmds.nodeType(shapes[0]) == shape_type:
        return obj

    return None


def assert_joint(obj=None, sl=False):
    if sl:
        sel = cmds.ls(sl=True, tr=True)
        if not sel:
            return cmds.warning("Nothing selected")
        obj = sel[0]

    if not obj:  # in case empty string passed in by UI element
        return

    if isinstance(obj, str) and cmds.objExists(obj):
        obj = obj
    elif isinstance(obj, om.MObjectHandle):
        obj = obj.object()
    else:
        return

    if cmds.nodeType(obj) == "joint":
        return obj


assert_mesh = partial(__assert_transform_type, shape_type="mesh")
assert_nurbs = partial(__assert_transform_type, shape_type="nurbsCurve")
assert_group = partial(__assert_transform_type, shape_type="group")


def get_meshes(objs=None, sl=False):
    if sl:
        objs = cmds.ls(sl=True, tr=True)
        if not objs:
            return cmds.warning("Nothing selected")

    if not objs:
        return cmds.warning("No objects passed in")

    valid_objs = [obj for obj in objs if get_shape(obj)]
    meshes = [i for i in valid_objs if __assert_transform_type(i, shape_type="mesh")]
    warn = [i for i in valid_objs if not __assert_transform_type(i, shape_type="mesh")]

    if not meshes:
        return cmds.warning("No meshes are selected")

    if warn:
        cmds.warning("Skipped non-mesh types: {}".format([str(i) for i in warn]))

    return meshes


def get_joints(objs=None, sl=False):
    if sl:
        objs = cmds.ls(sl=True, tr=True)
        if not objs:
            return cmds.warning("Nothing selected")

    if not objs:
        return cmds.warning("No objects passed in")

    return [assert_joint(i) for i in objs if assert_joint(i)]


def get_shape(polygon):
    shapes = cmds.listRelatives(polygon, shapes=True, fullPath=True) or []
    for shape in shapes:
        if not cmds.getAttr(f"{shape}.io"):
            return shape


def get_skin_cluster(polygon=None):
    if not polygon:
        polygon = __assert_transform_type(shape_type="mesh")

    if not polygon:
        return

    history = cmds.listHistory(polygon, type="skinCluster") or []

    if history:
        return history[0]

    cmds.warning("\nCan't find skinCluster")
