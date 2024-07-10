import maya.OpenMaya as om
import maya.OpenMayaAnim as oma
from maya import cmds
from .utils.helpers import undoable


def get_dag_path_by_name(name):
    selection_list = om.MSelectionList()
    selection_list.add(name)
    dag_path = om.MDagPath()
    selection_list.getDagPath(0, dag_path)
    return dag_path


def get_dag_names(dag_array, partial=False):
    if not partial:
        return [dag_array[i].fullPathName() for i in range(dag_array.length())]
    else:
        return [dag_array[i].partialPathName() for i in range(dag_array.length())]


# @timing
def get_full_selected_comp_ids():
    """ return the selected paths(MDagPathArray) and vertex ids(MIntArray)
    Careful that MDagPathArray could contain duplicates which is expected.
    (* We want to take advantage of this to trace the selection order)
    """
    selection = om.MSelectionList()
    om.MGlobal.getActiveSelectionList(selection, True)  # bool: track selection order
    sIt = om.MItSelectionList(selection)
    paths = om.MDagPathArray()
    i = 0
    py_idList = []
    while not sIt.isDone():
        dag = om.MDagPath()
        sIt.getDagPath(dag)
        if dag.apiType() != 296:
            return om.MGlobal.displayWarning("Selection must be vertex(s).")
        paths.append(dag)
        path = om.MDagPath()
        comp = om.MObject()
        stat = selection.getDagPath(i, path, comp)
        compFn = om.MFnSingleIndexedComponent(comp)
        ids = om.MIntArray()
        compFn.getElements(ids)
        py_idList.append(ids)
        i += 1
        sIt.next()
    return paths, py_idList


def dag_array_remove_dup(dag_array):
    names = get_dag_names(dag_array)
    names = list(set(names))
    new_array = om.MDagPathArray()
    [new_array.append(get_dag_path_by_name(i)) for i in names]
    return new_array


def get_component_by_ids(ids):
    fn_component = om.MFnSingleIndexedComponent()
    part_component_obj = fn_component.create(om.MFn.kMeshVertComponent)
    fn_component.addElements(ids)
    return part_component_obj


def getSkinCluster(dag):
    """A convenience function to return the skinCluster name and MObject
       # note : skin_fn_set = oma.MFnSkinCluster(skin_cluster_obj)
    params:
      dag (MDagPath): A MDagPath for the mesh we want to investigate.
    """
    skin_cluster = cmds.ls(cmds.listHistory(dag.fullPathName()), type="skinCluster")

    if len(skin_cluster) > 0:
        # get the MObject for that skinCluster node if there is one
        sel = om.MSelectionList()
        sel.add(skin_cluster[0])
        skin_cluster_obj = om.MObject()
        sel.getDependNode(0, skin_cluster_obj)
        return skin_cluster[0], skin_cluster_obj

    else:
        raise RuntimeError("Selected mesh has no skinCluster")


def get_infls(skin_cluster_obj, sort=True):
    skin_fn_set = oma.MFnSkinCluster(skin_cluster_obj)
    infl_objects = om.MDagPathArray()
    skin_fn_set.influenceObjects(infl_objects)
    infls_names = [infl_objects[i].fullPathName() for i in range(infl_objects.length())]
    return infls_names, infl_objects


# @timing
def _check_selection(pairs=False, targetNum=1):
    all_paths, all_vIds = get_full_selected_comp_ids()
    # print([paths[i].partialPathName() for i in range(paths.length())], vIds)
    # print(vIds)
    object_count = all_paths.length()

    # we should get the last selected object as target mesh by querying paths[paths.length() - 1]
    # target index should be queried by dict.
    # if using two source blend, target should be a MDagPathArray
    target_dags = om.MDagPathArray()
    target_indices = om.MIntArray()
    if not pairs:
        # sanity check
        if len(all_vIds[object_count - 1]) != 1:
            return om.MGlobal.displayWarning("Target selection must be single vertex.")
        if targetNum == 1:
            # sanity check
            if object_count < 2:
                return om.MGlobal.displayWarning("Selection not enough.")
            # build return data
            source_dict = {}
            for i in range(object_count - 1):
                key = all_paths[i].fullPathName()
                if key in source_dict.keys():
                    source_dict[key] = source_dict[key] + all_vIds[i]
                else:
                    source_dict[key] = all_vIds[i]
            target_dict = {}
            for i in range(object_count - 1, object_count):
                key = all_paths[i].fullPathName()
                if key in target_dict.keys():
                    target_dict[key] = target_dict[key] + all_vIds[i]
                else:
                    target_dict[key] = all_vIds[i]
            return source_dict, [target_dict], mergeDictionary(source_dict, target_dict)
        elif targetNum == 2:
            # sanity check, also index:-2 should be single
            if object_count < 2:
                return om.MGlobal.displayWarning("Selection not enough.")
            if len(all_vIds[object_count - 2]) != 1:
                return om.MGlobal.displayWarning("Target selection must be single vertex.")
            # build return data
            source_dict = {}
            for i in range(object_count - 2):
                key = all_paths[i].fullPathName()
                if key in source_dict.keys():
                    source_dict[key] = source_dict[key] + all_vIds[i]
                else:
                    source_dict[key] = all_vIds[i]
            targetA_dict = {}
            targetB_dict = {}
            for i in range(object_count - 2, object_count - 1):
                key = all_paths[i].fullPathName()
                if key in targetA_dict.keys():
                    targetA_dict[key] = targetA_dict[key] + all_vIds[i]
                else:
                    targetA_dict[key] = all_vIds[i]
            for i in range(object_count - 1, object_count):
                key = all_paths[i].fullPathName()
                if key in targetB_dict.keys():
                    targetB_dict[key] = targetB_dict[key] + all_vIds[i]
                else:
                    targetB_dict[key] = all_vIds[i]
            return source_dict, [targetA_dict, targetB_dict], mergeDictionary(
                mergeDictionary(source_dict, targetA_dict), targetB_dict)
    else:
        if targetNum == 1:
            # sanity check
            if object_count % 2 != 0:
                return om.MGlobal.displayWarning("Selection must be pairs (1 to 1).")
            # build return data
            source_dict = {}
            for i in range(int(object_count / 2)):
                key = all_paths[i].fullPathName()
                if key in source_dict.keys():
                    source_dict[key] = source_dict[key] + all_vIds[i]
                else:
                    source_dict[key] = all_vIds[i]
            target_dict = {}
            for i in range(int(object_count / 2), object_count):
                key = all_paths[i].fullPathName()
                if key in target_dict.keys():
                    target_dict[key] = target_dict[key] + all_vIds[i]
                else:
                    target_dict[key] = all_vIds[i]
            return source_dict, [target_dict], mergeDictionary(source_dict, target_dict)

        elif targetNum == 2:
            # sanity check
            if object_count % 3 != 0:
                return om.MGlobal.displayWarning("Selection must be pairs (1 to 2).")
            # build return data
            source_dict = {}
            for i in range(int(object_count / 3)):
                key = all_paths[i].fullPathName()
                if key in source_dict.keys():
                    source_dict[key] = source_dict[key] + all_vIds[i]
                else:
                    source_dict[key] = all_vIds[i]
            targetA_dict = {}
            targetB_dict = {}
            for i in range(int(object_count / 3), int(object_count * 2 / 3)):
                key = all_paths[i].fullPathName()
                if key in targetA_dict.keys():
                    targetA_dict[key] = targetA_dict[key] + all_vIds[i]
                else:
                    targetA_dict[key] = all_vIds[i]
            for i in range(int(object_count * 2 / 3), object_count):
                key = all_paths[i].fullPathName()
                if key in targetB_dict.keys():
                    targetB_dict[key] = targetB_dict[key] + all_vIds[i]
                else:
                    targetB_dict[key] = all_vIds[i]
            return source_dict, [targetA_dict, targetB_dict], mergeDictionary(
                mergeDictionary(source_dict, targetA_dict), targetB_dict)


# @timing
def get_minimum_shared_infls_by_mesh(all_dict):
    infl_names_result = []
    for mesh_name in all_dict.keys():
        dag = get_dag_path_by_name(mesh_name)
        skin_cluster_name, skin_cluster_obj = getSkinCluster(dag)
        infls_names, infl_objects = get_infls(skin_cluster_obj)
        infl_names_result.extend(infls_names)
    infl_names_result = list(set(infl_names_result))
    return infl_names_result


# @timing
def get_minimum_shared_infls_by_vertex(all_dict):
    shared_infl_names = []
    dagArray = om.MDagPathArray()
    for mesh_name in all_dict.keys():
        vert_ids = om.MIntArray()
        [vert_ids.append(i) for i in all_dict[mesh_name]]
        dag = get_dag_path_by_name(mesh_name)
        dagArray.append(dag)
        skin_cluster_name, skin_cluster_obj = getSkinCluster(dag)
        mFnSkinCluster = oma.MFnSkinCluster(skin_cluster_obj)
        inf_objects = om.MDagPathArray()
        mFnSkinCluster.influenceObjects(inf_objects)
        inf_indices = range(inf_objects.length())
        inf_count_util = om.MScriptUtil(inf_objects.length())
        # c++ utility needed for the get/set weights functions
        inf_count_ptr = inf_count_util.asUintPtr()
        inf_count = inf_count_util.asInt()
        # check the weights to get valid inf_ids, so we won't iterate all the infls during the real calculation.
        check_weights = om.MDoubleArray()
        mFnSkinCluster.getWeights(dag, get_component_by_ids(vert_ids), check_weights, inf_count_ptr)
        valid_infl_ids = om.MIntArray()
        for infl_index in range(inf_count):
            for vertex in range(vert_ids.length()):
                weight_index = vertex * inf_count + infl_index
                if check_weights[weight_index] > 0:
                    valid_infl_ids.append(infl_index)
                    break
        # print(valid_infl_ids, "-----------------------------")
        [shared_infl_names.append(inf_objects[inf_indices.index(i)].fullPathName()) for i in valid_infl_ids if
         inf_objects[inf_indices.index(i)].fullPathName() not in shared_infl_names]
    # print(shared_infl_names)

    return dagArray, shared_infl_names


# @timing
def _match_minimum_infls(dagArray, minimum_shared_infls):
    to_execute = []
    for i in range(dagArray.length()):
        dag = dagArray[i]
        skin_cluster_name, skin_cluster_obj = getSkinCluster(dag)
        infls_names, infl_objects = get_infls(skin_cluster_obj)
        infl_to_add = [i for i in minimum_shared_infls if i not in infls_names]
        to_execute.append((skin_cluster_name, infl_to_add))
    if to_execute:
        rig_node = cmds.ls("*.is_rig")
        if rig_node:
            # -- get the name of the rig
            rig_node = rig_node[0].split(".")[0]
            # -- query all connections
            dag_poses = cmds.listConnections("{0}.rigPoses".format(rig_node))
            aps = []
            if dag_poses:
                ctrls = cmds.dagPose(dag_poses[0], q=1, ap=1)
                if ctrls:
                    for ctl in ctrls:
                        aps.append(cmds.xform(ctl, q=1, m=1, ws=1))
                # -- restore the saved positions
                cmds.dagPose(dag_poses[0], restore=True)
                # -- add all new infls
                for skin_cluster_name, infl_to_add in to_execute:
                    cmds.skinCluster(skin_cluster_name, edit=True, addInfluence=infl_to_add, weight=0)
                # -- restore the pose again to the pose before running this command
                if ctrls and aps:
                    for j in range(len(ctrls)):
                        cmds.xform(ctrls[j], m=aps[j], ws=1)


def mergeDictionary(dict_1, dict_2):
    """
    Py3 version
    def mergeDictionary(dict_1, dict_2):
        dict_3 = {**dict_1, **dict_2}
        for key, value in dict_3.items():
            if key in dict_1 and key in dict_2:
                dict_3[key] = [i for i in value]
                dict_3[key].extend([i for i in dict_1[key]])
        return dict_3
    """
    dict_3 = dict_1.copy()
    for k, v in dict_2.items():
        if k in dict_1.keys():
            dict_3[k] = dict_3[k] + v
        else:
            dict_3[k] = v
    return dict_3


def infl_names_to_logical_indices(mFnSkinCluster, infl_names):
    inf_objects = om.MDagPathArray()
    mFnSkinCluster.influenceObjects(inf_objects)
    all_infl_names = [inf_objects[i].fullPathName() for i in range(inf_objects.length())]
    valid_logical_ids = om.MIntArray()
    [valid_logical_ids.append(all_infl_names.index(name)) for name in infl_names]
    return valid_logical_ids


def fix_weight_precision(weight_MDA, precision=0.005):
    [weight_MDA.set(0, i) for i in range(weight_MDA.length()) if weight_MDA[i] < precision]
    s = sum(list(weight_MDA))
    [weight_MDA.set(weight_MDA[i] / s, i) for i in range(weight_MDA.length())]
    return weight_MDA


def get_single_vertex_weight(dag, vid, infl_names):
    _, skin_cluster_obj = getSkinCluster(dag)
    mFnSkinCluster = oma.MFnSkinCluster(skin_cluster_obj)
    inf_objects = om.MDagPathArray()
    mFnSkinCluster.influenceObjects(inf_objects)
    valid_logical_ids = infl_names_to_logical_indices(mFnSkinCluster, infl_names)
    target_weight = om.MDoubleArray()
    mFnSkinCluster.getWeights(dag, get_component_by_ids(vid), valid_logical_ids, target_weight)
    # target_weight = fix_weight_precision(target_weight)
    return target_weight


# @timing
@undoable
def _collect_data(pairs=False, targetNum=1):
    try:
        source_dict, target_dict_list, all_dict = _check_selection(pairs, targetNum)
    except TypeError:
        return
    # sanity check
    all_dagArray, minimum_shared_infl_names = get_minimum_shared_infls_by_vertex(all_dict)
    # # check if selection has at least 2 shared infls
    if len(minimum_shared_infl_names) < 2:
        return om.MGlobal.displayWarning("selected vertices only shared one infl!")
    # # check if all the meshes has the required infls
    for i in range(all_dagArray.length()):
        dag = all_dagArray[i]
        skin_cluster_name, skin_cluster_obj = getSkinCluster(dag)
        infls_names, infl_objects = get_infls(skin_cluster_obj)
        for infl in minimum_shared_infl_names:
            if infl not in infls_names:
                return om.MGlobal.displayWarning("need to match infls first")

    # build return data
    source_dags = om.MDagPathArray()
    source_vert_ids = []  # py_list of MIntArray
    valid_infl_ids = []  # py_list of MIntArray
    mFnSkinClusters = []  # py_list of mFnSkinCluster
    old_source_weights = []  # py_list of MDoubleArray
    target_weights_list = []  # py_list of MDoubleArray

    # # 1. get all skinCluster object and the shared infl-names. - done
    # # note : method to get/set weights will use name order cuz index on skinClusters won't match in all cases
    # # 2. get target weights under specific condition,
    target_value_temp = []
    if not pairs:
        for i in range(targetNum):
            target_dict = target_dict_list[i]
            target_name, target_vert = list(target_dict.items())[0]
            dag = get_dag_path_by_name(target_name)
            target_weight_value = get_single_vertex_weight(dag, target_vert, minimum_shared_infl_names)
            target_value_temp.append(target_weight_value)
    # print(target_value_temp, "<- target_value_temp")
    # 3. return all the datas out to _compute with calc function
    if not pairs:
        for meshName in source_dict.keys():
            dag = get_dag_path_by_name(meshName)
            source_dags.append(dag)  # source_dags
            vert_ids = om.MIntArray()
            [vert_ids.append(i) for i in source_dict[meshName]]
            source_vert_ids.append(vert_ids)  # source_vert_ids
            skin_cluster, skin_cluster_obj = getSkinCluster(dag)
            cmds.skinPercent(skin_cluster, pruneWeights=0)  # create a undo point
            mFnSkinCluster = oma.MFnSkinCluster(skin_cluster_obj)
            mFnSkinClusters.append(mFnSkinCluster)  # mFnSkinClusters
            source_weight = om.MDoubleArray()
            infl_ids = infl_names_to_logical_indices(mFnSkinCluster, minimum_shared_infl_names)
            valid_infl_ids.append(infl_ids)  # valid_infl_ids
            # print(valid_infl_ids, "<- valid_infl_ids")
            mFnSkinCluster.getWeights(dag, get_component_by_ids(vert_ids), infl_ids, source_weight)
            source_weight_fixed = list(source_weight)
            old_source_weights.append(source_weight_fixed)  # old_source_weights
            for i in range(len(target_value_temp)):
                # target_weight = om.MDoubleArray()
                target_weight = []
                for j in range(vert_ids.length()):
                    target_weight += target_value_temp[i]
                target_weights_list.append(target_weight)
    return source_dags, source_vert_ids, valid_infl_ids, mFnSkinClusters, old_source_weights, target_weights_list


def _solve(data, blend_value):
    try:
        source_dags, source_vert_ids, valid_infl_ids, mFnSkinClusters, old_source_weights, target_weights_list = data
    except TypeError:
        return
    # sanity check
    if len(target_weights_list) != len(mFnSkinClusters):
        return om.MGlobal.displayWarning("Unexpected target_weights_list length, should match SkinClusters count.")
    # print(target_weights_list, "<-target_weights_list")
    for i in range(source_dags.length()):
        dag = source_dags[i]
        vert_ids = source_vert_ids[i]
        infl_ids = valid_infl_ids[i]
        mFnSkinCluster = mFnSkinClusters[i]
        old_source_weight = old_source_weights[i]
        target_weight = target_weights_list[i]
        new_weight = calc_blend_weight(old_source_weight, target_weight, blend_value)
        mFnSkinCluster.setWeights(dag, get_component_by_ids(vert_ids), infl_ids, new_weight)


def _solve_two_source(data, blend_value):
    try:
        source_dags, source_vert_ids, valid_infl_ids, mFnSkinClusters, old_source_weights, target_weights_list = data
    except TypeError:
        return
    # sanity check
    if len(target_weights_list) != len(mFnSkinClusters) * 2:
        return om.MGlobal.displayWarning("Unexpected target_weights_list length, should match SkinClusters count.")
    # print(target_weights_list, "<-target_weights_list")

    for i in range(len(mFnSkinClusters)):
        dag = source_dags[i]
        vert_ids = source_vert_ids[i]
        infl_ids = valid_infl_ids[i]
        mFnSkinCluster = mFnSkinClusters[i]
        old_source_weight = old_source_weights[i]
        targetA_weight = target_weights_list[i*2]
        targetB_weight = target_weights_list[i*2+1]
        if blend_value < 0:
            try:
                new_weight = calc_blend_weight(old_source_weight, targetA_weight, blend_value * -1)
            except TypeError:
                return
        else:
            try:
                new_weight = calc_blend_weight(old_source_weight, targetB_weight, blend_value)
            except TypeError:
                return
        mFnSkinCluster.setWeights(dag, get_component_by_ids(vert_ids), infl_ids, new_weight)


def _blend_half(data):
    try:
        source_dags, source_vert_ids, valid_infl_ids, mFnSkinClusters, old_source_weights, target_weights_list = data
    except TypeError:
        return
    # sanity check
    if len(target_weights_list) != len(mFnSkinClusters) * 2:
        return om.MGlobal.displayWarning("Unexpected target_weights_list length, should match SkinClusters count.")
    for i in range(len(mFnSkinClusters)):
        dag = source_dags[i]
        vert_ids = source_vert_ids[i]
        infl_ids = valid_infl_ids[i]
        mFnSkinCluster = mFnSkinClusters[i]
        old_source_weight = old_source_weights[i]
        targetA_weight_part = target_weights_list[i*2][:infl_ids.length()]
        targetB_weight_part = target_weights_list[i*2 + 1][:infl_ids.length()]
        try:
            new_weight_part = calc_blend_weight(targetA_weight_part, targetB_weight_part, 0.5)
        except TypeError:
            return
        new_weight = om.MDoubleArray()
        for _ in range(vert_ids.length()):
            new_weight += new_weight_part

        mFnSkinCluster.setWeights(dag, get_component_by_ids(vert_ids), infl_ids, new_weight)


def calc_blend_weight(old_source_weight, target_weight, blend_value):
    # sanity check
    if len(old_source_weight) != len(target_weight):
        om.MGlobal.displayWarning("calc error, source&target length are not match.")
        raise TypeError
    new_weight = om.MDoubleArray(len(target_weight))  # get the same length first
    # for i in range(target_weight.length()):
    for i in range(len(target_weight)):
        new_value = old_source_weight[i] * (1.0 - blend_value) + target_weight[i] * blend_value
        # new_value = min(new_value, 1.0)
        new_weight.set(new_value, i)
    return new_weight


# @timing
def _collect_data_old():
    # get the selected mesh and components
    selection = om.MSelectionList()
    om.MGlobal.getActiveSelectionList(selection, True)  # bool: track selection order

    if not selection.length():
        return

    selected_components = om.MObject()
    dag = om.MDagPath()
    selection.getDagPath(0, dag, selected_components)
    """
    # snippets
    iter = om.MItSelectionList(selection)
    while not iter.isDone():
        # variables
        dag = om.MDagPath()
        iter.getDagPath(dag)
        print(dag.fullPathName())
        iter.next()
    """
    dag.extendToShape()

    if dag.apiType() != 296:
        om.MGlobal.displayError("Selection must be a polygon mesh.")
        return
    else:
        # variable
        indices = om.MIntArray()
        vert_ids = om.MIntArray()
        # loop selection
        iter = om.MItSelectionList(selection)
        while not iter.isDone():
            # variables
            component = om.MObject()
            dag = om.MDagPath()

            iter.getDagPath(dag, component)

            if not component.isNull():
                objIndices = om.MIntArray()
                components = om.MFnSingleIndexedComponent(component)
                components.getElements(indices)

                for i in range(indices.length()):
                    vert_ids.append(indices[i])

            iter.next()
    skin_cluster, skin_cluster_obj = getSkinCluster(dag)

    # doing this can speed up iteration and also allows you to undo all of this
    cmds.skinPercent(skin_cluster, pruneWeights=0)

    mFnSkinCluster = oma.MFnSkinCluster(skin_cluster_obj)

    inf_objects = om.MDagPathArray()
    # returns a list of the DagPaths of the joints affecting the mesh
    mFnSkinCluster.influenceObjects(inf_objects)
    inf_count_util = om.MScriptUtil(inf_objects.length())

    # c++ utility needed for the get/set weights functions
    inf_count_ptr = inf_count_util.asUintPtr()
    inf_count = inf_count_util.asInt()
    # inf_ids = om.MIntArray()

    # check the weights to get valid inf_ids so we won't iterate all the infls.
    check_weights = om.MDoubleArray()
    mFnSkinCluster.getWeights(dag, get_component_by_ids(vert_ids), check_weights, inf_count_ptr)

    valid_infl_ids = om.MIntArray()
    for v in range(vert_ids.length()):
        for i in range(inf_count):
            index = v * inf_count + i
            if i in valid_infl_ids:
                continue
            if check_weights[index] > 0:
                valid_infl_ids.append(i)
    valid_inf_count = valid_infl_ids.length()
    # sort valid_infl_ids
    for i in range(valid_inf_count - 1):
        if valid_infl_ids[i] > valid_infl_ids[i + 1]:
            temp = valid_infl_ids[i]
            valid_infl_ids[i] = valid_infl_ids[i + 1]
            valid_infl_ids[i + 1] = temp
    # get the weight of source components
    old_weights = om.MDoubleArray()
    mFnSkinCluster.getWeights(dag, get_component_by_ids(vert_ids), valid_infl_ids, old_weights)

    return dag, vert_ids, valid_inf_count, valid_infl_ids, mFnSkinCluster, old_weights


def calc_blend_weight_old(data, blend_value):
    dag, vert_ids, valid_inf_count, valid_infl_ids, mFnSkinCluster, old_weights = data
    trueIndices = list(vert_ids)
    trueIndices.sort()
    target_index = trueIndices.index(vert_ids[-1]) * valid_inf_count
    target_weights = old_weights[target_index:target_index + valid_inf_count]
    source_weights = om.MDoubleArray()
    source_weights.copy(old_weights)
    [source_weights.remove(target_index) for i in range(valid_inf_count)]

    new_weights = om.MDoubleArray(source_weights)

    for i in range(int(source_weights.length() / valid_inf_count)):
        for j in range(valid_inf_count):
            index = i * valid_inf_count + j
            value = source_weights[index] * (1.0 - blend_value) + target_weights[j] * blend_value
            new_weights.set(value, int(index))
    for k in range(valid_inf_count):
        new_weights.insert(target_weights[-k - 1], target_index)

    mFnSkinCluster.setWeights(dag, get_component_by_ids(vert_ids), valid_infl_ids, new_weights, True,
                              source_weights)


def calc_blend_weight_two_source(data, blend_value):
    dag, vert_ids, valid_inf_count, valid_infl_ids, mFnSkinCluster, old_weights = data
    trueIndices = list(vert_ids)
    trueIndices.sort()
    target_index_a = trueIndices.index(vert_ids[-2]) * valid_inf_count
    target_weights_a = old_weights[target_index_a:target_index_a + valid_inf_count]
    target_index_b = trueIndices.index(vert_ids[-1]) * valid_inf_count
    target_weights_b = old_weights[target_index_b:target_index_b + valid_inf_count]
    source_weights = om.MDoubleArray()
    source_weights.copy(old_weights)
    [source_weights.remove(max(target_index_b, target_weights_a)) for i in range(valid_inf_count)]
    [source_weights.remove(min(target_index_b, target_weights_a)) for i in range(valid_inf_count)]

    new_weights = om.MDoubleArray(source_weights)

    # build new weights for source vertices first
    for i in range(int(vert_ids[:-2].length())):
        for j in range(valid_inf_count):
            index = i * valid_inf_count + j
            if blend_value >= 0:
                value = source_weights[index] * (1.0 - blend_value) + target_weights_b[j] * blend_value
            else:
                value = source_weights[index] * (1.0 + blend_value) - target_weights_a[j] * blend_value
            new_weights.set(value, int(index))
    # reassemble the target weights, do the larger index first.
    for k in range(valid_inf_count):
        larger, smaller = max(target_index_a, target_index_b), min(target_index_a, target_index_b)
        new_weights.insert(target_weights_b[-k - 1], max(target_index_a, target_index_b))
        new_weights.insert(target_weights_a[-k - 1], max(target_index_a, target_index_b))

    mFnSkinCluster.setWeights(dag, get_component_by_ids(vert_ids), valid_infl_ids, new_weights, True,
                              source_weights)


"""
backup
def calc_blend_weight(data, blend_value):
    dag, vert_ids, inf_count, inf_ids, mFnSkinCluster, old_weights = data
    trueIndices = list(vert_ids)
    trueIndices.sort()
    target_index = trueIndices.index(vert_ids[-1]) * inf_count
    target_weights = old_weights[target_index:target_index + inf_count]
    source_weights = om.MDoubleArray()
    source_weights.copy(old_weights)
    [source_weights.remove(target_index) for i in range(inf_count)]

    new_weights = om.MDoubleArray(source_weights)

    for i in range(int(source_weights.length() / inf_count)):
        for j in range(inf_count):
            index = i * inf_count + j
            value = source_weights[index] * (1.0 - blend_value) + target_weights[j] * blend_value
            new_weights.set(value, int(index))
    for k in range(inf_count):
        new_weights.insert(target_weights[-k - 1], target_index)

    # mFnSkinCluster.setWeights(dag, get_component_by_ids(vert_ids[:-1]), inf_ids, new_weights, True,
    #                           source_weights)
    mFnSkinCluster.setWeights(dag, get_component_by_ids(vert_ids), inf_ids, new_weights, True,
                              source_weights)
"""
