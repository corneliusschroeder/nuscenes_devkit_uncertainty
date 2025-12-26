# nuScenes dev-kit.
# Code written by Holger Caesar, 2018.

from typing import List, Dict, Any

from scipy import stats
import numpy as np
from pyquaternion import Quaternion

from nuscenes.eval.common.data_classes import EvalBox
from nuscenes.utils.data_classes import Box

DetectionBox = Any  # Workaround as direct imports lead to cyclic dependencies.

MIN_VARIANCE = 1e-5


def within_cofidence_interval(gt_box: EvalBox, pred_box: EvalBox, confidence: float, distribution = stats.norm, period: float = 2*np.pi ):
    """
    Determines whether bounding box position (x, y) and extent (v_x, v_y) and rotation are within given confidence interval.
    :param gt_box: GT annotation sample.
    :param pred_box: Predicted sample.
    :confidence: Confidence percentage in (0; 1.0)
    :distribution: Distribution that implements the percent point function (ppf) with mean 0 and variance 1.
        Default: stats.norm (i.e. Normal Gaussian)
    :return: indicator array if offest x, y and bbox w, l were within the confidence interval.
    """
    z_score = distribution.ppf((1 + confidence) / 2)
    std_dev = np.sqrt(pred_box.uncertainty)

    distance_from_mean = z_score * std_dev
    
    full_dist = np.abs(np.array(pred_box.translation) - np.array(gt_box.translation))
    vel_difference = np.abs(np.array(pred_box.velocity) - np.array(gt_box.velocity))
    orient_diff = np.array(yaw_diff(gt_box, pred_box, period))

    return np.concatenate([full_dist[:2] <= distance_from_mean[:2], vel_difference <= distance_from_mean[7:], \
                           [orient_diff <= distance_from_mean[6]]]) + 0 


def gaussian_nll_error(gt_box: EvalBox, pred_box: EvalBox, epsilon: float=MIN_VARIANCE) -> np.ndarray:
    """
    Gaussian negative log-likelihood metric for position and bbox parameters
    :param gt_box: GT annotation sample.
    :param pred_box: Predicted sample.
    :return: Gaussian NLL difference for position and bounding box parameters.
    """
    # [ x, y, z, w, l, h, rot, vx, vy ]
    #   0  1  2  3  4  5   6    7   8
    variances = np.array(pred_box.uncertainty)
    # clip variances to avoid division by zero.
    variances = np.clip(variances, a_min=epsilon, a_max=None)
    log_variances = np.log(variances)

    # extract variances for position and velocity.
    pos_var, pos_logvar = variances[[0, 1]], log_variances[[0, 1]]
    vel_var, vel_logvar = variances[[7, 8]], log_variances[[7, 8]]
    size_var, size_logvar = variances[[3, 4, 5]], log_variances[[3, 4, 5]]
    
    # only consider (x, y) for the translation
    pos_norm_squared = (np.array(pred_box.translation[:2]) - np.array(gt_box.translation[:2])) ** 2

    # velocity is (dx, dy)
    vel_norm_squared = (np.array(pred_box.velocity) - np.array(gt_box.velocity)) ** 2

    # size is (w, l, h)
    size_norm_squared = (np.array(pred_box.size) - np.array(gt_box.size)) ** 2

    # Compute the negative log-likelihood for position and velocity
    # todo: the correct formula of NLL is 0.5 * (norm_squared / variance + log_variance + np.log(2 * np.pi))
    #       but we omit the constant term 0.5 * np.log(2 * np.pi) for simplicity.
    #       Keeping it like this for now to be consistent with the original implementation.
    nll_pos = pos_norm_squared / pos_var + pos_logvar
    nll_vel = vel_norm_squared / vel_var + vel_logvar
    nll_size = size_norm_squared / size_var + size_logvar

    return nll_pos, nll_vel, nll_size

def epistemic_variance(gt_box: EvalBox, pred_box: EvalBox) -> np.ndarray:
    """
    Computes the epistemic variance of the predicted box.
    :param gt_box: GT annotation sample.
    :param pred_box: Predicted sample.
    :return: Epistemic variance of the predicted box.
    """
    # [ x, y, z, w, l, h, rot, vx, vy ]
    #   0  1  2  3  4  5   6    7   8
    epistemic_variance = np.array(pred_box.epistemic_var)
    pos_epistemic_var = epistemic_variance[[0, 1]]  # x, y
    vel_epistemic_var = epistemic_variance[[7, 8]]  # vx, vy
    size_epistemic_var = epistemic_variance[[3, 4, 5]]  # w, l, h
    return pos_epistemic_var, vel_epistemic_var, size_epistemic_var

def aleatoric_variance(gt_box: EvalBox, pred_box: EvalBox) -> np.ndarray:
    """
    Computes the aleatoric variance of the predicted box.
    :param gt_box: GT annotation sample.
    :param pred_box: Predicted sample.
    :return: Aleatoric variance of the predicted box.
    """
    # [ x, y, z, w, l, h, rot, vx, vy ]
    #   0  1  2  3  4  5   6    7   8
    aleatoric_variance = np.array(pred_box.aleatoric_var)
    pos_aleatoric_var = aleatoric_variance[[0, 1]]  # x, y
    vel_aleatoric_var = aleatoric_variance[[7, 8]]  # vx, vy
    size_aleatoric_var = aleatoric_variance[[3, 4, 5]]  # w, l, h
    return pos_aleatoric_var, vel_aleatoric_var, size_aleatoric_var

def total_variance(gt_box: EvalBox, pred_box: EvalBox) -> np.ndarray:
    """
    Computes the total variance of the predicted box.
    :param gt_box: GT annotation sample.
    :param pred_box: Predicted sample.
    :return: Total variance of the predicted box.
    """
    pos_var, vel_var, size_var = pred_box.uncertainty[:2], pred_box.uncertainty[7:9], pred_box.uncertainty[3:6]
    return pos_var, vel_var, size_var

def center_offset(gt_box: EvalBox, pred_box: EvalBox) -> np.ndarray:
    """
    Computes the offset between the box centers (xy only).
    :param gt_box: GT annotation sample.
    :param pred_box: Predicted sample.
    :return: Offset x and y.
    """
    offset_x = pred_box.translation[0] - gt_box.translation[0]
    offset_y = pred_box.translation[1] - gt_box.translation[1]
    return offset_x, offset_y

def center_offset_var(gt_box: EvalBox, pred_box: EvalBox, epsilon=MIN_VARIANCE) -> np.ndarray:
    """
    Computes the variance of the offset between the box centers (xy only).
    :param gt_box: GT annotation sample.
    :param pred_box: Predicted sample.
    :return: Variance of offset x and y.
    """

    # [ x, y, z, w, l, h, rot, vx, vy ]
    #   0  1  2  3  4  5   6    7   8
    variances = np.array(pred_box.uncertainty)
    # clip variances to avoid division by zero.
    variances = np.clip(variances, a_min=epsilon, a_max=None)
    # extract variances for position x and y.
    return variances[0], variances[1]
    

def velocity_offset(gt_box: EvalBox, pred_box: EvalBox) -> np.ndarray:
    """
    Computes the offset between the velocity vectors (xy only).
    :param gt_box: GT annotation sample.
    :param pred_box: Predicted sample.
    :return: Offset vx and vy.
    """
    offset_vx = pred_box.velocity[0] - gt_box.velocity[0]
    offset_vy = pred_box.velocity[1] - gt_box.velocity[1]
    return offset_vx, offset_vy

def velocity_offset_var(gt_box: EvalBox, pred_box: EvalBox, epsilon=MIN_VARIANCE) -> np.ndarray:
    """
    Computes the variance of the offset between the velocity vectors (xy only).
    :param gt_box: GT annotation sample.
    :param pred_box: Predicted sample.
    :return: Variance of offset vx and vy.
    """

    # [ x, y, z, w, l, h, rot, vx, vy ]
    #   0  1  2  3  4  5   6    7   8
    variances = np.array(pred_box.uncertainty)
    # clip variances to avoid division by zero.
    variances = np.clip(variances, a_min=epsilon, a_max=None)
    # extract variances for velocity vx and vy.
    return variances[7], variances[8]

def center_distance(gt_box: EvalBox, pred_box: EvalBox) -> float:
    """
    L2 distance between the box centers (xy only).
    :param gt_box: GT annotation sample.
    :param pred_box: Predicted sample.
    :return: L2 distance.
    """
    return np.linalg.norm(np.array(pred_box.translation[:2]) - np.array(gt_box.translation[:2]))


def velocity_l2(gt_box: EvalBox, pred_box: EvalBox) -> float:
    """
    L2 distance between the velocity vectors (xy only).
    If the predicted velocities are nan, we return inf, which is subsequently clipped to 1.
    :param gt_box: GT annotation sample.
    :param pred_box: Predicted sample.
    :return: L2 distance.
    """
    return np.linalg.norm(np.array(pred_box.velocity) - np.array(gt_box.velocity))


def yaw_diff(gt_box: EvalBox, eval_box: EvalBox, period: float = 2*np.pi) -> float:
    """
    Returns the yaw angle difference between the orientation of two boxes.
    :param gt_box: Ground truth box.
    :param eval_box: Predicted box.
    :param period: Periodicity in radians for assessing angle difference.
    :return: Yaw angle difference in radians in [0, pi].
    """
    yaw_gt = quaternion_yaw(Quaternion(gt_box.rotation))
    yaw_est = quaternion_yaw(Quaternion(eval_box.rotation))

    return abs(angle_diff(yaw_gt, yaw_est, period))

def yaw_diff_var(gt_box: EvalBox, pred_box: EvalBox, epsilon=MIN_VARIANCE) -> np.ndarray:
    """
    Returns the yaw angle difference variance between the orientation of two boxes.
    :param gt_box: Ground truth box.
    :param eval_box: Predicted box.
    :param period: Periodicity in radians for assessing angle difference.
    :return: Yaw angle difference in radians in [0, pi].
    """
    # [ x, y, z, w, l, h, rot, vx, vy ]
    #   0  1  2  3  4  5   6    7   8
    variances = np.array(pred_box.uncertainty)
    # clip variances to avoid division by zero.
    variances = np.clip(variances, a_min=epsilon, a_max=None)
    # extract variances for velocity vx and vy.
    return np.array(variances[6])


def angle_diff(x: float, y: float, period: float):
    """
    Get the smallest angle difference between 2 angles: the angle from y to x.
    :param x: To angle.
    :param y: From angle.
    :param period: Periodicity in radians for assessing angle difference.
    :return: <float>. Signed smallest between-angle difference in range (-pi, pi).
    """

    # calculate angle difference, modulo to [0, 2*pi]
    diff = (x - y + period / 2) % period - period / 2
    if diff > np.pi:
        diff = diff - (2 * np.pi)  # shift (pi, 2*pi] to (-pi, 0]

    return diff


def attr_acc(gt_box: DetectionBox, pred_box: DetectionBox) -> float:
    """
    Computes the classification accuracy for the attribute of this class (if any).
    If the GT class has no attributes or the annotation is missing attributes, we assign an accuracy of nan, which is
    ignored later on.
    :param gt_box: GT annotation sample.
    :param pred_box: Predicted sample.
    :return: Attribute classification accuracy (0 or 1) or nan if GT annotation does not have any attributes.
    """
    if gt_box.attribute_name == '':
        # If the class does not have attributes or this particular sample is missing attributes, return nan, which is
        # ignored later. Note that about 0.4% of the sample_annotations have no attributes, although they should.
        acc = np.nan
    else:
        # Check that label is correct.
        acc = float(gt_box.attribute_name == pred_box.attribute_name)
    return acc


def scale_iou(sample_annotation: EvalBox, sample_result: EvalBox) -> float:
    """
    This method compares predictions to the ground truth in terms of scale.
    It is equivalent to intersection over union (IOU) between the two boxes in 3D,
    if we assume that the boxes are aligned, i.e. translation and rotation are considered identical.
    :param sample_annotation: GT annotation sample.
    :param sample_result: Predicted sample.
    :return: Scale IOU.
    """
    # Validate inputs.
    sa_size = np.array(sample_annotation.size)
    sr_size = np.array(sample_result.size)
    assert all(sa_size > 0), 'Error: sample_annotation sizes must be >0.'
    assert all(sr_size > 0), 'Error: sample_result sizes must be >0.'

    # Compute IOU.
    min_wlh = np.minimum(sa_size, sr_size)
    volume_annotation = np.prod(sa_size)
    volume_result = np.prod(sr_size)
    intersection = np.prod(min_wlh)  # type: float
    union = volume_annotation + volume_result - intersection  # type: float
    iou = intersection / union

    return iou


def quaternion_yaw(q: Quaternion) -> float:
    """
    Calculate the yaw angle from a quaternion.
    Note that this only works for a quaternion that represents a box in lidar or global coordinate frame.
    It does not work for a box in the camera frame.
    :param q: Quaternion of interest.
    :return: Yaw angle in radians.
    """

    # Project into xy plane.
    v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))

    # Measure yaw using arctan.
    yaw = np.arctan2(v[1], v[0])

    return yaw


def boxes_to_sensor(boxes: List[EvalBox], pose_record: Dict, cs_record: Dict):
    """
    Map boxes from global coordinates to the vehicle's sensor coordinate system.
    :param boxes: The boxes in global coordinates.
    :param pose_record: The pose record of the vehicle at the current timestamp.
    :param cs_record: The calibrated sensor record of the sensor.
    :return: The transformed boxes.
    """
    boxes_out = []
    for box in boxes:
        # Create Box instance.
        box = Box(box.translation, box.size, Quaternion(box.rotation))

        # Move box to ego vehicle coord system.
        box.translate(-np.array(pose_record['translation']))
        box.rotate(Quaternion(pose_record['rotation']).inverse)

        #  Move box to sensor coord system.
        box.translate(-np.array(cs_record['translation']))
        box.rotate(Quaternion(cs_record['rotation']).inverse)

        boxes_out.append(box)

    return boxes_out


def cummean(x: np.array) -> np.array:
    """
    Computes the cumulative mean up to each position in a NaN sensitive way
    - If all values are NaN return an array of ones.
    - If some values are NaN, accumulate arrays discording those entries.
    """
    if sum(np.isnan(x)) == len(x):
        # Is all numbers in array are NaN's.
        return np.ones(len(x))  # If all errors are NaN set to error to 1 for all operating points.
    else:
        # Accumulate in a nan-aware manner.
        sum_vals = np.nancumsum(x.astype(float))  # Cumulative sum ignoring nans.
        count_vals = np.cumsum(~np.isnan(x))  # Number of non-nans up to each position.
        return np.divide(sum_vals, count_vals, out=np.zeros_like(sum_vals), where=count_vals != 0)
