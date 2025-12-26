# nuScenes dev-kit.
# Code written by Holger Caesar & Oscar Beijbom, 2018.

import argparse
import json
import os
import random
import time
import pandas as pd
from typing import Optional, Tuple, Dict, Any
from tqdm import tqdm
from pyquaternion import Quaternion

import numpy as np

from nuscenes import NuScenes
from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_tp
from nuscenes.eval.detection.constants import TP_METRICS
from nuscenes.eval.detection.data_classes import DetectionConfig, DetectionMetrics, DetectionBox, \
    DetectionMetricDataList
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.loaders import load_prediction, load_gt, add_center_dist, filter_eval_boxes
from nuscenes.eval.detection.render import summary_plot, class_pr_curve, class_tp_curve, dist_pr_curve, visualize_sample, class_ece_curve, class_prec_rec_curve, plot_bev_heatmaps
from nuscenes.eval.common.config import config_factory
from nuscenes.calibration.ece import expected_calibration_error


def _md_or_none(md_list, cname, dist):
    try:
        return md_list[(cname, dist)]
    except KeyError:
        return None

class DetectionEval:
    """
    This is the official nuScenes detection evaluation code.
    Results are written to the provided output_dir.

    nuScenes uses the following detection metrics:
    - Mean Average Precision (mAP): Uses center-distance as matching criterion; averaged over distance thresholds.
    - True Positive (TP) metrics: Average of translation, velocity, scale, orientation and attribute errors.
    - nuScenes Detection Score (NDS): The weighted sum of the above.

    Here is an overview of the functions in this method:
    - init: Loads GT annotations and predictions stored in JSON format and filters the boxes.
    - run: Performs evaluation and dumps the metric data to disk.
    - render: Renders various plots and dumps to disk.

    We assume that:
    - Every sample_token is given in the results, although there may be not predictions for that sample.

    Please see https://www.nuscenes.org/object-detection for more details.
    """
    def __init__(self,
                 nusc: NuScenes,
                 config: DetectionConfig,
                 result_path: str,
                 eval_set: str,
                 output_dir: str = None,
                 verbose: bool = True):
        """
        Initialize a DetectionEval object.
        :param nusc: A NuScenes object.
        :param config: A DetectionConfig object.
        :param result_path: Path of the nuScenes JSON result file.
        :param eval_set: The dataset split to evaluate on, e.g. train, val or test.
        :param output_dir: Folder to save plots and results to.
        :param verbose: Whether to print to stdout.
        """
        self.nusc = nusc
        self.result_path = result_path
        self.eval_set = eval_set
        self.output_dir = output_dir
        self.verbose = verbose
        self.cfg = config

        # Check result file exists.
        assert os.path.exists(result_path), 'Error: The result file does not exist!'

        # Make dirs.
        self.plot_dir = os.path.join(self.output_dir, 'plots')
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.isdir(self.plot_dir):
            os.makedirs(self.plot_dir)

        # Load data.
        if verbose:
            print('Initializing nuScenes detection evaluation')
        self.pred_boxes, self.meta = load_prediction(self.result_path, self.cfg.max_boxes_per_sample, DetectionBox,
                                                     verbose=verbose)
        self.gt_boxes = load_gt(self.nusc, self.eval_set, DetectionBox, verbose=verbose)

        assert set(self.pred_boxes.sample_tokens) == set(self.gt_boxes.sample_tokens), \
            "Samples in split doesn't match samples in predictions."

        # Add center distances.
        self.pred_boxes = add_center_dist(nusc, self.pred_boxes)
        self.gt_boxes = add_center_dist(nusc, self.gt_boxes)

        # Filter boxes (distance, points per box, etc.).
        if verbose:
            print('Filtering predictions')
        self.pred_boxes = filter_eval_boxes(nusc, self.pred_boxes, self.cfg.class_range, verbose=verbose)
        if verbose:
            print('Filtering ground truth annotations')
        self.gt_boxes = filter_eval_boxes(nusc, self.gt_boxes, self.cfg.class_range, verbose=verbose)

        self.sample_tokens = self.gt_boxes.sample_tokens

    def _build_ego_poses(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Map sample_token -> (R_ego 3x3, t_ego 3, in world coords) for LIDAR_TOP."""
        poses = {}
        for tok in self.sample_tokens:
            sample = self.nusc.get('sample', tok)
            sd = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
            ego = self.nusc.get('ego_pose', sd['ego_pose_token'])
            R = Quaternion(ego['rotation']).rotation_matrix.astype(float)   # world->ego uses R^T
            t = np.array(ego['translation'], dtype=float)
            poses[tok] = (R, t)
        return poses

    def evaluate(self) -> Tuple[DetectionMetrics, DetectionMetricDataList]:
        """
        Performs the actual evaluation.
        :return: A tuple of high-level and the raw metric data.
        """
        start_time = time.time()

        # -----------------------------------
        # Step 1: Accumulate metric data for all classes and distance thresholds.
        # -----------------------------------
        if self.verbose:
            print('Accumulating metric data...')
        metric_data_list = DetectionMetricDataList()

        ego_poses = self._build_ego_poses()

        print(f'Uncertainty distribution in eval: {self.cfg.distribution}')
        for class_name in tqdm(self.cfg.class_names, desc='Accumulating metric data'):
            for dist_th in self.cfg.dist_ths:
                compute_calibration = dist_th == self.cfg.dist_th_tp
                md = accumulate(
                    self.gt_boxes, 
                    self.pred_boxes, 
                    class_name, 
                    self.cfg.dist_fcn_callable, 
                    dist_th, 
                    uncertainty_distribution=self.cfg.distribution,
                    compute_ci=compute_calibration,
                    compute_ece=compute_calibration,
                    ego_poses=ego_poses,
                )
                metric_data_list.set(class_name, dist_th, md)

        # -----------------------------------
        # Step 2: Calculate metrics from the data.
        # -----------------------------------
        if self.verbose:
            print('Calculating metrics...')

        targets = set()
        metrics = DetectionMetrics(self.cfg)

        for class_name in tqdm(self.cfg.class_names, desc='Calculating metrics'):
            # Compute APs.
            for dist_th in self.cfg.dist_ths:
                metric_data = metric_data_list[(class_name, dist_th)]
                ap = calc_ap(metric_data, self.cfg.min_recall, self.cfg.min_precision)
                metrics.add_label_ap(class_name, dist_th, ap)

            # use metric_data for current class and dist_th_tp (default is 2.0)
            metric_data = metric_data_list[(class_name, self.cfg.dist_th_tp)]

            # Compute TP metrics.
            for metric_name in TP_METRICS:
                if class_name in ['traffic_cone'] and metric_name in ['attr_err', 'vel_err', 'orient_err']:
                    tp = np.nan
                elif class_name in ['barrier'] and metric_name in ['attr_err', 'vel_err']:
                    tp = np.nan
                else:
                    tp = calc_tp(metric_data, self.cfg.min_recall, metric_name)
                metrics.add_label_tp(class_name, metric_name, tp)

            # compute confidence intervals
            axes = ['x', 'y', 'v_x', 'v_y', 'orient']
            for ci, interval in metric_data.ci_evaluation.items():
                for i, axis in enumerate(axes): 
                    name = f"CI_{ci}_of_{axis}"
                    first_ind = round(100 * self.cfg.min_recall) + 1
                    last_ind = metric_data.max_recall_ind
                    tp = np.mean(np.array(interval)[first_ind: last_ind + 1, i]) if last_ind >= first_ind else 1.
                    metrics.add_label_tp(class_name, name, tp)

            # compute ECE
            for target_name, calibration_df in metric_data.calib_dfs.items():
                targets.add(target_name)
                ece = expected_calibration_error(calibration_df)
                metrics.add_label_ece(class_name, target_name, ece)

            if hasattr(metric_data, "aleatoric_var") and metric_data.aleatoric_var is not None:
                print('DEBUG:', class_name, 'aleatoric variance:', metric_data.aleatoric_var.mean())
                metrics.add_label_var(class_name, "aleatoric", float(np.mean(metric_data.aleatoric_var)))
            if hasattr(metric_data, "epistemic_var") and metric_data.epistemic_var is not None:
                metrics.add_label_var(class_name, "epistemic", float(np.mean(metric_data.epistemic_var)))
            if hasattr(metric_data, "total_var") and metric_data.total_var is not None:
                metrics.add_label_var(class_name, "total", float(np.mean(metric_data.total_var)))
        
        # compute total ECE per target (over all classes)
        for target in targets:
            all_dfs = []
            for class_name in self.cfg.class_names:
                md = metric_data_list[(class_name, self.cfg.dist_th_tp)]
                if target in md.calib_dfs:
                    all_dfs.append(md.calib_dfs[target])

            combined_df = pd.concat(all_dfs, ignore_index=True)
            ece = expected_calibration_error(combined_df)
            metrics.add_target_wise_ece(target, ece)

        # Compute evaluation time.
        metrics.add_runtime(time.time() - start_time)

        return metrics, metric_data_list

    def render(self, metrics: DetectionMetrics, md_list: DetectionMetricDataList, wandb_log: Optional[bool] = True) -> None:
        """
        Renders various PR and TP curves.
        :param metrics: DetectionMetrics instance.
        :param md_list: DetectionMetricDataList instance.
        """
        if self.verbose:
            print('Rendering PR and TP curves')

        def savepath(name):
            return os.path.join(self.plot_dir, name + '.pdf')

        summary_plot(md_list, metrics, min_precision=self.cfg.min_precision, min_recall=self.cfg.min_recall,
                     dist_th_tp=self.cfg.dist_th_tp, savepath=savepath('summary'))

        for detection_name in self.cfg.class_names:
            class_pr_curve(md_list, metrics, detection_name, self.cfg.min_precision, self.cfg.min_recall,
                           savepath=savepath(detection_name + '_pr'), wandb_log=wandb_log)

            class_tp_curve(md_list, metrics, detection_name, self.cfg.min_recall, self.cfg.dist_th_tp,
                           savepath=savepath(detection_name + '_tp'), wandb_log=wandb_log)
            
            class_ece_curve(md_list, metrics, detection_name, self.cfg.dist_th_tp,
                            savepath=savepath(detection_name + '_ece'), wandb_log=wandb_log)
            
            class_prec_rec_curve(md_list, detection_name, self.cfg.dist_th_tp,
                                    savepath=savepath(detection_name + '_prec_rec'), wandb_log=wandb_log)
            
            # ---- BEV heatmaps (new) ----
            # Use the same distance threshold as TP/ECE by default.
            try:
                md = md_list[(detection_name, self.cfg.dist_th_tp)]
            except KeyError:
                # If that (class, dist) pair doesn't exist, skip heatmaps for this class.
                md = None

            if md is not None and getattr(md, "bev_heatmaps", None):
                # Choose a concise but informative set of layers to plot.
                # Only include keys that actually exist in md.bev_heatmaps.
                available = set(md.bev_heatmaps.keys())
                desired = [
                    "mse_pos", "ale_pos", "epi_pos",
                    "mse_vel", "ale_vel", "epi_vel",
                    "ale_mean", "epi_mean", "count",
                ]
                keys = [k for k in desired if k in available]

                # Plot & save; also log to W&B if enabled.
                # Mask very sparse cells to avoid noisy colorbars.
                outfile = savepath(f"{detection_name}_bev_heatmaps")
                try:
                    plot_bev_heatmaps(
                        md,
                        keys=keys,
                        min_count=2,
                        group_vmax=False,
                        ncols=3,
                        figsize_per_plot=3.0,
                        savepath=outfile,
                        wandb_log=bool(wandb_log),
                        wandb_prefix=f"BEV/{detection_name}"
                    )
                except Exception as e:
                    if self.verbose:
                        print(f"[WARN] Skipping BEV heatmaps for {detection_name}: {e}")


        # ---- Combined BEV heatmaps across all classes ----
        # Weighted merge of per-class heatmaps: mean = sum(class_mean * class_count) / sum(count)
        value_keys = [
            "mse_pos", "ale_pos", "epi_pos",
            "mse_vel", "ale_vel", "epi_vel",
            "ale_mean", "epi_mean"
        ]
        combined = None
        meta_keys = ("x_edges", "y_edges", "x_range", "y_range", "bin_size")

        for cname in self.cfg.class_names:
            md = _md_or_none(md_list, cname, self.cfg.dist_th_tp)
            if md is None or not getattr(md, "bev_heatmaps", None):
                continue
            hm = md.bev_heatmaps
            if "count" not in hm:
                continue

            if combined is None:
                # init accumulators
                combined = {
                    "count": hm["count"].astype(float).copy(),
                }
                # copy meta so plotting uses consistent axes
                for mk in meta_keys:
                    if mk in hm:
                        combined[mk] = hm[mk]
                # sum_k = mean_k * count
                for k in value_keys:
                    if k in hm:
                        combined[f"sum_{k}"] = np.nan_to_num(hm[k]) * hm["count"]
            else:
                # sanity: if grid differs, skip (or assert)
                same_grid = (
                    np.array_equal(combined.get("x_edges"), hm.get("x_edges")) and
                    np.array_equal(combined.get("y_edges"), hm.get("y_edges"))
                )
                if not same_grid:
                    if self.verbose:
                        print(f"[WARN] Skipping {cname} in ALL heatmap: grid mismatch.")
                    continue

                combined["count"] += hm["count"]
                for k in value_keys:
                    if k in hm and f"sum_{k}" in combined:
                        combined[f"sum_{k}"] += np.nan_to_num(hm[k]) * hm["count"]

        # finalize and plot
        if combined is not None and np.any(combined["count"] > 0):
            for k in list(combined.keys()):
                if not k.startswith("sum_"):
                    continue
                name = k[4:]
                denom = combined["count"].copy()
                with np.errstate(invalid="ignore", divide="ignore"):
                    m = combined[k] / denom
                    m[denom == 0] = np.nan
                combined[name] = m
                del combined[k]

            # wrap dict to reuse plot_bev_heatmaps(md_like, ...)
            class _MDWrap:
                def __init__(self, bev):
                    self.bev_heatmaps = bev
            md_all = _MDWrap(combined)

            available = set(combined.keys())
            keys = [k for k in value_keys if k in available] + ["count"]

            outfile = savepath("bev_heatmaps_ALL")
            try:
                plot_bev_heatmaps(
                    md_all,
                    keys=keys,
                    min_count=2,            # mask sparse bins
                    group_vmax=False,   # comparable colormaps within a group
                    ncols=3,
                    figsize_per_plot=3.0,
                    savepath=outfile,
                    wandb_log=bool(wandb_log),
                    wandb_prefix="BEV/ALL",
                )
            except Exception as e:
                if self.verbose:
                    print(f"[WARN] Skipping combined BEV heatmaps: {e}")

        for dist_th in self.cfg.dist_ths:
            dist_pr_curve(md_list, metrics, dist_th, self.cfg.min_precision, self.cfg.min_recall,
                          savepath=savepath('dist_pr_' + str(dist_th)), wandb_log=wandb_log)

    def main(
        self,
        plot_examples: int = 0,
        render_curves: bool = True,
        wandb_log: Optional[bool] = True,
    ) -> Dict[str, Any]:
        """
        Main function that loads the evaluation code, visualizes samples, runs the evaluation and renders stat plots.
        :param plot_examples: How many example visualizations to write to disk.
        :param render_curves: Whether to render PR and TP curves to disk.
        :return: A dict that stores the high-level metrics and meta data.
        """
        if plot_examples > 0:
            # Select a random but fixed subset to plot.
            random.seed(42)
            sample_tokens = list(self.sample_tokens)
            random.shuffle(sample_tokens)
            sample_tokens = sample_tokens[:plot_examples]

            # Visualize samples.
            example_dir = os.path.join(self.output_dir, 'examples')
            if not os.path.isdir(example_dir):
                os.mkdir(example_dir)
            for sample_token in sample_tokens:
                visualize_sample(
                    self.nusc,
                    sample_token,
                    self.gt_boxes if self.eval_set != 'test' else EvalBoxes(),
                    # Don't render test GT.
                    self.pred_boxes,
                    eval_range=max(self.cfg.class_range.values()),
                    savepath=os.path.join(example_dir, '{}.png'.format(sample_token)),
                    wandb_log=wandb_log,
                )

        # Run evaluation.
        metrics, metric_data_list = self.evaluate()

        # Render PR and TP curves.
        if render_curves:
            self.render(metrics, metric_data_list, wandb_log=wandb_log)

        # Dump the metric data, meta and metrics to disk.
        if self.verbose:
            print('Saving metrics to: %s' % self.output_dir)
        metrics_summary = metrics.serialize()
        metrics_summary['meta'] = self.meta.copy()
        with open(os.path.join(self.output_dir, 'metrics_summary.json'), 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        with open(os.path.join(self.output_dir, 'metrics_details.json'), 'w') as f:
            json.dump(metric_data_list.serialize(), f, indent=2)

        # Print high-level metrics.
        print('mAP: %.4f' % (metrics_summary['mean_ap']))
        err_name_mapping = {
            'trans_err': 'mATE',
            'scale_err': 'mASE',
            'orient_err': 'mAOE',
            'vel_err': 'mAVE',
            'attr_err': 'mAAE',
            'nll_gauss_error_all': 'mGNLL',
            'trans_gauss_err': 'mNLL_POS', 
            'vel_gauss_err': 'mNLL_VEL',
            'rot_gauss_err': 'mNLL_ROT',
            'size_gauss_err': 'mNLL_SIZE',
        }
        for tp_name, tp_val in metrics_summary['tp_errors'].items():
            print('%s: %.4f' % (err_name_mapping[tp_name], tp_val))
        print('NDS: %.4f' % (metrics_summary['nd_score']))
        print('Eval time: %.1fs' % metrics_summary['eval_time'])

        # Print per-class metrics.
        print()
        print('Per-class results:')
        print('Object Class\tAP\tATE\tASE\tAOE\tAVE\tAAE\tmGNLL\tmNLL_POS\tmNLL_VEL\tmNLL_SIZE')
        class_aps = metrics_summary['mean_dist_aps']
        class_tps = metrics_summary['label_tp_errors']
        for class_name in class_aps.keys():
            print('%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
                  % (class_name, class_aps[class_name],
                     class_tps[class_name]['trans_err'],
                     class_tps[class_name]['scale_err'],
                     class_tps[class_name]['orient_err'],
                     class_tps[class_name]['vel_err'],
                     class_tps[class_name]['attr_err'],
                     class_tps[class_name]['nll_gauss_error_all'],
                     class_tps[class_name]['trans_gauss_err'],
                     class_tps[class_name]['vel_gauss_err'],
                     class_tps[class_name]['size_gauss_err']
                    )
            )

        print('Mean ECE:')
        for class_name, ece in metrics_summary['mean_label_ece'].items():
            print('%s: %.4f' % (class_name, ece))
        return metrics_summary


class NuScenesEval(DetectionEval):
    """
    Dummy class for backward-compatibility. Same as DetectionEval.
    """


if __name__ == "__main__":

    # Settings.
    parser = argparse.ArgumentParser(description='Evaluate nuScenes detection results.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('result_path', type=str, help='The submission as a JSON file.')
    parser.add_argument('--output_dir', type=str, default='~/nuscenes-metrics',
                        help='Folder to store result metrics, graphs and example visualizations.')
    parser.add_argument('--eval_set', type=str, default='val',
                        help='Which dataset split to evaluate on, train, val or test.')
    parser.add_argument('--dataroot', type=str, default='/data/sets/nuscenes',
                        help='Default nuScenes data directory.')
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                        help='Which version of the nuScenes dataset to evaluate on, e.g. v1.0-trainval.')
    parser.add_argument('--config_path', type=str, default='',
                        help='Path to the configuration file.'
                             'If no path given, the CVPR 2019 configuration will be used.')
    parser.add_argument('--plot_examples', type=int, default=10,
                        help='How many example visualizations to write to disk.')
    parser.add_argument('--render_curves', type=int, default=1,
                        help='Whether to render PR and TP curves to disk.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Whether to print to stdout.')
    args = parser.parse_args()

    result_path_ = os.path.expanduser(args.result_path)
    output_dir_ = os.path.expanduser(args.output_dir)
    eval_set_ = args.eval_set
    dataroot_ = args.dataroot
    version_ = args.version
    config_path = args.config_path
    plot_examples_ = args.plot_examples
    render_curves_ = bool(args.render_curves)
    verbose_ = bool(args.verbose)

    if config_path == '':
        cfg_ = config_factory('detection_cvpr_2019')
    else:
        with open(config_path, 'r') as _f:
            cfg_ = DetectionConfig.deserialize(json.load(_f))

    nusc_ = NuScenes(version=version_, verbose=verbose_, dataroot=dataroot_)
    nusc_eval = DetectionEval(nusc_, config=cfg_, result_path=result_path_, eval_set=eval_set_,
                              output_dir=output_dir_, verbose=verbose_)
    nusc_eval.main(plot_examples=plot_examples_, render_curves=render_curves_)
