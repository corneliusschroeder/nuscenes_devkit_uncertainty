"""
nuScenes dev-kit.
Code written by Holger Caesar, Caglayan Dicle and Oscar Beijbom, 2019.

This code is based on:

py-motmetrics at:
https://github.com/cheind/py-motmetrics
"""
from collections import OrderedDict
from itertools import count

import motmetrics
import numpy as np
import pandas as pd


class MOTAccumulatorCustom(motmetrics.mot.MOTAccumulator):
    def __init__(self):
        super().__init__()

    @staticmethod
    def new_event_dataframe_with_data(indices, events):
        """
        Create a new DataFrame filled with data.
        This version overwrites the original in MOTAccumulator achieves about 2x speedups.

        Params
        ------
        indices: list
            list of tuples (frameid, eventid)
        events: list
            list of events where each event is a list containing
            'Type', 'OId', HId', 'D'
        """

        idx = pd.MultiIndex.from_tuples(indices, names=['FrameId', 'Event'])
        df = pd.DataFrame(events, index=idx, columns=['Type', 'OId', 'HId', 'D'])
        #df['mov_class_res'] = np.nan
        return df

    @staticmethod
    def new_event_dataframe():
        """ Create a new DataFrame for event tracking. """
        idx = pd.MultiIndex(levels=[[], []], codes=[[], []], names=['FrameId', 'Event'])
        cats = pd.Categorical([], categories=['RAW', 'FP', 'MISS', 'SWITCH', 'MATCH'])
        df = pd.DataFrame(
            OrderedDict([
                ('Type', pd.Series(cats)),  # Type of event. One of FP (false positive), MISS, SWITCH, MATCH
                ('OId', pd.Series(dtype=object)),
                # Object ID or -1 if FP. Using float as missing values will be converted to NaN anyways.
                ('HId', pd.Series(dtype=object)),
                # Hypothesis ID or NaN if MISS. Using float as missing values will be converted to NaN anyways.
                ('D', pd.Series(dtype=float)),  # Distance or NaN when FP or MISS
                ('mov_Pred', pd.Series(dtype=object)),   # Result of the movement classifier 
                ('vel_Pred', pd.Series(dtype=object)),   # Prediction of velocity from tracker 
                ('mov_GT', pd.Series(dtype=object)) # GT movement state
            ]),
            index=idx
        )
        return df

    @property
    def events(self):
        if self.dirty_events:
            if hasattr(self, 'cached_events_df') and self.cached_events_df is not None:
                # Save the movement classification results before regenerating
                old_df = self.cached_events_df
                has_movement_data = 'mov_Pred' in old_df.columns and old_df['mov_Pred'].notna().any()
                has_vel_data = 'vel_Pred' in old_df.columns and old_df['vel_Pred'].notna().any()

                # Generate new DataFrame
                self.cached_events_df = MOTAccumulatorCustom.new_event_dataframe_with_data(self._indices, self._events)
                
                # Restore movement classification results if they existed
                if has_movement_data:
                    # Create a mapping from old indices to movement results
                    movement_Pred_map = {}
                    movement_GT_map = {}
                    vel_Pred_map = {}

                    for idx, value in old_df['mov_Pred'].items():
                        if pd.notna(value):
                            movement_Pred_map[idx] = value

                    for idx, value in old_df['vel_Pred'].items():
                        if pd.notna(value):
                            vel_Pred_map[idx] = value

                    for idx, value in old_df['mov_GT'].items():
                        if pd.notna(value):
                            movement_GT_map[idx] = value
                    
                    # Apply the saved values to the new DataFrame
                    for idx, value in movement_Pred_map.items():
                        if idx in self.cached_events_df.index:
                            self.cached_events_df.at[idx, 'mov_Pred'] = value
                    
                    for idx, value in vel_Pred_map.items():
                        if idx in self.cached_events_df.index:
                            self.cached_events_df.at[idx, 'vel_Pred'] = value

                    for idx, value in movement_GT_map.items():
                        if idx in self.cached_events_df.index:
                            self.cached_events_df.at[idx, 'mov_GT'] = value
            else:
                # First time, just create the DataFrame
                self.cached_events_df = MOTAccumulatorCustom.new_event_dataframe_with_data(self._indices, self._events)
                
            self.dirty_events = False
        return self.cached_events_df
    
    @staticmethod
    def merge_event_dataframes(dfs, update_frame_indices=True, update_oids=True, update_hids=True,
                               return_mappings=False):
        """Merge dataframes.

        Params
        ------
        dfs : list of pandas.DataFrame or MotAccumulator
            A list of event containers to merge

        Kwargs
        ------
        update_frame_indices : boolean, optional
            Ensure that frame indices are unique in the merged container
        update_oids : boolean, unique
            Ensure that object ids are unique in the merged container
        update_hids : boolean, unique
            Ensure that hypothesis ids are unique in the merged container
        return_mappings : boolean, unique
            Whether or not to return mapping information

        Returns
        -------
        df : pandas.DataFrame
            Merged event data frame
        """

        mapping_infos = []
        new_oid = count()
        new_hid = count()

        r = MOTAccumulatorCustom.new_event_dataframe()
        for df in dfs:

            if isinstance(df, MOTAccumulatorCustom):
                df = df.events

            copy = df.copy()
            infos = {}

            # Update index
            if update_frame_indices:
                next_frame_id = max(r.index.get_level_values(0).max() + 1,
                                    r.index.get_level_values(0).unique().shape[0])
                if np.isnan(next_frame_id):
                    next_frame_id = 0
                copy.index = copy.index.map(lambda x: (x[0] + next_frame_id, x[1]))
                infos['frame_offset'] = next_frame_id

            # Update object / hypothesis ids
            if update_oids:
                oid_map = dict([oid, str(next(new_oid))] for oid in copy['OId'].dropna().unique())
                copy['OId'] = copy['OId'].map(lambda x: oid_map[x], na_action='ignore')
                infos['oid_map'] = oid_map

            if update_hids:
                hid_map = dict([hid, str(next(new_hid))] for hid in copy['HId'].dropna().unique())
                copy['HId'] = copy['HId'].map(lambda x: hid_map[x], na_action='ignore')
                infos['hid_map'] = hid_map

            r = r.append(copy)
            mapping_infos.append(infos)

        if return_mappings:
            return r, mapping_infos
        else:
            return r
