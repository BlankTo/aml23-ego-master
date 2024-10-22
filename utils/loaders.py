import os
import glob
import torch
import os.path
import numpy as np
from abc import ABC
import pandas as pd
from PIL import Image
import random as rand
import torch.utils.data as data

from utils.logger import logger
from .epic_record import EpicVideoRecord
from .emg_record import ActionEMGRecord
from .spec_record import ActionEMGspecRecord


class EpicKitchensDataset(data.Dataset, ABC):
    def __init__(self, split, modalities, mode, dataset_conf, num_frames_per_clip, num_clips, dense_sampling,
                 transform=None, load_feat=False, additional_info=False, **kwargs):
        """
        split: str (D1, D2 or D3)
        modalities: list(str, str, ...)
        mode: str (train, test/val)
        dataset_conf must contain the following:
            - annotations_path: str
            - stride: int
        dataset_conf[modality] for the modalities used must contain:
            - data_path: str
            - tmpl: str
            - features_name: str (in case you are loading features for a predefined modality)
            - (Event only) rgb4e: int
        num_frames_per_clip: dict(modality: int)
        num_clips: int
        dense_sampling: dict(modality: bool)
        additional_info: bool, set to True if you want to receive also the uid and the video name from the get function
            notice, this may be useful to do some proper visualizations!
        """
        self.modalities = modalities  # considered modalities (ex. [RGB, Flow, Spec, Event])
        self.mode = mode  # 'train', 'val' or 'test'
        self.dataset_conf = dataset_conf
        self.num_frames_per_clip = num_frames_per_clip
        self.dense_sampling = dense_sampling
        self.num_clips = num_clips
        self.stride = self.dataset_conf.stride
        self.additional_info = additional_info

        if self.mode == "train":
            pickle_name = split + "_train.pkl"
        elif kwargs.get('save', None) is not None:
            pickle_name = split + "_" + kwargs["save"] + ".pkl"
        else:
            pickle_name = split + "_test.pkl"

        self.list_file = pd.read_pickle(os.path.join(self.dataset_conf.annotations_path, pickle_name))
        logger.info(f"Dataloader for {split}-{self.mode} with {len(self.list_file)} samples generated")

        self.video_list = [EpicVideoRecord(tup, self.dataset_conf) for tup in self.list_file.iterrows()]
        self.transform = transform  # pipeline of transforms
        self.load_feat = load_feat

        if self.load_feat:
            logger.info(f"loading features")
            self.model_features = None
            for m in self.modalities:
                # load features for each modality
                model_features = pd.DataFrame(pd.read_pickle(os.path.join("saved_features", f"RGB_{self.num_frames_per_clip}_{'dense' if self.dense_sampling else 'uniform'}_{pickle_name}"))['features'])[["uid", "features_" + m]]                
                if self.model_features is None:
                    self.model_features = model_features
                else:
                    self.model_features = pd.merge(self.model_features, model_features, how="inner", on="uid")

            self.model_features = pd.merge(self.model_features, self.list_file, how="inner", on="uid")
        
        else: print('no load feat')

        logger.info('-------------------------------------------------------------------------------')
        
        for name, value in {attr: getattr(self, attr) for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")}.items():
            if (name != 'video_list') and ('_' not in name):
                logger.info(f"{name}: {value}")

        logger.info('-------------------------------------------------------------------------------')

    def _get_train_indices(self, record, modality='RGB'):
        
        #print('get_train_indices')
        
        ##################################################################
        # DONE: implement sampling for training mode                     #
        # Give the record and the modality, this function should return  #
        # a list of integers representing the frames to be selected from #
        # the video clip.                                                #
        # Remember that the returned array should have size              #
        #           num_clip x num_frames_per_clip                       #
        ##################################################################

        ## here the implementation

        num_frames_per_clip = self.num_frames_per_clip[modality]
        num_clips = self.num_clips
        dense_stride = self.stride
        duration = record.num_frames[modality]

        indices = []
        central_frames = []

        if self.dense_sampling[modality]: # dense sampling

            if duration < num_frames_per_clip: raise ValueError(f"video too short: {duration}")

            if duration < num_frames_per_clip * dense_stride + num_clips:
                dense_stride = 1

            if num_clips <= duration - (num_frames_per_clip * dense_stride):
                clips_start = np.array(rand.sample(range(0, duration - ( num_frames_per_clip * dense_stride )), num_clips), dtype= int)
                clips_start.sort()
            else:
                print(f'very short video - zeros -> {num_clips} > {duration} - ({num_frames_per_clip} * {dense_stride})')
                clips_start = []
                for _ in range(num_clips):
                    clips_start.append(0)
                clips_start = np.array(clips_start)
            
            for clip_start in clips_start:
                frame_id = clip_start

                for i in range(num_frames_per_clip):
                    indices.append(frame_id)

                    if i == num_frames_per_clip // 2:
                        central_frames.append(record.start_frame + frame_id)

                    frame_id += dense_stride

        else: # Uniform sampling:

            if duration < num_frames_per_clip: raise ValueError("video too short")

            clip_len = int(max(num_frames_per_clip, duration / num_clips))

            clips_start = np.linspace(0, duration - clip_len, num= num_clips, dtype= int)

            for clip_start in clips_start:

                clip_indices = np.linspace(clip_start, clip_start + clip_len, num= num_frames_per_clip, dtype= int)
                indices.extend(clip_indices)
                central_frames.append(record.start_frame + clip_indices[len(clip_indices) // 2])

        return indices, central_frames

        #raise NotImplementedError("You should implement _get_train_indices")

    def _get_val_indices(self, record, modality):
        
        #print('get_val_indices')

        ##################################################################
        # DONE: implement sampling for testing mode                      #
        # Give the record and the modality, this function should return  #
        # a list of integers representing the frames to be selected from #
        # the video clip.                                                #
        # Remember that the returned array should have size              #
        #           num_clip x num_frames_per_clip                       #
        ##################################################################

        ## here the implementation

        num_frames_per_clip = self.num_frames_per_clip[modality]
        num_clips = self.num_clips
        dense_stride = self.stride
        duration = record.num_frames[modality]

        indices = []
        central_frames = []

        if self.dense_sampling[modality]: # dense sampling

            if duration < num_frames_per_clip: raise ValueError(f"video too short: {duration}")

            if duration < num_frames_per_clip * dense_stride:
                dense_stride = 1

            clips_start = np.linspace(0, duration - ( num_frames_per_clip * dense_stride ), num= num_clips + 2, dtype= int)[1:-1]
            
            for clip_start in clips_start:
                frame_id = clip_start

                for i in range(num_frames_per_clip):
                    indices.append(frame_id)

                    if i == num_frames_per_clip // 2:
                        central_frames.append(record.start_frame + frame_id)

                    frame_id += dense_stride


        else: # Uniform sampling:

            if duration < num_frames_per_clip: raise ValueError("video too short")

            clip_len = int(max(num_frames_per_clip, duration / num_clips))

            clips_start = np.linspace(0, duration - clip_len, num= num_clips, dtype= int)

            for clip_start in clips_start:

                clip_indices = np.linspace(clip_start, clip_start + clip_len, num= num_frames_per_clip, dtype= int)
                indices.extend(clip_indices)
                central_frames.append(record.start_frame + clip_indices[len(clip_indices) // 2])

        return indices, central_frames

        #raise NotImplementedError("You should implement _get_train_indices")

    def __getitem__(self, index):

        frames = {}
        label = None
        # record is a row of the pkl file containing one sample/action
        # notice that it is already converted into a EpicVideoRecord object so that here you can access
        # all the properties of the sample easily
        record = self.video_list[index]

        if self.load_feat:
            sample = {}
            sample_row = self.model_features[self.model_features["uid"] == int(record.uid)]
            assert len(sample_row) == 1
            for m in self.modalities:
                sample[m] = sample_row["features_" + m].values[0]
                #print(':::::::::::::::::::::::::::::::::::::: self.LOAD_FEAT')
            
            if self.additional_info:
                return sample, record.label, record.untrimmed_video_name, record.uid, len(sample[m])
            else:
                return sample, record.label

        segment_indices = {}
        # notice that all indexes are sampled in the[0, sample_{num_frames}] range, then the start_index of the sample
        # is added as an offset
        for modality in self.modalities:
            if self.mode == "train":
                # here the training indexes are obtained with some randomization
                segment_indices[modality], central_frames = self._get_train_indices(record, modality)
            else:
                # here the testing indexes are obtained with no randomization, i.e., centered
                segment_indices[modality], central_frames = self._get_val_indices(record, modality)

        for m in self.modalities:
            img, label = self.get(m, record, segment_indices[m])
            frames[m] = img

        if self.additional_info:
            return frames, label, record.untrimmed_video_name, record.uid, central_frames
        else:
            return frames, label

    def get(self, modality, record, indices):
        images = list()
        for frame_index in indices:
            p = int(frame_index)
            error = True
            last = False
            while error:
                error = False
                try:
                    frame = self._load_data(modality, record, p)
                except FileNotFoundError as e: # some frames are missing in the datasets, skipping to next frame
                    #print(f'error in get - {(record.uid, p, record.num_frames[modality])}')
                    if not last:
                        if p == record.num_frames[modality]:
                            last = True
                    if last: p -= 1
                    else: p += 1
                    error = True

            images.extend(frame)
        # finally, all the transformations are applied
        process_data = self.transform[modality](images)
        return process_data, record.label

    def _load_data(self, modality, record, idx):
        data_path = self.dataset_conf[modality].data_path

        tmpl = self.dataset_conf[modality].tmpl

        if modality == 'RGB' or modality == 'RGBDiff':

            # here the offset for the starting index of the sample is added
            idx_untrimmed = record.start_frame + idx

            try:
                img = Image.open(os.path.join(data_path, record.untrimmed_video_name, record.untrimmed_video_name, tmpl.format(idx_untrimmed))).convert('RGB')
            except FileNotFoundError:
                #print("Img not found")
                max_idx_video = int(sorted(glob.glob(os.path.join(data_path, record.untrimmed_video_name, record.untrimmed_video_name, "img_*")))[-1].split("_")[-1].split(".")[0])
                if idx_untrimmed > max_idx_video:
                    img = Image.open(os.path.join(data_path, record.untrimmed_video_name, record.untrimmed_video_name, tmpl.format(max_idx_video))).convert('RGB')
                else:
                    raise FileNotFoundError
            return [img]
        
        else:
            raise NotImplementedError("Modality not implemented")

    def __len__(self):
        return len(self.video_list)

class ActionNetDataset(data.Dataset, ABC):
    def __init__(self, mode, dataset_conf, **kwargs):

        """
        mode: str (train, test/val)
        dataset_conf must contain the following:
            - annotations_path: str
            - stride: int
        dataset_conf[modality] for the modalities used must contain:
            - data_path: str
            - tmpl: str
            - features_name: str (in case you are loading features for a predefined modality)
            - (Event only) rgb4e: int
        """

        self.mode = mode  # 'train', 'val' or 'test'
        self.dataset_conf = dataset_conf

        if self.mode == "train": emg_name = f"action_{dataset_conf.emg_clip_duration}s_EMG_train.pkl"
        else: emg_name = f"action_{dataset_conf.emg_clip_duration}s_EMG_test.pkl"

        self.list_file = pd.read_pickle(os.path.join('saved_features', emg_name))
        logger.info(f"Dataloader for {self.mode} with {len(self.list_file)} samples generated")

        self.emg_list = [ActionEMGRecord(self.list_file.iloc[i], self.dataset_conf) for i in range(len(self.list_file))]

    def __getitem__(self, index):

        record_emg = self.emg_list[index]
        return {"EMG": torch.cat((record_emg.myo_left_readings, record_emg.myo_right_readings), dim= 1)}, torch.tensor(record_emg.label)

    def __len__(self):
        return len(self.emg_list)
    
class ActionNetDataset_100(data.Dataset, ABC):
    def __init__(self, mode, dataset_conf, **kwargs):

        """
        mode: str (train, test/val)
        dataset_conf must contain the following:
            - annotations_path: str
            - stride: int
        dataset_conf[modality] for the modalities used must contain:
            - data_path: str
            - tmpl: str
            - features_name: str (in case you are loading features for a predefined modality)
            - (Event only) rgb4e: int
        """

        self.mode = mode  # 'train', 'val' or 'test'
        self.dataset_conf = dataset_conf

        if self.mode == "train": emg_name = f"action_{dataset_conf.emg_clip_duration}s_100_EMG_train.pkl"
        else: emg_name = f"action_{dataset_conf.emg_clip_duration}s_100_EMG_test.pkl"

        self.list_file = pd.read_pickle(os.path.join('saved_features', emg_name))
        logger.info(f"Dataloader for {self.mode} with {len(self.list_file)} samples generated")

        self.emg_list = [ActionEMGRecord(self.list_file.iloc[i], self.dataset_conf) for i in range(len(self.list_file))]

    def __getitem__(self, index):

        record_emg = self.emg_list[index]
        return {"EMG": torch.cat((record_emg.myo_left_readings, record_emg.myo_right_readings), dim= 1)}, torch.tensor(record_emg.label)

    def __len__(self):
        return len(self.emg_list)
    

class ActionNetSpectrogramDataset(data.Dataset, ABC):
    def __init__(self, mode, dataset_conf, **kwargs):

        """
        mode: str (train, test/val)
        dataset_conf must contain the following:
            - annotations_path: str
            - stride: int
        dataset_conf[modality] for the modalities used must contain:
            - data_path: str
            - tmpl: str
            - features_name: str (in case you are loading features for a predefined modality)
            - (Event only) rgb4e: int
        """
        self.mode = mode  # 'train', 'val' or 'test'
        self.dataset_conf = dataset_conf

        if self.mode == "train": spec_name = f"action_{dataset_conf.emg_clip_duration}s_EMGspec_train.pkl"
        else: spec_name = f"action_{dataset_conf.emg_clip_duration}s_EMGspec_test.pkl"

        self.list_file = pd.read_pickle(os.path.join('saved_features', spec_name))
        logger.info(f"Dataloader for {self.mode} with {len(self.list_file)} samples generated")

        self.spec_list = [ActionEMGspecRecord(self.list_file.iloc[i], self.dataset_conf) for i in range(len(self.list_file))]

    def __getitem__(self, index):

        record_spec = self.spec_list[index]
        return {"spec": torch.cat((record_spec.left_spectrogram, record_spec.right_spectrogram), dim=0)}, torch.tensor(record_spec.label)

    def __len__(self):
        return len(self.spec_list)
