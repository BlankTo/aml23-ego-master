import glob
import random
from abc import ABC
import pandas as pd
from .epic_record import EpicVideoRecord
import torch.utils.data as data
from PIL import Image
import os
import os.path
from utils.logger import logger
import math

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
            self.model_features = None
            for m in self.modalities:
                # load features for each modality
                model_features = pd.DataFrame(pd.read_pickle(os.path.join("saved_features",
                                                                          self.dataset_conf[m].features_name + "_" +
                                                                          pickle_name))['features'])[["uid", "features_" + m]]
                if self.model_features is None:
                    self.model_features = model_features
                else:
                    self.model_features = pd.merge(self.model_features, model_features, how="inner", on="uid")

            self.model_features = pd.merge(self.model_features, self.list_file, how="inner", on="uid")

        logger.info('-------------------------------------------------------------------------------')
        
        for name, value in {attr: getattr(self, attr) for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")}.items():
            if (name != 'video_list') and ('_' not in name):
                logger.info(f"{name}: {value}")

        logger.info('-------------------------------------------------------------------------------')

    def _get_train_indices(self, record, modality='RGB'):
        ##################################################################
        # TODO: implement sampling for training mode                     #
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

        duration = record.num_frames

        indices = []

        if self.dense_sampling[modality]: # dense sampling

            #print('---------------------------- video duration ----------------------------')
            #print(duration[modality])

            dense_clip_length = num_frames_per_clip * dense_stride

            if duration[modality] < num_frames_per_clip: raise ValueError("video length must be greater than or equal to num_frames_per_clip")

            if duration[modality] < dense_clip_length * num_clips:

                dense_clip_length = num_frames_per_clip
                dense_stride = 1

            if duration[modality] <= (((num_clips - 1) * dense_clip_length) + (num_clips * (dense_clip_length - 1))):
                    
                    #print('no rand')
    
                    step = max(1, math.ceil((duration[modality] - dense_clip_length) / (num_clips - 1)))
                    idxs = [i * step for i in range(num_clips)]
                    clips_start = [min(idx, duration[modality] - dense_clip_length) for idx in idxs]

            else:

                #print('rand')

                clips_start = []

                for i in range(num_clips):
                    ok = False
                    while not ok:
                        clip_start = random.randint(0, duration[modality] - 1 - dense_clip_length)
                        ok = True
                        for other_clip_start in clips_start:
                            if (clip_start >= other_clip_start and clip_start <= other_clip_start + dense_clip_length) or (clip_start + dense_clip_length >= other_clip_start and clip_start + dense_clip_length <= other_clip_start + dense_clip_length):
                                ok = False
                                break
                    clips_start.append(clip_start)

            clips_start.sort()

            #print(clips_start)
            
            for clip_start in clips_start:
                frame_id = clip_start

                for _ in range(num_frames_per_clip):
                    indices.append(frame_id)
                    frame_id += dense_stride
            
            #print(indices)

        else: # Uniform sampling:

            #print('---------------------------- video duration ----------------------------')
            #print(duration[modality])

            clips_start = []

            if duration[modality] < 2 * num_frames_per_clip * num_clips:

                #print('no stride')

                for clip_start in range(0, duration[modality] - num_frames_per_clip, math.ceil((duration[modality] - num_frames_per_clip) / num_clips)):

                    clips_start.append(clip_start)
                    
                    for frame_id in range(clip_start, clip_start + num_frames_per_clip):

                        indices.append(frame_id)

            else:

                uniform_clip_length = math.floor(duration[modality] / num_clips)

                #print(f'uniform stride of {uniform_clip_length / num_frames_per_clip}')

                clip_start = 0
                for _ in range(num_clips):

                    clips_start.append(clip_start)

                    frame_id = clip_start

                    uniform_stride = math.floor(uniform_clip_length / num_frames_per_clip)

                    for _ in range(num_frames_per_clip):

                        indices.append(frame_id)

                        frame_id += uniform_stride

                    clip_start += uniform_clip_length

        #print(clips_start)

        return indices

        #raise NotImplementedError("You should implement _get_train_indices")

    def _get_val_indices(self, record, modality):

        ##################################################################
        # TODO: implement sampling for testing mode                      #
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

        duration = record.num_frames

        indices = []

        if self.dense_sampling[modality]: # dense sampling

            #print('---------------------------- video duration ----------------------------')
            #print(duration[modality])

            dense_clip_length = num_frames_per_clip * dense_stride

            if duration[modality] < num_frames_per_clip: raise ValueError("video length must be greater than or equal to num_frames_per_clip")

            if duration[modality] < dense_clip_length * num_clips:

                dense_clip_length = num_frames_per_clip
                dense_stride = 1
    
            step = max(1, math.ceil((duration[modality] - dense_clip_length) / (num_clips - 1)))
            idxs = [i * step for i in range(num_clips)]
            clips_start = [min(idx, duration[modality] - dense_clip_length) for idx in idxs]
            
            for clip_start in clips_start:
                frame_id = clip_start

                for _ in range(num_frames_per_clip):
                    indices.append(frame_id)
                    frame_id += dense_stride

            #print(clips_start)

        else: # Uniform sampling:

            #print('---------------------------- video duration ----------------------------')
            #print(duration[modality])

            clips_start = []

            if duration[modality] < 2 * num_frames_per_clip * num_clips:

                #print('no stride')

                for clip_start in range(0, duration[modality] - num_frames_per_clip, math.ceil((duration[modality] - num_frames_per_clip) / num_clips)):

                    clips_start.append(clip_start)
                    
                    for frame_id in range(clip_start, clip_start + num_frames_per_clip):

                        indices.append(frame_id)

            else:

                uniform_clip_length = math.floor(duration[modality] / num_clips)

                #print(f'uniform stride of {uniform_clip_length / num_frames_per_clip}')

                clip_start = 0
                for _ in range(num_clips):

                    clips_start.append(clip_start)

                    frame_id = clip_start

                    uniform_stride = math.floor(uniform_clip_length / num_frames_per_clip)

                    for _ in range(num_frames_per_clip):

                        indices.append(frame_id)

                        frame_id += uniform_stride

                    clip_start += uniform_clip_length

        #print(clips_start)

        return indices

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
            if self.additional_info:
                return sample, record.label, record.untrimmed_video_name, record.uid
            else:
                return sample, record.label

        segment_indices = {}
        # notice that all indexes are sampled in the[0, sample_{num_frames}] range, then the start_index of the sample
        # is added as an offset
        for modality in self.modalities:
            #logger.info(f'MODALITY: {modality}')
            if self.mode == "train":
                # here the training indexes are obtained with some randomization
                segment_indices[modality] = self._get_train_indices(record, modality)
            else:
                # here the testing indexes are obtained with no randomization, i.e., centered
                segment_indices[modality] = self._get_val_indices(record, modality)

        for m in self.modalities:
            img, label = self.get(m, record, segment_indices[m])
            frames[m] = img

        if self.additional_info:
            return frames, label, record.untrimmed_video_name, record.uid
        else:
            return frames, label

    def get(self, modality, record, indices):
        images = list()
        for frame_index in indices:
            p = int(frame_index)
            # here the frame is loaded in memory
            error = True
            last = False
            #print('---')
            while error:
                error = False
                #print(f'this is visible - {frame_index}')
                try:
                    frame = self._load_data(modality, record, p)
                except FileNotFoundError as e:
                    if not last:
                        if p == record.num_frames[modality]:
                            last = True
                    if last: p -= 1
                    else: p += 1
                    error = True

            #print('this is not')
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
