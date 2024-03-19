import os
import random
import re
from PIL import ImageFile
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms._transforms_video as transforms_video
from transformers import CLIPTokenizer
import numpy as np
from decord import VideoReader, cpu

""" VideoFrameDataset """

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    '''
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')
    '''
    Im = Image.open(path)
    return Im.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    """
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
    """
    return pil_loader(path)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def find_classes(dir):
    assert (os.path.exists(dir)), f'{dir} does not exist'
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def class_name_to_idx(annotation_dir):
    """
    return class indices from 0 ~ num_classes-1
    """
    fpath = os.path.join(annotation_dir, "classInd.txt")
    with open(fpath, "r") as f:
        data = f.readlines()
        class_to_idx = {x.strip().split(" ")[1].lower(): int(x.strip().split(" ")[0]) - 1 for x in data}
    return class_to_idx


def split_by_captical(s):
    s_list = re.sub(r"([A-Z])", r" \1", s).split()
    string = ""
    for s in s_list:
        string += s + " "
    return string.rstrip(" ").lower()


def _read_video(video_path, video_id, sample_frame_num, is_train=True):
    """
    read frames from long video
    args:
        video_id: str,
        sample_frame_num: frames used
    return:
        img_arrays: [num_frm, 3, H, W]
        chunk_mask: [num_frm, n_clip], , mask for indicating frames belong to each clip

    """

    video_path = os.path.join(video_path, video_id + '.mp4')
    vr = VideoReader(video_path, ctx=cpu(0))
    num_frame = len(vr)
    if is_train:
        interval = int(num_frame / (sample_frame_num - 1))
        start = np.random.randint(0, interval + 1)
        end = np.random.randint(num_frame - 1 - interval, num_frame)
        frame_idx = np.linspace(start, end, sample_frame_num).astype(int)
    else:
        frame_idx = np.linspace(0, num_frame - 1, sample_frame_num).astype(int)

    img_arrays = vr.get_batch(frame_idx)

    img_arrays = img_arrays.float() / 255

    img_arrays = img_arrays.permute(0, 3, 1, 2)  # N,C,H,W
    vr.save_frames(video_path, video_id, img_arrays)

    return img_arrays

def tokenize(tokenizer, texts, total_chunk=8, max_length=50, use_split=True, **kwargs):
    '''
    tokenizing text for pretraining
    args:
        tokenizer: for tokenize texts
        texts: list of text segment
        total_chunk: num of text segments
        use_split: whether split the texts
    return:
        text_ids: sequence of token id
        attention_mask: segment_ids to distinguish the sentences
        chunk: index of [CLS]

    '''
    if use_split:

        def merge(texts, tolen=8):
            if len(texts) <= tolen:
                return texts
            else:
                while len(texts) > tolen:
                    texts_2g = [len(texts[i]) + len(texts[i + 1]) for i in range(len(texts) - 1)]
                    min_index = texts_2g.index(min(texts_2g))
                    texts_group = []
                    for i in range(len(texts)):
                        if i != min_index and i != min_index + 1:
                            texts_group.append(texts[i])
                        elif i == min_index:
                            texts_group.append(' '.join(texts[i:i + 2]))
                        else:
                            continue
                    texts = texts_group
                return texts

        if len(texts) > total_chunk:
            texts = merge(texts, tolen=total_chunk)

        encoded = [tokenizer(x, padding='max_length', truncation=True, max_length=max_length) for x in texts]

        text_ids = [x.input_ids for x in encoded]
        attention_mask = [x.attention_mask for x in encoded]

        if len(texts) < total_chunk:
            for i in range(total_chunk - len(texts)):
                text_ids.append([0 for x in range(max_length)])
                attention_mask.append([0 for x in range(max_length)])

    else:
        texts = ' '.join(texts)
        encoded = [tokenizer(x, padding='max_length', truncation=True, max_length=max_length) for x in [texts]]

        text_ids = [x.input_ids for x in encoded]
        attention_mask = [x.attention_mask for x in encoded]

    return text_ids, attention_mask


def make_dataset(dir, nframes, texts, frame_stride=1, clip_step=None,
                        tokenizer=None, total_chunk=8, max_length=50, use_split=True, **kwargs):
    """
        Load videos from MSR-VTT or activityNet
        assert videos are saved in first-level directory:
            dir:
                videoxxx1
                    frame1.jpg
                    frame2.jpg
                videoxxx2
        """
    if clip_step is None:
        # consecutive clips with no frame overlap
        clip_step = nframes
    # make videos
    clips = []  # 2d list
    videos = []  # 2d list
    for video_name in sorted(os.listdir(dir)):
        if video_name != '_broken_clips':
            video_path = os.path.join(dir, video_name)
            assert (os.path.isdir(video_path))

            frames = []
            for i, fname in enumerate(sorted(os.listdir(video_path))):
                assert (is_image_file(fname)), f'fname={fname},video_path={video_path},dir={dir}'

                # get frame info
                img_path = os.path.join(video_path, fname)
                class_name = tokenize(tokenizer=tokenizer, texts=texts, total_chunk=total_chunk, max_length=max_length,
                                      use_split=use_split)
                frame_info = {
                    "img_path": img_path,
                    "class_index": class_name,
                    "class_name": texts,
                    "class_caption": texts  # boxing speed bag
                }
                frames.append(frame_info)

            # make videos
            if len(frames) >= nframes:
                videos.append(frames)

            # make clips
            frames = frames[::frame_stride]
            start_indices = list(range(len(frames)))[::clip_step]
            for i in start_indices:
                clip = frames[i:i + nframes]
                if len(clip) == nframes:
                    clips.append(clip)
    return clips, videos


def make_dataset_ucf(dir, nframes, class_to_idx, frame_stride=1, clip_step=None):
    """
    Load consecutive clips and consecutive frames from `dir`.

    args:
        nframes: num of frames of every video clips
        class_to_idx: for mapping video name to video id
        frame_stride: select frames with a stride.
        clip_step: select clips with a step. if clip_step< nframes, 
            there will be overlapped frames among two consecutive clips.

    assert videos are saved in first-level directory:
        dir:
            videoxxx1
                frame1.jpg
                frame2.jpg
            videoxxx2
    """
    if clip_step is None:
        # consecutive clips with no frame overlap
        clip_step = nframes
    # make videos
    clips = []  # 2d list
    videos = []  # 2d list
    for video_name in sorted(os.listdir(dir)):
        if video_name != '_broken_clips':
            video_path = os.path.join(dir, video_name)
            assert (os.path.isdir(video_path))

            frames = []
            for i, fname in enumerate(sorted(os.listdir(video_path))):
                assert (is_image_file(fname)), f'fname={fname},video_path={video_path},dir={dir}'

                # get frame info
                img_path = os.path.join(video_path, fname)
                class_name = video_name.split("_")[1].lower()  # v_BoxingSpeedBag_g12_c05 -> boxingspeedbag
                class_caption = split_by_captical(
                    video_name.split("_")[1])  # v_BoxingSpeedBag_g12_c05 -> BoxingSpeedBag -> boxing speed bag
                frame_info = {
                    "img_path": img_path,
                    "class_index": class_to_idx[class_name],
                    "class_name": class_name,  # boxingspeedbag
                    "class_caption": class_caption  # boxing speed bag
                }
                frames.append(frame_info)

            # make videos
            if len(frames) >= nframes:
                videos.append(frames)

            # make clips
            frames = frames[::frame_stride]
            start_indices = list(range(len(frames)))[::clip_step]
            for i in start_indices:
                clip = frames[i:i + nframes]
                if len(clip) == nframes:
                    clips.append(clip)
    return clips, videos


def load_and_transform_frames(frame_list, loader, img_transform=None):
    assert (isinstance(frame_list, list))
    clip = []
    labels = []
    for frame in frame_list:

        if isinstance(frame, tuple):
            fpath, label = frame
        elif isinstance(frame, dict):
            fpath = frame["img_path"]
            label = {
                "class_index": frame["class_index"],
                "class_name": frame["class_name"],
                "class_caption": frame["class_caption"],
            }

        labels.append(label)
        img = loader(fpath)
        if img_transform is not None:
            img = img_transform(img)
        img = img.view(img.size(0), 1, img.size(1), img.size(2))
        clip.append(img)
    return clip, labels[0]  # all frames have same label.


class VideoFrameDataset(data.Dataset):
    def __init__(self,
                 data_root,
                 resolution,
                 video_length,  # clip length
                 dataset_name="",
                 subset_split="",
                 annotation_dir=None,
                 spatial_transform="",
                 temporal_transform="",
                 frame_stride=1,
                 clip_step=None,
                 tokenizer=None,
                 total_chunk=0,
                 max_length=50,
                 use_split=True
                 ):

        self.loader = default_loader
        self.video_length = video_length
        self.subset_split = subset_split
        self.temporal_transform = temporal_transform
        self.spatial_transform = spatial_transform
        self.frame_stride = frame_stride
        self.dataset_name = dataset_name

        self.tokenizer = CLIPTokenizer.pre_trained('clip-vit-base-patch32')
        self.total_chunk = total_chunk
        self.max_length = max_length
        self.use_split = use_split

        assert (subset_split in ["train", "test", "all", ""])  # "" means no subset_split directory.
        assert (self.temporal_transform in ["", "rand_clips"])

        if subset_split == 'all':
            video_dir = os.path.join(data_root, "train")
        else:
            video_dir = os.path.join(data_root, subset_split)

        if dataset_name == 'UCF-101':
            if annotation_dir is None:
                annotation_dir = os.path.join(data_root, "ucfTrainTestlist")
            class_to_idx = class_name_to_idx(annotation_dir)
            assert (len(class_to_idx) == 101), f'num of classes = {len(class_to_idx)}, not 101'
        else:
            class_to_idx = None

        # make dataset
        if dataset_name == 'ucf101':
            self.clips, self.videos = make_dataset_ucf(video_dir, video_length, class_to_idx,
                                                       frame_stride=frame_stride, clip_step=clip_step)
        elif dataset_name == 'msrvtt':
            video_text = load_json_from_msrvtt(video_dir)
            self.clips, self.videos = make_dataset(video_dir, video_length, video_text,
                                                   frame_stride=frame_stride, clip_step=clip_step,
                                                   tokenizer=self.tokenizer, total_chunk=self.total_chunk,
                                                   max_length=self.max_length, use_split=self.use_split)
        elif dataset_name == 'activityNet':
            video_text = load_json_from_avtivityNet(video_dir)
            self.clips, self.videos = make_dataset(video_dir, video_length, video_text,
                                                   frame_stride=frame_stride, clip_step=clip_step,
                                                   tokenizer=self.tokenizer, total_chunk=self.total_chunk,
                                                   max_length=self.max_length, use_split=self.use_split)
        elif dataset_name == 'webvid':
            video_text = load_json_from_webvid(video_dir)
            self.clips, self.videos = make_dataset(video_dir, video_length, video_text,
                                                   frame_stride=frame_stride, clip_step=clip_step,
                                                   tokenizer=self.tokenizer, total_chunk=self.total_chunk,
                                                   max_length=self.max_length, use_split=self.use_split)
        else:
            assert "dataset name must be ucf101, or msrvtt, webvid, or activityNet"
        assert (len(self.clips[0]) == video_length), f"Invalid clip length = {len(self.clips[0])}"
        if self.temporal_transform == 'rand_clips':
            self.clips = self.videos

        # if subset_split == 'all':
        #     # add test videos
        #     video_dir = video_dir.rstrip('/train') + '/test'
        #     cs, vs = func(video_dir, video_length, class_to_idx)
        #     if self.temporal_transform == 'rand_clips':
        #         self.clips += vs
        #     else:
        #         self.clips += cs

        print('[VideoFrameDataset] number of videos:', len(self.videos))
        print('[VideoFrameDataset] number of clips', len(self.clips))

        # check data
        if len(self.clips) == 0:
            raise (RuntimeError(f"Found 0 clips in {video_dir}. \n"
                                "Supported image extensions are: " +
                                ",".join(IMG_EXTENSIONS)))

        # data transform
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        if self.spatial_transform == "center_crop_resize":
            print('Spatial transform: center crop and then resize')
            self.video_transform = transforms.Compose([
                transforms.Resize(resolution),
                transforms_video.CenterCropVideo(resolution),
            ])
        elif self.spatial_transform == "resize":
            print('Spatial transform: resize with no crop')
            self.video_transform = transforms.Resize((resolution, resolution))
        elif self.spatial_transform == "random_crop":
            self.video_transform = transforms.Compose([
                transforms_video.RandomCropVideo(resolution),
            ])
        elif self.spatial_transform == "":
            self.video_transform = None
        else:
            raise NotImplementedError

    def __getitem__(self, index):
        # get clip info
        if self.temporal_transform == 'rand_clips':
            raw_video = self.clips[index]
            rand_idx = random.randint(0, len(raw_video) - self.video_length)
            clip = raw_video[rand_idx:rand_idx + self.video_length]
        else:
            clip = self.clips[index]
        assert (
                    len(clip) == self.video_length), f'current clip_length={len(clip)}, target clip_length={self.video_length}, {clip}'

        # make clip tensor
        frames, labels = load_and_transform_frames(clip, self.loader, self.img_transform)
        assert (
                    len(frames) == self.video_length), f'current clip_length={len(frames)}, target clip_length={self.video_length}, {clip}'
        frames = torch.cat(frames, 1)  # c,t,h,w
        if self.video_transform is not None:
            frames = self.video_transform(frames)

        example = dict()
        example["image"] = frames
        if labels is not None and self.dataset_name == 'UCF-101':
            example["caption"] = labels["class_caption"]
            example["class_label"] = labels["class_index"]
            example["class_name"] = labels["class_name"]
        example["frame_stride"] = self.frame_stride
        return example

    def __len__(self):
        return len(self.clips)
