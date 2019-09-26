import os
import math
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from GAWWN.tools.config import cfg

def get_img_locs(img_path, parts, imsize, transform=None, normalize=None):
    img = Image.open(img_path).convert('RGB')
    load_size = cfg.IMAGE.LOADSIZE
    width, height = img.size
    
    # scale to (load_size * load_size) 
    t_img = img.resize((load_size, load_size))
    factor_x = load_size / width
    factor_y = load_size / height
    for i in range(len(parts)):
        parts[i][0] = max(1, math.floor(factor_x * parts[i][0]))
        parts[i][1] = max(1, math.floor(factor_y * parts[i][1]))
    w1 = math.ceil(np.random.uniform(1e-2, load_size - imsize))
    h1 = math.ceil(np.random.uniform(1e-2, load_size - imsize))
    # crop to (imsize * imsize)
    img = t_img.crop((w1, h1, w1 + imsize, h1 + imsize))

    flip = np.random.uniform() > 0.5
    if flip:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    num_elt = cfg.KEYPOINT.NUM_ELT
    keypoint_dim = cfg.KEYPOINT.DIM
    locs = torch.zeros((num_elt, keypoint_dim, keypoint_dim))
   
    for i in range(len(parts)):
        parts[i][0] = max(1, parts[i][0] - w1)
        parts[i][1] = max(1, parts[i][1] - h1)
        if flip:
            parts[i][0] = max(1, imsize - parts[i][0] + 1)
        if parts[i][2] > 0.1:
            x = min(keypoint_dim - 1, round(parts[i][0] * keypoint_dim / imsize))
            y = min(keypoint_dim - 1, round(parts[i][1] * keypoint_dim / imsize))
            locs[i][int(y)][int(x)] = 1
    
    if transform is not None:
        img = transform(img)
    if normalize is not None:
        img = normalize(img)

    return img * 2 - 1, locs


class ImageTextLocDataset(data.Dataset):
    def __init__(self, data_path, split = "all",
                    transfrom = None, target_transform = None):
        self.transfrom = transfrom
        self.target_transform = target_transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.embedding_num = cfg.TEXT.CAPTIONS_PER_IMAGE
        self.data_path = data_path
        self.imsize = cfg.IMAGE.FINESIZE
        # for i in range(cfg.TREE.BRANCH_NUM):
        #     self.imsize.append(base_size)
        #     base_size = base_size * 2

        self.char2idx, self.idx2char = self.makeDict()
        
        self.idxs, self.idx2filename, self.images, \
            self.part_locs, self.captions, self.txt_vecs = self.load_data(data_path)
        self.train_idxs, self.test_idxs = train_test_split(self.idxs, test_size=0.1, random_state=5)

        if split == "train":
            self.idxs = self.train_idxs
        elif split == "test":
            self.idxs = self.test_idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, index):
        idx = self.idxs[index] - 1
        filename = self.idx2filename[idx + 1]
        img = self.images[idx]
        part_locs = self.part_locs[idx]
        no = np.random.randint(0, cfg.TEXT.CAPTIONS_PER_IMAGE)
        txt_vec = self.txt_vecs[idx][no]
        cap = self.get_captions(index, no)
        return img, txt_vec, part_locs, filename, cap

    def get_captions(self, index, no):
        idx = self.idxs[index] - 1
        captions = self.captions[idx]
        cap = captions[no]
        txt = ""
        for c in cap:
            if c == 0:
                break
            txt += self.idx2char[c]
        return txt

    def load_data(self, data_path):
        all_data = torch.load(os.path.join(data_path, "data.pth"))
        filepath = os.path.join(data_path, "images.txt")
        idx2filename = dict()
        with open(filepath, "r") as f:
            for line in f:
                words = line.split()
                idx2filename[int(words[0])] = words[1][:-4]
        idxs = [i for i in range(1, len(idx2filename) + 1)]
        part_locs = []
        captions = []
        images = []
        txt_vecs = []
        for idx in idxs:
            filename = idx2filename[idx]
            info = all_data[filename.split('/')[1]]
            img_path = os.path.join(data_path, "images", filename + ".jpg")
            img, locs = get_img_locs(img_path, info["parts"], self.imsize, self.transfrom, self.norm)
            txt_vec = torch.from_numpy(info["txt"])
            cap = info["char"].T
            part_locs.append(locs)
            txt_vecs.append(txt_vec)
            images.append(img)
            captions.append(cap)
        return idxs, idx2filename, images, part_locs, captions, txt_vecs

    def makeDict(self):
        alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "
        char2idx = {}
        idx2char = {}
        for i in range(len(alphabet)):
            char2idx[alphabet[i]] = i + 1
            idx2char[i + 1] = alphabet[i] 
        return char2idx, idx2char
                            

        
