import logging
import typing as t

import numpy as np


# Label = namedtuple('LabelInfo', ['label', 'label_cn', 'keep', 'train_id', 'category_id', 'color'])
# fmt: off
ADE20K_SEM_SEG_CATEGORIES = [
    "wall",                                                     # 墙
    "building",                                                 # 建筑物
    "sky",                                                      # 天空
    "floor",                                                    # 地板
    "tree",                                                     # 树
    "ceiling",                                                  # 天花板
    "road, route",                                              # 道路
    "bed",                                                      # 床
    "window",                                                   # 窗户
    "grass",                                                    # 草
    "cabinet",                                                  # 柜子
    "sidewalk, pavement",                                       # 人行道                         
    "person",                                                   # 人
    "earth, ground",                                            # 地面  
    "door",                                                     # 门
    "table",                                                    # 桌子
    "mountain, mount",                                          # 山
    "plant",                                                    # 植物
    "curtain",                                                  # 窗帘                 
    "chair",                                                    # 椅子
    "car",                                                      # 车
    "water",                                                    # 水
    "painting, picture",                                        # 画
    "sofa",                                                     # 沙发
    "shelf",                                                    # 架子
    "house",                                                    # 房子
    "sea",                                                      # 海
    "mirror",                                                   # 镜子
    "rug",                                                      # 地毯
    "field",                                                    # 田野
    "armchair",                                                 # 扶手椅
    "seat",                                                     # 座位
    "fence",                                                    # 围栏
    "desk",                                                     # 书桌
    "rock, stone",                                              # 石头
    "wardrobe, closet, press",                                  # 衣柜
    "lamp",                                                     # 灯
    "tub",                                                      # 浴缸
    "rail",                                                     # 栏杆
    "cushion",                                                  # 枕头
    "base, pedestal, stand",                                    # 底座
    "box",                                                      # 盒子
    "column, pillar",                                           # 柱子
    "signboard, sign",                                          # 招牌
    "chest of drawers, chest, bureau, dresser",                 # 抽屉柜
    "counter",                                                  # 柜台
    "sand",                                                     # 沙
    "sink",                                                     # 水槽
    "skyscraper",                                               # 摩天大楼
    "fireplace",                                                # 壁炉
    "refrigerator, icebox",                                     # 冰箱
    "grandstand, covered stand",                                # 看台
    "path",                                                     # 路径
    "stairs",                                                   # 楼梯  
    "runway",                                                   # 跑道
    "case, display case, showcase, vitrine",                    # 陈列柜
    "pool table, billiard table, snooker table",                # 台球桌
    "pillow",                                                   # 枕头
    "screen door, screen",                                      # 推拉门
    "stairway, staircase",                                      # 楼梯
    "river",                                                    # 河
    "bridge, span",                                             # 桥
    "bookcase",                                                 # 书架
    "blind, screen",                                            # 百叶窗 
    "coffee table",                                             # 茶几
    "toilet, can, commode, crapper, pot, potty, stool, throne", # 马桶
    "flower",                                                   # 花
    "book",                                                     # 书
    "hill",                                                     # 小山
    "bench",                                                    # 长凳
    "countertop",                                               # 台面
    "stove",                                                    # 炉子
    "palm, palm tree",                                          # 棕榈树
    "kitchen island",                                           # 厨房中岛
    "computer",                                                 # 电脑
    "swivel chair",                                             # 旋转椅
    "boat",                                                     # 船
    "bar",                                                      # 吧台
    "arcade machine",                                           # 游戏机
    "hovel, hut, hutch, shack, shanty",                         # 棚屋
    "bus",                                                      # 公共汽车
    "towel",                                                    # 毛巾
    "light",                                                    # 灯
    "truck",                                                    # 卡车                   
    "tower",                                                    # 塔
    "chandelier",                                               # 吊灯
    "awning, sunshade, sunblind",                               # 遮阳篷
    "street lamp",                                              # 路灯
    "booth",                                                    # 亭
    "tv",                                                       # 电视
    "plane",                                                    # 飞机
    "dirt track",                                               # 土路
    "clothes",                                                  # 衣服
    "pole",                                                     # 杆
    "land, ground, soil",                                       # 土地                              
    "bannister, banister, balustrade, balusters, handrail",     # 栏杆
    "escalator, moving staircase, moving stairway",             # 电梯 
    "ottoman, pouf, pouffe, puff, hassock",                     # 脚凳
    "bottle",                                                   # 瓶子 
    "buffet, counter, sideboard",                               # 自助餐
    "poster, posting, placard, notice, bill, card",             # 海报
    "stage",                                                    # 舞台
    "van",                                                      # 货车
    "ship",                                                     # 船
    "fountain",                                                 # 喷泉
    "conveyer belt, conveyor belt, conveyer, conveyor, transporter",    # 传送带
    "canopy",                                                   # 篷
    "washer, automatic washer, washing machine",                # 洗衣机
    "plaything, toy",                                           # 玩具
    "pool",                                                     # 游泳池
    "stool",                                                    # 凳子
    "barrel, cask",                                             # 桶
    "basket, handbasket",                                       # 篮子                          
    "falls",                                                    # 瀑布  
    "tent",                                                     # 帐篷
    "bag",                                                      # 包
    "minibike, motorbike",                                      # 摩托车
    "cradle",                                                   # 摇篮
    "oven",                                                     # 烤箱
    "ball",                                                     # 球
    "food, solid food",                                         # 食物
    "step, stair",                                              # 楼梯
    "tank, storage tank",                                       # 罐
    "trade name",                                               # 商标
    "microwave",                                                # 微波炉
    "pot",                                                      # 锅
    "animal",                                                   # 动物
    "bicycle",                                                  # 自行车
    "lake",                                                     # 湖
    "dishwasher",                                               # 洗碗机
    "screen",                                                   # 屏幕
    "blanket, cover",                                           # 毯子
    "sculpture",                                                # 雕塑
    "hood, exhaust hood",                                       # 排风罩
    "sconce",                                                   # 壁灯
    "vase",                                                     # 花瓶
    "traffic light",                                            # 交通灯
    "tray",                                                     # 托盘
    "trash can",                                                # 垃圾桶
    "fan",                                                      # 风扇
    "pier",                                                     # 码头
    "crt screen",                                               # 阴极射线管显示器
    "plate",                                                    # 盘子
    "monitor",                                                  # 显示器
    "bulletin board",                                           # 布告栏
    "shower",                                                   # 淋浴
    "radiator",                                                 # 暖气片
    "glass, drinking glass",                                    # 玻璃
    "clock",                                                    # 时钟
    "flag",                                                     # 旗帜
]

INDOOR_ADE20K_SEM_SEG_CATEGORIES = [
    "wall",                                                     # 墙
    "ceiling",                                                  # 天花板
    "floor",                                                    # 地板
    "bed",                                                      # 床
    "window",                                                   # 窗户
    "cabinet",                                                  # 柜子
    "door",                                                     # 门
    "table",                                                    # 桌子
    "curtain",                                                  # 窗帘
    "chair",                                                    # 椅子
    "painting, picture",                                        # 画
    "bag",                                                      # 包
    "sofa",                                                     # 沙发
    "shelf",                                                    # 架子
    "mirror",                                                   # 镜子
    "rug",                                                      # 地毯
    "armchair",                                                 # 扶手椅
    "seat",                                                     # 座位
    "desk",                                                     # 书桌
    "wardrobe, closet, press",                                  # 衣柜
    "lamp",                                                     # 灯
    "tub",                                                      # 浴缸
    "cushion",                                                  # 垫子
    "base, pedestal, stand",                                    # 底座
    "box",                                                      # 盒子
    "column, pillar",                                           # 柱子
    "signboard, sign",                                          # 招牌
    "stage",                                                    # 舞台
    "bannister, banister, balustrade, balusters, handrail",     # 栏杆
    "escalator, moving staircase, moving stairway",             # 电梯 
    "chest of drawers, chest, bureau, dresser",                 # 抽屉柜
    "counter",                                                  # 柜台
    "sink",                                                     # 洗手池
    "fireplace",                                                # 壁炉
    "refrigerator, icebox",                                     # 冰箱
    "stairs",                                                   # 楼梯
    "case, display case, showcase, vitrine",                    # 展示柜
    "pool table, billiard table, snooker table",                # 台球桌
    "pillow",                                                   # 枕头
    "coffee table",                                             # 茶几
    "flower",                                                   # 花
    "book",                                                     # 书
    "bench",                                                    # 长凳
    "screen door, screen",                                      # 屏风
    "stairway, staircase",                                      # 楼梯
    "bookcase",                                                 # 书架
    "blind, screen",                                            # 百叶窗
    "coffee table",                                             # 茶几
    "toilet, can, commode, crapper, pot, potty, stool, throne", # 马桶
    "countertop",                                               # 台面
    "stove",                                                    # 炉子
    "towel",                                                    # 毛巾
    "kitchen island",                                           # 厨房岛台
    "computer",                                                 # 电脑
    "swivel chair",                                             # 旋转椅
    "light",                                                    # 灯
    "chandelier",                                               # 吊灯
    "awning, sunshade, sunblind",                               # 遮阳篷
    "tv",                                                       # 电视
    "clothes",                                                  # 衣服
    "ottoman, pouf, pouffe, puff, hassock",                     # 脚凳
    "bottle",                                                   # 瓶子
    "buffet, counter, sideboard",                               # 餐柜
    "poster, posting, placard, notice, bill, card",             # 海报
    "washer, automatic washer, washing machine",                # 洗衣机
    "plaything, toy",                                           # 玩具
    "stool",                                                    # 凳子
    "barrel, cask",                                             # 桶
    "dishwasher",                                               # 洗碗机
    "basket, handbasket",                                       # 篮子
    "cradle",                                                   # 摇篮
    "oven",                                                     # 烤箱
    "ball",                                                     # 球
    "food, solid food",                                         # 食物
    "microwave",                                                # 微波炉
    "screen",                                                   # 屏幕
    "pot",                                                      # 锅
    "blanket, cover",                                           # 毯子
    "sculpture",                                                # 雕塑
    "hood, exhaust hood",                                       # 排气罩
    "sconce",                                                   # 壁灯
    "vase",                                                     # 花瓶
    "tray",                                                     # 托盘
    "trash can",                                                # 垃圾桶
    "fan",                                                      # 风扇
    "plate",                                                    # 盘子
    "monitor",                                                  # 监视器
    "bulletin board",                                           # 公告板
    "shower",                                                   # 淋浴
    "radiator",                                                 # 暖气片
    "glass, drinking glass",                                    # 玻璃杯
    "clock",                                                    # 时钟
    "light",                                                    # 灯
    "chandelier",                                               # 吊灯
]


# find empty room
STRUCTURE_LABELS = [
    "wall",                                                     # 墙
    "ceiling",                                                  # 天花板
    "floor",                                                    # 地板
    "window",                                                   # 窗户
    "door",                                                     # 门
    "screen door, screen",                                      # 屏风
    "curtain",                                                  # 窗帘
    "stairs",                                                   # 楼梯
    "stairway, staircase",                                      # 楼梯
    "column, pillar",                                           # 柱子
    "blind, screen",                                            # 百叶窗
    "bannister, banister, balustrade, balusters, handrail",     # 栏杆
    "step, stair",                                              # 楼梯
    # only in kitchen
    "kitchen island",                                           # 厨房岛台
    # only in living room
    "fireplace",                                                # 壁炉
]   

KITCHEN_LABELS = [
    "cabinet",                                                  # 柜子
    "sink",                                                     # 洗手池
    "countertop",                                               # 台面
    "counter",                                                  # 柜台
    "stove",                                                    # 炉子
    "hood, exhaust hood",                                       # 排气罩
    "plate",                                                    # 盘子
    "food, solid food",                                         # 食物
    "kitchen island",                                           # 厨房岛台
    "refrigerator, icebox",                                     # 冰箱
    "microwave",                                                # 微波炉
    "oven",                                                     # 烤箱
    "dishwasher",                                               # 洗碗机
    "pot",                                                      # 锅
    "buffet, counter, sideboard",                               # 餐柜
    "glass, drinking glass",                                    # 玻璃杯
]

BATHROOM_LABELS = [
    "shower",                                                   # 淋浴
    "sink",                                                     # 洗手池
    "countertop",                                               # 台面
    "counter",                                                  # 柜台
    "glass, drinking glass",                                    # 玻璃杯
    "toilet, can, commode, crapper, pot, potty, stool, throne", # 马桶
    "shelf",                                                    # 架子
    "mirror",                                                   # 镜子
    "tub",                                                      # 浴缸
    "chandelier",                                               # 吊灯
    "towel",                                                    # 毛巾
    "clothes",                                                  # 衣服
]

LIVINGROOM_LABELS = [
    "sofa",                                                     # 沙发
    "chair",                                                    # 椅子
    "table",                                                    # 桌子
    "lamp",                                                     # 灯
    "rug",                                                      # 地毯
    "shelf",                                                    # 架子
    "cushion",                                                  # 垫子
    "armchair",                                                 # 扶手椅
    "stool",                                                    # 凳子
    "tv",                                                       # 电视
    "chandelier",                                               # 吊灯
    "vase",                                                     # 花瓶
    "radiator",                                                 # 暖气片
    "glass, drinking glass",                                    # 玻璃杯
    "clock",                                                    # 时钟
    "fireplace",                                                # 壁炉
    "coffee table",                                             # 茶几
]

BEDROOM_LABELS = [
    "bed",                                                      # 床
    "lamp",                                                     # 灯
    "rug",                                                      # 地毯
    "pillow",                                                   # 枕头
    "cushion",                                                  # 垫子
    "armchair",                                                 # 扶手椅
    "ottoman, pouf, pouffe, puff, hassock",                     # 脚凳
    "stool",                                                    # 凳子
    "poster, posting, placard, notice, bill, card",             # 海报
    "sconce",                                                   # 壁灯
    "chest of drawers, chest, bureau, dresser",                 # 抽屉柜
    "wardrobe, closet, press",                                  # 衣柜
    "painting, picture",                                        # 画
]                        

HOME_OFFICE_LABELS = [
    "computer",                                                 # 电脑
    "desk",                                                     # 书桌
    "chair",                                                    # 椅子
    "lamp",                                                     # 灯
    "rug",                                                      # 地毯
    "shelf",                                                    # 架子
    "case, display case, showcase, vitrine",                    # 展示柜
    "clock",                                                    # 时钟
    "painting, picture",                                        # 画
]
# fmt: on


CERTAIN_BATHROOM_LABELS = [
    "toilet, can, commode, crapper, pot, potty, stool, throne",
    "sink",
    "tub",
]

CERTAIN_LVINGROOM_LABELS = ["fireplace"]
CERTAIN_BEDROOM_LABELS = ["bed"]
CERTAIN_KITCHEN_LABELS = [
    "stove",
    "refrigerator, icebox",
    "kitchen island",
    "dishwasher",
    "oven",
]

# EDIT mode
KEEP_LABELS = [
    "wall",  # 墙
    "ceiling",  # 天花板
    "floor",  # 地板
    "window",  # 窗户
    "door",  # 门
    "fence",  # 围栏
    "rail",  # 栏杆
    "sink",  # 洗手池
    "toilet, can, commode, crapper, pot, potty, stool, throne",  # 马桶
    "countertop",  # 台面
    "stove",  # 炉子
    "curtain",  # 窗帘
    "stairs",  # 楼梯
    "stairway, staircase",  # 楼梯
    "column, pillar",  # 柱子
    "blind, screen",  # 百叶窗
    "escalator, moving staircase, moving stairway",  # 电梯
    "counter",  # 柜台
    "bannister, banister, balustrade, balusters, handrail",  # 栏杆
    # "light",                                                    # 灯
    # "chandelier",                                               # 吊灯
    "step, stair",  # 楼梯
    "stage",  # 舞台
    # only in kitchen
    "kitchen island",  # 厨房岛台
    # only in living room
    "fireplace",  # 壁炉
    "cabinet",  # 橱柜
]

# KEEP_STRUCTURE_LABELS = KEEP_LABELS.copy()
# KEEP_STRUCTURE_LABELS.extend(["wall", "floor", "ceiling"])

CERTAIN_HOME_OFFICE_LABELS = []

CERTAIN_HOME_OFFICE_SET = set(CERTAIN_HOME_OFFICE_LABELS)
CERTAIN_KITCHEN_SET = set(CERTAIN_KITCHEN_LABELS)
CERTAIN_BEDROOM_SET = set(CERTAIN_BEDROOM_LABELS)
CERTAIN_LVINGROOM_SET = set(CERTAIN_LVINGROOM_LABELS)
CERTAIN_BATHROOM_SET = set(CERTAIN_BATHROOM_LABELS)
INDOOR_LABELS = INDOOR_ADE20K_SEM_SEG_CATEGORIES
INDOOR_SETS = set(INDOOR_ADE20K_SEM_SEG_CATEGORIES)
STRUCTURE_SETS = set(STRUCTURE_LABELS)
KITCHEN_SETS = set(KITCHEN_LABELS)
BATHROOM_SETS = set(BATHROOM_LABELS)
LIVINGROOM_SETS = set(LIVINGROOM_LABELS)
BEDROOM_SETS = set(BEDROOM_LABELS)
HOME_OFFICE_SETS = set(HOME_OFFICE_LABELS)
# STANDARD_STRUCTURE_SETS = set(STANDARD_STRUCTURE_LABELS)
# NON_STANDARD_STRUCTURE_SETS = STRUCTURE_SETS - STANDARD_STRUCTURE_SETS
# NON_STANDARD_STRUCTURE_LABELS = list(NON_STANDARD_STRUCTURE_SETS)
# KEEP_SET = set(KEEP_LABELS)

NON_KEEP_LABELS = [label for label in INDOOR_SETS if label not in KEEP_LABELS]

ADE20K_LABEL2ID = {label: i for i, label in enumerate(ADE20K_SEM_SEG_CATEGORIES)}
# transformers for ADE20K
ADE20K_LABEL2ID.update({"unlabeled": 255})
ADE20K_ID2LABEL = {i: label for label, i in ADE20K_LABEL2ID.items()}
from collections import OrderedDict


def get_label2id(sets: set, is_order=True) -> dict:

    if is_order:
        label_ids = sorted([ADE20K_LABEL2ID[label] for label in sets])
        return OrderedDict(
            {ADE20K_ID2LABEL[label_id]: label_id for label_id in label_ids}
        )
    else:
        return {label: ADE20K_LABEL2ID[label] for label in sets}


INDOOR_LABEL2ID = get_label2id(INDOOR_SETS)
STRUCTURE_LABEL2ID = get_label2id(STRUCTURE_SETS)
KITCHEN_LABEL2ID = get_label2id(KITCHEN_SETS)
BATHROOM_LABEL2ID = get_label2id(BATHROOM_SETS)
LIVINGROOM_LABEL2ID = get_label2id(LIVINGROOM_SETS)
BEDROOM_LABEL2ID = get_label2id(BEDROOM_SETS)
HOME_OFFICE_LABEL2ID = get_label2id(HOME_OFFICE_SETS)
# STANDARD_STRUCTURE_LABEL2ID = get_label2id(STANDARD_STRUCTURE_SETS)
# NON_STANDARD_STRUCTURE_LABEL2ID = get_label2id(NON_STANDARD_STRUCTURE_SETS)
KEEP_LABEL2ID = get_label2id(KEEP_LABELS)
NON_KEEP_LABEL2ID = get_label2id(NON_KEEP_LABELS)


def get_ade20k_labels_from_mask(mask: np.ndarray) -> t.List[str]:
    label_ids = np.unique(mask)
    labels = [ADE20K_SEM_SEG_CATEGORIES[label_id] for label_id in label_ids]
    return labels


def guess_room_type_by_mask(mask: np.ndarray, invalid_threshold: float = 0.2):
    h, w = mask.shape
    area = h * w
    label_ids, label_cnt = np.unique(mask, return_counts=True)
    id2cnt = dict(zip(label_ids, label_cnt))

    home_ids = [ADE20K_LABEL2ID[label] for label in HOME_OFFICE_LABELS]
    kitchen_ids = [ADE20K_LABEL2ID[label] for label in KITCHEN_LABELS]
    bedroom_ids = [ADE20K_LABEL2ID[label] for label in BEDROOM_LABELS]
    livingroom_ids = [ADE20K_LABEL2ID[label] for label in LIVINGROOM_LABELS]
    bathroom_ids = [ADE20K_LABEL2ID[label] for label in BEDROOM_LABELS]

    home_cnt = sum([id2cnt.get(i, 0) for i in home_ids])
    kitchen_cnt = sum([id2cnt.get(i, 0) for i in kitchen_ids])
    bedroom_cnt = sum([id2cnt.get(i, 0) for i in bedroom_ids])
    livingroom_cnt = sum([id2cnt.get(i, 0) for i in livingroom_ids])
    bathroom_cnt = sum([id2cnt.get(i, 0) for i in bathroom_ids])

    room_cnt = [home_cnt, kitchen_cnt, bedroom_cnt, livingroom_cnt, bathroom_cnt]
    room_types = ["home_office", "kitchen", "bedroom", "livingroom", "bathroom"]

    index = max(range(len(room_cnt)), key=lambda i: room_cnt[i])
    if room_cnt[index] / area < invalid_threshold:
        return "unknown"
    else:
        return room_types[index]


def guess_room_type_by_label_cnt(
    label_ids: np.ndarray, label_cnt: np.ndarray, invalid_threshold: float = 0.2
):
    label_ids = np.asarray(label_ids)
    label_cnt = np.asarray(label_cnt)
    area = label_cnt.sum()
    id2cnt = dict(zip(label_ids, label_cnt))

    home_ids = [ADE20K_LABEL2ID[label] for label in HOME_OFFICE_LABELS]
    kitchen_ids = [ADE20K_LABEL2ID[label] for label in KITCHEN_LABELS]
    bedroom_ids = [ADE20K_LABEL2ID[label] for label in BEDROOM_LABELS]
    livingroom_ids = [ADE20K_LABEL2ID[label] for label in LIVINGROOM_LABELS]
    bathroom_ids = [ADE20K_LABEL2ID[label] for label in BEDROOM_LABELS]

    home_cnt = sum([id2cnt.get(i, 0) for i in home_ids])
    kitchen_cnt = sum([id2cnt.get(i, 0) for i in kitchen_ids])
    bedroom_cnt = sum([id2cnt.get(i, 0) for i in bedroom_ids])
    livingroom_cnt = sum([id2cnt.get(i, 0) for i in livingroom_ids])
    bathroom_cnt = sum([id2cnt.get(i, 0) for i in bathroom_ids])

    room_cnt = [home_cnt, kitchen_cnt, bedroom_cnt, livingroom_cnt, bathroom_cnt]
    room_types = ["home_office", "kitchen", "bedroom", "livingroom", "bathroom"]

    index = max(range(len(room_cnt)), key=lambda i: room_cnt[i])
    if room_cnt[index] / area < invalid_threshold:
        return "unknown"
    else:
        return room_types[index]


def is_valid_indoor_by_label_cnt(
    label_cnt: np.ndarray, threshold: float = 0.85
) -> bool:
    indoor_label_ids = INDOOR_LABEL2ID.values()
    indoor_cnt = sum([label_cnt[label_id] for label_id in indoor_label_ids])
    total_cnt = label_cnt.sum()
    return indoor_cnt / total_cnt > threshold


class ADE20kMask:

    def __init__(self, mask: np.ndarray = None, **kwargs):
        self.label_ids = kwargs.get("label_ids")
        self.label_cnts = kwargs.get("label_cnts")
        if mask is not None:
            self.mask = np.asarray(mask)
            self.label_ids, self.label_cnts = np.unique(self.mask, return_counts=True)
        assert (
            self.label_cnts is not None and self.label_ids is not None
        ), "label_ids and label_cnt must be provided"
        # print(self.label_cnts)
        self.area = np.sum(self.label_cnts)
        # print(self.area)
        self.labels = [ADE20K_ID2LABEL[label_id] for label_id in self.label_ids]

    @classmethod
    def from_mask(cls, mask: np.ndarray):
        return cls(mask=mask)

    @classmethod
    def from_mask_path(cls, mask_path: str):
        from PIL import Image

        mask = np.asarray(Image.open(mask_path).convert("L"))
        return cls(mask)

    @classmethod
    def from_label(cls, label_ids, label_cnts):
        return cls(label_ids=np.asarray(label_ids), label_cnts=np.asarray(label_cnts))

    def get_mask_by_labels(self, labels: list[str], invert: bool = False) -> np.ndarray:
        if self.mask is None:
            raise ValueError("mask is not provided")
        label_ids = [ADE20K_LABEL2ID[label] for label in labels]
        return np.isin(self.mask, label_ids, invert=invert)

    def get_exist_labels(self, ref_labels: list[str]) -> list[str]:
        ref_label_ids = [ADE20K_LABEL2ID[label] for label in ref_labels]
        return [
            ADE20K_ID2LABEL[label_id]
            for label_id in self.label_ids
            if label_id in ref_label_ids
        ]

    def find_intersect_area(self, ref_labels, threshold):
        ref_label_ids = [ADE20K_LABEL2ID[label] for label in ref_labels]
        cross_label = np.isin(self.label_ids, ref_label_ids)
        cross_area = np.sum(self.label_cnts[cross_label])
        ratio = 1.0 * cross_area / self.area
        return ratio > threshold

    def get_labels_score(self, labels) -> float:
        label_ids = [ADE20K_LABEL2ID[label] for label in labels]
        mask = np.isin(self.label_ids, label_ids)
        area = np.sum(self.label_cnts[mask])
        ratio = 1.0 * area / self.area
        return ratio
