import os,re
import sys
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from ultralytics import YOLO
import ultralytics.utils.patches as ultralytics_patches

# 恢复原始 PIL.Image.open，禁用 ultralytics 的 HEIF fallback 检查
Image.open = ultralytics_patches._image_open


def get_base_dir():
    """
    获取程序所在目录
    - 调试运行：返回当前 py 文件所在目录
    - 打包后运行：返回 exe 所在目录
    """
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


# ================== 配置区域 ==================
BASE_DIR = get_base_dir()

# 默认输入输出目录（独立运行时使用）
IN_DIR = os.path.join(BASE_DIR, "input")
OUT_DIR = os.path.join(BASE_DIR, "output")

# 默认模型目录（独立运行时使用）
REPO_DIR = os.path.join(BASE_DIR, "parseq-main", "parseq-main")
YOLO_WEIGHT = os.path.join(BASE_DIR, "best.pt")
HUB_CACHE_DIR = os.path.join(BASE_DIR, "torch_cache")

IMG_EXTS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
# ================== 模型局部变量参数 ==================
ALLOWED_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
TARGET_LEN = 9
RETRY_BRANCH_ENABLE = True
MIN_CONF_FOR_DIRECT_ACCEPT = 0.96
# 轻量评分参数
LENGTH_SCORE = 4.0
ALLOWED_CHAR_SCORE = 0.35
CONF_SCORE_WEIGHT = 3.5

# 是否启用位置约束修正
POSITION_FIX_ENABLE = True
# 位置修正后的轻微罚分，避免规则修正过猛
POSITION_FIX_PENALTY = 0.15

# 第一位 I/T 视觉二次判断
FIRST_CHAR_IT_VISUAL_FIX_ENABLE = True
DET_CONF_THRES = 0.25
MAX_DET_PER_IMAGE = 5

PAD_X = 8
PAD_Y = 4

MIN_BOX_W = 80
MIN_BOX_H = 20
MIN_BOX_AREA = 2500
MIN_RATIO = 3.0
MAX_RATIO = 20.0

KEEP_ONLY_BEST = True

# ================== 模型全局变量 ==================
_PARSEQ = None
_IMG_TRANSFORM = None
_DEVICE = None
_YOLO_MODEL = None

# ================== 工具函数 ==================
def iter_image_paths(in_path: str):
    p = Path(in_path)
    if p.is_file() and p.suffix.lower() in IMG_EXTS:
        return [p]
    elif p.is_dir():
        paths = [x for x in p.iterdir() if x.suffix.lower() in IMG_EXTS]
        paths.sort()
        return paths
    else:
        raise NotADirectoryError(f"无效路径: {in_path}")


def post_process_text(text: str) -> str:
    text = text.upper().strip()

    mapping = {
        "S": "5",
        "O": "0",
        "Q": "0",
        "Z": "2",
        "L": "1",
    }

    processed = []
    for ch in text:
        if ch in mapping:
            ch = mapping[ch]
        if ch in ALLOWED_CHARS:
            processed.append(ch)

    return "".join(processed)

def position_post_fix(text: str) -> str:
    """
    9位镭刻码位置约束修正：
    - 第1位更像字母：1 -> I
    - 第7位更像字母：0 -> C, 5/8 -> B
    - 最后两位更像数字：I/L -> 1, O/Q/C -> 0, B -> 8, S -> 5, Z -> 2
    """
    if len(text) != TARGET_LEN:
        return text

    chars = list(text)

    # -------- 第1位：数字 → 字母 --------
    if chars[0] == "1":
        chars[0] = "I"

    # -------- 第7位：数字 → 字母 --------
    if chars[6] == "0":
        chars[6] = "C"
    elif chars[6] in ["3", "5", "8"]:
        chars[6] = "B"

    # -------- 最后两位：字母 → 数字 --------
    for i in [7, 8]:
        if chars[i] in ["I", "L"]:
            chars[i] = "1"
        elif chars[i] in ["O", "Q", "C"]:
            chars[i] = "0"
        elif chars[i] == "B":
            chars[i] = "8"
        elif chars[i] == "S":
            chars[i] = "5"
        elif chars[i] == "Z":
            chars[i] = "2"
        elif chars[7] == "9":
            chars[7] = "0"

    return "".join(chars)


def judge_first_char_i_or_t(crop_pil: Image.Image):
    """
    第一位 I/T 视觉判断加强版。

    返回:
    "I"  -> 视觉上更像 I
    "T"  -> 视觉上更像 T
    None -> 不确定，不修正

    核心思路：
    1. 先从整行裁剪图里定位第一位字符区域
    2. 再比较第一位字符顶部横杠宽度和中部竖线宽度
    3. 只有特征明显时才返回 I/T
    """
    try:
        rgb = crop_pil.convert("RGB")
        w, h = rgb.size

        if w <= 0 or h <= 0:
            return None

        gray = np.array(rgb.convert("L"))

        # 放大，便于细线条分析
        scale = 4
        gray = cv2.resize(
            gray,
            (gray.shape[1] * scale, gray.shape[0] * scale),
            interpolation=cv2.INTER_CUBIC
        )

        H, W = gray.shape

        # 去掉底部亮边干扰，只看上面 85%
        gray_work = gray[:int(H * 0.85), :]

        # 局部对比增强
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_work = clahe.apply(gray_work)

        # 高通增强：突出镭刻细边缘
        bg = cv2.GaussianBlur(gray_work, (0, 0), 7)
        hp = cv2.absdiff(gray_work, bg)
        hp = cv2.normalize(hp, None, 0, 255, cv2.NORM_MINMAX)

        # 二值化
        _, binary = cv2.threshold(
            hp,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # 适当膨胀，让同一个字符的细线连起来
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2))
        binary = cv2.dilate(binary, kernel, iterations=1)

        # 只在左侧约 2 个字符宽度内找第一位字符
        search_w = int(W / TARGET_LEN * 2.0)
        search_w = max(20, min(W, search_w))
        left_part = binary[:, :search_w]

        # 找连通域，选择最靠左且尺寸合理的区域
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(left_part, connectivity=8)

        components = []
        for i in range(1, num_labels):
            x, y, bw, bh, area = stats[i]

            if area < 20:
                continue
            if bh < H * 0.18:
                continue
            if bw < 2:
                continue

            components.append((x, y, bw, bh, area))

        if not components:
            # 兜底：取左侧 1 个字符宽度
            x0 = 0
            x1 = int(W / TARGET_LEN * 1.15)
            y0 = 0
            y1 = int(H * 0.85)
        else:
            # 选择最靠左的字符区域
            components.sort(key=lambda c: (c[0], -c[4]))
            x, y, bw, bh, area = components[0]

            pad_x = int(4 * scale)
            pad_y = int(3 * scale)

            x0 = max(0, x - pad_x)
            x1 = min(search_w, x + bw + pad_x)
            y0 = max(0, y - pad_y)
            y1 = min(binary.shape[0], y + bh + pad_y)

        char_bin = binary[y0:y1, x0:x1]

        if char_bin.size == 0:
            return None

        # 再计算字符区域内的前景 bbox
        ys, xs = np.where(char_bin > 0)
        if len(xs) == 0 or len(ys) == 0:
            return None

        bx0, bx1 = xs.min(), xs.max()
        by0, by1 = ys.min(), ys.max()

        char = char_bin[by0:by1 + 1, bx0:bx1 + 1]

        ch, cw = char.shape
        if ch <= 0 or cw <= 0:
            return None

        # 顶部区域：T 的顶部横杠更明显
        top_band = char[:max(1, int(ch * 0.30)), :]

        # 中部区域：I/T 的竖线主要在这里
        mid_band = char[int(ch * 0.35):max(int(ch * 0.85), int(ch * 0.35) + 1), :]

        top_cols = np.count_nonzero(top_band.sum(axis=0) > 0)
        mid_cols = np.count_nonzero(mid_band.sum(axis=0) > 0)

        top_width_ratio = top_cols / max(cw, 1)
        mid_width_ratio = mid_cols / max(cw, 1)
        ratio = top_cols / max(mid_cols, 1)

        # T：顶部横杠明显宽，中部竖线明显窄
        if top_width_ratio > 0.72 and ratio > 1.55 and mid_width_ratio < 0.62:
            return "T"

        # I：顶部不够宽，或者顶部和中部差距不明显
        if top_width_ratio < 0.62 and ratio < 1.45:
            return "I"

        # 对你这种 I 被误成 T 的情况，再加一个保守判断：
        # 如果中部竖线很窄，顶部也不是特别宽，更像 I
        if mid_width_ratio < 0.42 and top_width_ratio < 0.70:
            return "I"

        return None

    except Exception:
        return None


def refine_first_char_i_t_by_vision(text: str, crop_pil: Image.Image):
    """
    第一位 I/T 视觉二次判断。
    不做强规则，只在视觉判断明确时修正。
    """
    if not FIRST_CHAR_IT_VISUAL_FIX_ENABLE:
        return text, ""

    if len(text) != TARGET_LEN:
        return text, ""

    # 只处理 I/T 互相混淆
    if text[0] not in ["I", "T"]:
        return text, ""

    visual_char = judge_first_char_i_or_t(crop_pil)

    if visual_char is None:
        return text, ""

    if visual_char != text[0]:
        new_text = visual_char + text[1:]
        return new_text, "+it_visual"

    return text, ""


def enhance_image_for_retry(img: Image.Image):
    """
    多分支增强，不再只返回一张图
    返回: [(tag, pil_img), ...]
    """
    rgb = img.convert("RGB")
    w, h = rgb.size
    gray = np.array(rgb.convert("L"))

    # 统一先放大，给后续所有分支打基础
    up = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

    def gray_to_rgb_pil(arr):
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB))

    variants = []

    # 1) 只放大，不做强处理
    variants.append(("resize_only", gray_to_rgb_pil(up)))

    # 2) CLAHE + 温和反锐化，避免 C 被补成 0
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(up)
    blur = cv2.GaussianBlur(clahe_img, (0, 0), 1.0)
    unsharp = cv2.addWeighted(clahe_img, 1.12, blur, -0.12, 0)
    variants.append(("clahe_unsharp", gray_to_rgb_pil(unsharp)))

    # 3) 背景归一化，适合曲面灰度渐变
    bg = cv2.GaussianBlur(up, (0, 0), 17)
    flat = cv2.subtract(up, bg)
    flat = cv2.normalize(flat, None, 0, 255, cv2.NORM_MINMAX)
    flat = clahe.apply(flat)
    variants.append(("bg_flatten", gray_to_rgb_pil(flat)))

    # 4) 轻度二值分支
    adap_base = clahe.apply(up)
    bin_img = cv2.adaptiveThreshold(
        adap_base,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        6
    )

    # 轻微腐蚀，避免字符边缘过厚导致 C 闭口
    kernel_bin = np.ones((2, 2), np.uint8)
    bin_img = cv2.erode(bin_img, kernel_bin, iterations=1)
    variants.append(("adaptive_bin", gray_to_rgb_pil(bin_img)))

    # 5) 中值滤波 + CLAHE，适合噪点稍多的情况
    median_img = cv2.medianBlur(up, 3)
    median_img = clahe.apply(median_img)
    variants.append(("median_clahe", gray_to_rgb_pil(median_img)))

    # 6) 字符分离增强，重点缓解 538->533、C02->C12 这类粘连误识别
    kernel_sep = np.ones((1, 2), np.uint8)
    eroded = cv2.erode(up, kernel_sep, iterations=1)
    variants.append(("char_separate", gray_to_rgb_pil(eroded)))

    # 7) bilateral + CLAHE，尽量保边，改善 0/1/9 混淆
    bilateral = cv2.bilateralFilter(up, 7, 50, 50)
    bilateral = clahe.apply(bilateral)
    variants.append(("bilateral_clahe", gray_to_rgb_pil(bilateral)))

    # 8) blackhat，增强细暗纹理 / 细刻字边缘
    kernel_bh = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
    blackhat = cv2.morphologyEx(up, cv2.MORPH_BLACKHAT, kernel_bh)
    blackhat = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX)
    blackhat = clahe.apply(blackhat)
    variants.append(("blackhat", gray_to_rgb_pil(blackhat)))

    # 9) vertical_stroke，强调竖向笔画，帮助区分 I / T
    # 改得更强一点：先提取竖线，再和原图融合
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    vertical = cv2.morphologyEx(up, cv2.MORPH_OPEN, kernel_v)
    vertical = cv2.normalize(vertical, None, 0, 255, cv2.NORM_MINMAX)
    vertical_mix = cv2.addWeighted(up, 0.55, vertical, 0.65, 0)
    vertical_mix = np.clip(vertical_mix, 0, 255).astype(np.uint8)
    vertical_mix = clahe.apply(vertical_mix)
    variants.append(("vertical_stroke", gray_to_rgb_pil(vertical_mix)))

    # 去重，避免分支结果完全相同
    uniq = {}
    for tag, pil_img in variants:
        key = pil_img.tobytes()
        if key not in uniq:
            uniq[key] = (tag, pil_img)

    return list(uniq.values())


def score_text_candidate(text: str, conf: float) -> float:
    """
    不强制固定完整码格式，只做弱结构评分：
    1. 长度是否等于9
    2. 合法字符数量
    3. OCR置信度
    4. 第1位更像字母
    5. 第7位更像字母
    6. 最后两位更像数字
    """
    score = 0.0

    # 1) 长度优先
    if len(text) == TARGET_LEN:
        score += LENGTH_SCORE
    else:
        score -= abs(len(text) - TARGET_LEN) * 2.0

    # 2) 合法字符越多越好
    valid_cnt = sum(1 for ch in text if ch in ALLOWED_CHARS)
    score += valid_cnt * ALLOWED_CHAR_SCORE

    # 3) 置信度
    conf = max(0.0, min(conf, 1.0))
    score += conf * CONF_SCORE_WEIGHT

    # 4) 第1位更像字母：解决 1xxx -> Ixxx
    if len(text) >= 1:
        if text[0].isalpha():
            score += 1.0
        else:
            score -= 1.0

    # 5) 第7位更像字母：解决 C->0、B->5
    if len(text) >= 7:
        if text[6].isalpha():
            score += 1.5
        else:
            score -= 1.5

        if text[6] in ["A", "B", "C"]:
            score += 0.4

    # 6) 最后两位更像数字
    if len(text) >= 9:
        if text[7].isdigit():
            score += 1.0
        else:
            score -= 1.0

        if text[8].isdigit():
            score += 1.0
        else:
            score -= 1.0

    return score


def choose_best_result(candidates):
    """
    candidates: [
        {"text": xxx, "conf": xxx, "source_tag": xxx},
        ...
    ]

    做两件事：
    1. 对所有增强分支结果统一评分
    2. 对每个结果再生成一个 position_post_fix 后的候选，参与评分
    """
    best_text = ""
    best_conf = 0.0
    best_tag = "unknown"
    best_score = -1e9

    for item in candidates:
        raw_text = item["text"]
        conf = item["conf"]
        source_tag = item["source_tag"]

        # 原始候选
        eval_list = [
            (raw_text, source_tag, 0.0)
        ]

        # 位置修正候选
        if POSITION_FIX_ENABLE:
            fixed_text = position_post_fix(raw_text)
            if fixed_text != raw_text:
                eval_list.append(
                    (fixed_text, source_tag + "+posfix", -POSITION_FIX_PENALTY)
                )

        for text, tag, penalty in eval_list:
            score = score_text_candidate(text, conf) + penalty

            if score > best_score:
                best_score = score
                best_text = text
                best_conf = conf
                best_tag = tag

    return best_text, best_conf, best_tag


def recognize_once(parseq, pil_img, img_transform, device):
    x = img_transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = parseq(x)
        pred = logits.softmax(-1)
        label, confidence = parseq.tokenizer.decode(pred)

    raw_text = label[0]
    final_text = post_process_text(raw_text)

    conf = 0.0
    try:
        c0 = confidence[0]
        if torch.is_tensor(c0):
            if c0.numel() == 1:
                conf = float(c0.item())
            else:
                conf = float(c0.mean().item())
        else:
            conf = float(c0)
    except Exception:
        conf = 0.0

    return final_text, conf


def detect_boxes(yolo_model, pil_img, conf_thres=0.35, max_det=5):
    results = yolo_model.predict(
        source=pil_img,
        conf=conf_thres,
        device="cpu",
        verbose=False,
        max_det=max_det
    )

    if not results:
        return []

    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        return []

    xyxy = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()

    boxes = []
    for box, conf in zip(xyxy, confs):
        x1, y1, x2, y2 = box.tolist()
        boxes.append((int(x1), int(y1), int(x2), int(y2), float(conf)))

    boxes.sort(key=lambda x: x[4], reverse=True)
    return boxes


def filter_candidate_boxes(boxes):
    filtered = []

    for x1, y1, x2, y2, conf in boxes:
        w = x2 - x1
        h = y2 - y1
        area = w * h
        ratio = w / max(h, 1)

        if w < MIN_BOX_W:
            continue
        if h < MIN_BOX_H:
            continue
        if area < MIN_BOX_AREA:
            continue
        if ratio < MIN_RATIO or ratio > MAX_RATIO:
            continue

        filtered.append((x1, y1, x2, y2, conf))

    return filtered


def select_best_box(boxes):
    if not boxes:
        return []

    scored = []
    for x1, y1, x2, y2, conf in boxes:
        w = x2 - x1
        h = y2 - y1
        area = w * h
        ratio = w / max(h, 1)

        target_ratio = 6.0
        ratio_score = max(0.0, 1.0 - abs(ratio - target_ratio) / target_ratio)
        area_score = min(area / 20000.0, 1.0)

        score = conf * 0.55 + ratio_score * 0.25 + area_score * 0.20
        scored.append(((x1, y1, x2, y2, conf), score))

    scored.sort(key=lambda x: x[1], reverse=True)
    best_box = scored[0][0]
    return [best_box]


def postprocess_boxes(boxes):
    boxes = filter_candidate_boxes(boxes)
    if KEEP_ONLY_BEST:
        boxes = select_best_box(boxes)
    return boxes


def expand_box(x1, y1, x2, y2, img_w, img_h, pad_x=8, pad_y=4):
    """
    裁剪框优化：
    - 左右正常扩展
    - 上方多留一点
    - 下方少留一点，减少底部亮边干扰
    """
    ex1 = int(max(0, x1 - pad_x))
    ex2 = int(min(img_w, x2 + pad_x))

    # 上多，下少
    ey1 = int(max(0, y1 - pad_y * 2))
    ey2 = int(min(img_h, y2 + max(1, int(pad_y * 0.5))))

    return ex1, ey1, ex2, ey2


def draw_boxes_on_image(img_bgr, recognized_items):
    """
    只在图上绘制检测框，不绘制识别文字和置信度
    """
    for item in recognized_items:
        x1, y1, x2, y2 = item["box"]
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 0, 255), 4)

    return img_bgr


def save_result_image(image_path, save_path, result_dict):
    """
    保存带有识别框的结果图像
    :param image_path: 输入图像路径
    :param save_path: 输出图像保存路径
    :param result_dict: 包含识别结果的字典
    """
    pil_img = Image.open(image_path).convert("RGB")  # 读取输入图像并转为RGB
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)  # 转换为BGR格式（OpenCV）
    img_bgr = draw_boxes_on_image(img_bgr, result_dict["recognized_items"])  # 绘制识别框
    cv2.imwrite(save_path, img_bgr)  # 保存图像到指定路径


# ================== 模型初始化 ==================
def init_models():
    global _PARSEQ, _IMG_TRANSFORM, _DEVICE, _YOLO_MODEL

    if _PARSEQ is not None and _YOLO_MODEL is not None:
        return

    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("device =", _DEVICE)

    print("loading parseq...")
    os.makedirs(HUB_CACHE_DIR, exist_ok=True)
    os.environ["TORCH_HOME"] = HUB_CACHE_DIR
    torch.hub.set_dir(HUB_CACHE_DIR)

    #log_emit(log_callback, f"torch hub cache dir = {HUB_CACHE_DIR}")
    #log_emit(log_callback, "loading parseq...")

    if REPO_DIR not in sys.path:
        sys.path.insert(0, REPO_DIR)

    from strhub.data.module import SceneTextDataModule
    _PARSEQ = torch.hub.load(
        REPO_DIR,
        "parseq",
        pretrained=True,
        source="local"
    ).eval().to(_DEVICE)

    _IMG_TRANSFORM = SceneTextDataModule.get_transform(_PARSEQ.hparams.img_size)
    print("parseq loaded")

    print("loading yolo...")
    _YOLO_MODEL = YOLO(YOLO_WEIGHT)
    print("yolo loaded")


# ================== 对外主函数 ==================
def detect_and_recognize_image(image_path):
    init_models()

    try:
        pil_img = Image.open(image_path).convert("RGB")
    except Exception as e:
        return {
            "image_path": str(image_path),
            "recognized_items": [],
            "best_text": "",
            "has_result": False,
            "error": f"IMAGE_OPEN_FAILED: {e}"
        }

    img_w, img_h = pil_img.size

    raw_boxes = detect_boxes(
        _YOLO_MODEL,
        pil_img,
        conf_thres=DET_CONF_THRES,
        max_det=MAX_DET_PER_IMAGE
    )

    boxes = postprocess_boxes(raw_boxes)

    if not boxes:
        return {
            "image_path": str(image_path),
            "recognized_items": [],
            "best_text": "",
            "has_result": False,
            "error": ""
        }

    recognized_items = []

    for j, (x1, y1, x2, y2, det_conf) in enumerate(boxes, 1):
        ex1, ey1, ex2, ey2 = expand_box(
            x1, y1, x2, y2,
            img_w, img_h,
            pad_x=PAD_X,
            pad_y=PAD_Y
        )

        crop_pil = pil_img.crop((ex1, ey1, ex2, ey2))

        # 收集所有候选
        candidates = []

        # 1) 原图直接识别
        first_text, first_conf = recognize_once(_PARSEQ, crop_pil, _IMG_TRANSFORM, _DEVICE)
        candidates.append({
            "text": first_text,
            "conf": first_conf,
            "source_tag": "first"
        })

        # 2) 如果原图结果已经非常好，也仍然可以保留多分支；
        #    这里不强行提前返回，方便比较增强后的效果
        if RETRY_BRANCH_ENABLE:
            retry_variants = enhance_image_for_retry(crop_pil)
            for variant_tag, retry_img in retry_variants:
                retry_text, retry_conf = recognize_once(_PARSEQ, retry_img, _IMG_TRANSFORM, _DEVICE)
                candidates.append({
                    "text": retry_text,
                    "conf": retry_conf,
                    "source_tag": variant_tag
                })

        # 3) 所有候选统一择优
        final_text, final_conf, source_tag = choose_best_result(candidates)

        # 4) 第一位 I/T 视觉二次判断
        fixed_text, it_tag = refine_first_char_i_t_by_vision(final_text, crop_pil)
        if it_tag:
            final_text = fixed_text
            source_tag = source_tag + it_tag

        recognized_items.append({
            "box": (ex1, ey1, ex2, ey2),
            "det_conf": det_conf,
            "text": final_text,
            "rec_conf": final_conf,
            "source_tag": source_tag,
            "crop_index": j,
            "all_candidates": candidates
        })

    best_text = recognized_items[0]["text"] if recognized_items else ""

    return {
        "image_path": str(image_path),
        "recognized_items": recognized_items,
        "best_text": best_text,
        "has_result": len(recognized_items) > 0,
        "error": ""
    }