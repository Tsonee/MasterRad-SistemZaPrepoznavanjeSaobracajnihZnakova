import sys
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


# ============================================================
# PARAMETRI SEGMENTACIJE BOJE
# ============================================================

# Praktično proširene granice crvene boje.
# Fleyeh koristi približno 0-10 i 240-255.
HUE_RED_LOW_MAX = 15
HUE_RED_HIGH_MIN = 235

SATURATION_MIN = 40

VALUE_MIN = 30
VALUE_MAX = 245

SUBIMAGE_SIZE = 16

# Fleyeh koristi 60.
# Niža vrednost bolje zadržava tanke i perspektivno
# deformisane crvene obode.
MIN_WHITE_PIXELS = 30


SEGMENTATION_COLORS = {
    "blue": [145, 175],
    "yellow": [25, 50],
}


# ============================================================
# PARAMETRI CNN KLASIFIKACIJE I PRIKAZA ROI OBLASTI
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "traffic_sign_cnn.keras"

CNN_IMAGE_SIZE = (64, 64)
CNN_MIN_CONFIDENCE = 0.98
ROI_PADDING_RATIO = 0.10

# Prikazuje svaki ROI u posebnom OpenCV prozoru.
SHOW_INDIVIDUAL_ROI_WINDOWS = True

# Dimenzija jedne kartice u zajednickom ROI prikazu.
ROI_TILE_WIDTH = 180
ROI_TILE_HEIGHT = 220
ROI_GRID_COLUMNS = 4

# Redosled mora biti potpuno isti kao tokom treniranja modela.
CLASS_NAMES = [
    "limit_20",
    "limit_30",
    "limit_50",
    "limit_60",
    "limit_70",
    "limit_80",
    "limit_100",
    "limit_120",
    "stop",
]

DISPLAY_NAMES = {
    "limit_20": "Ogranicenje 20",
    "limit_30": "Ogranicenje 30",
    "limit_50": "Ogranicenje 50",
    "limit_60": "Ogranicenje 60",
    "limit_70": "Ogranicenje 70",
    "limit_80": "Ogranicenje 80",
    "limit_100": "Ogranicenje 100",
    "limit_120": "Ogranicenje 120",
    "stop": "STOP",
}


def load_cnn_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "CNN model nije pronadjen:\n"
            f"{MODEL_PATH}\n\n"
            "Postavi traffic_sign_cnn.keras u isti folder kao ovaj program."
        )

    print(f"Ucitavanje CNN modela: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("CNN model je uspesno ucitan.")
    return model


def extract_padded_roi(image, bounding_box):
    """Iseca detektovani znak uz mali procenat okolnog prostora."""
    x, y, width, height = bounding_box
    image_height, image_width = image.shape[:2]

    pad_x = max(2, int(round(width * ROI_PADDING_RATIO)))
    pad_y = max(2, int(round(height * ROI_PADDING_RATIO)))

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(image_width, x + width + pad_x)
    y2 = min(image_height, y + height + pad_y)

    roi = image[y1:y2, x1:x2].copy()
    return roi, (x1, y1, x2 - x1, y2 - y1)


def classify_sign_roi(model, roi):
    """Klasifikuje OpenCV BGR ROI i vraca klasu i sigurnost."""
    if roi is None or roi.size == 0:
        return None, 0.0, None

    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi_resized = cv2.resize(
        roi_rgb,
        CNN_IMAGE_SIZE,
        interpolation=cv2.INTER_AREA,
    )

    # Model vec sadrzi Rescaling(1/255), zato ovde ne delimo sa 255.
    input_batch = np.expand_dims(roi_resized, axis=0).astype(np.float32)
    probabilities = model.predict(input_batch, verbose=0)[0]

    predicted_index = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_index])
    predicted_class = CLASS_NAMES[predicted_index]

    return predicted_class, confidence, probabilities


def classify_detections_with_cnn(image, detections, model):
    """Dodaje CNN rezultat i isečeni ROI svakom kandidatu."""
    for detection_index, detection in enumerate(detections, start=1):
        roi, padded_box = extract_padded_roi(
            image,
            detection["features"]["bounding_box"],
        )

        predicted_class, confidence, probabilities = classify_sign_roi(
            model,
            roi,
        )

        detection["roi_index"] = detection_index
        detection["cnn_class"] = predicted_class
        detection["cnn_confidence"] = confidence
        detection["cnn_probabilities"] = probabilities
        detection["cnn_roi"] = roi
        detection["cnn_bounding_box"] = padded_box

    return detections


def create_roi_montage(detections):
    """Pravi jednu sliku sa svim ROI oblastima poslatim CNN-u."""
    if not detections:
        empty = np.zeros((180, 520, 3), dtype=np.uint8)
        cv2.putText(
            empty,
            "Nema ROI oblasti za CNN klasifikaciju",
            (20, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return empty

    columns = min(ROI_GRID_COLUMNS, len(detections))
    rows = int(np.ceil(len(detections) / columns))

    montage = np.zeros(
        (rows * ROI_TILE_HEIGHT, columns * ROI_TILE_WIDTH, 3),
        dtype=np.uint8,
    )

    for index, detection in enumerate(detections):
        roi = detection.get("cnn_roi")
        if roi is None or roi.size == 0:
            continue

        row = index // columns
        column = index % columns
        x0 = column * ROI_TILE_WIDTH
        y0 = row * ROI_TILE_HEIGHT

        # Rezervisemo gornji deo kartice za sliku, a donji za tekst.
        image_area_height = ROI_TILE_HEIGHT - 55
        available_width = ROI_TILE_WIDTH - 16
        available_height = image_area_height - 16

        roi_height, roi_width = roi.shape[:2]
        scale = min(
            available_width / max(roi_width, 1),
            available_height / max(roi_height, 1),
        )

        resized_width = max(1, int(round(roi_width * scale)))
        resized_height = max(1, int(round(roi_height * scale)))

        roi_resized = cv2.resize(
            roi,
            (resized_width, resized_height),
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
        )

        image_x = x0 + (ROI_TILE_WIDTH - resized_width) // 2
        image_y = y0 + (image_area_height - resized_height) // 2

        montage[
            image_y:image_y + resized_height,
            image_x:image_x + resized_width,
        ] = roi_resized

        predicted_class = detection.get("cnn_class")
        confidence = detection.get("cnn_confidence", 0.0)
        readable_name = DISPLAY_NAMES.get(predicted_class, str(predicted_class))

        accepted = (
            predicted_class is not None
            and confidence >= CNN_MIN_CONFIDENCE
        )
        border_color = (0, 255, 0) if accepted else (0, 165, 255)

        cv2.rectangle(
            montage,
            (x0 + 2, y0 + 2),
            (x0 + ROI_TILE_WIDTH - 3, y0 + ROI_TILE_HEIGHT - 3),
            border_color,
            2,
        )

        cv2.putText(
            montage,
            f"ROI {index + 1}: {readable_name}",
            (x0 + 8, y0 + ROI_TILE_HEIGHT - 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            border_color,
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            montage,
            f"Sigurnost: {confidence * 100:.1f}%",
            (x0 + 8, y0 + ROI_TILE_HEIGHT - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            border_color,
            1,
            cv2.LINE_AA,
        )

    return montage


def show_individual_rois(detections):
    """Otvara svaki ROI u posebnom prozoru."""
    if not SHOW_INDIVIDUAL_ROI_WINDOWS:
        return

    for index, detection in enumerate(detections, start=1):
        roi = detection.get("cnn_roi")
        if roi is None or roi.size == 0:
            continue

        predicted_class = detection.get("cnn_class")
        confidence = detection.get("cnn_confidence", 0.0)
        readable_name = DISPLAY_NAMES.get(predicted_class, str(predicted_class))

        # Povecavamo veoma male isecke samo radi lakseg pregleda.
        preview = roi.copy()
        height, width = preview.shape[:2]
        scale = max(1.0, min(4.0, 180.0 / max(height, width, 1)))
        if scale > 1.0:
            preview = cv2.resize(
                preview,
                (int(round(width * scale)), int(round(height * scale))),
                interpolation=cv2.INTER_NEAREST,
            )

        title = (
            f"ROI {index} - {readable_name} "
            f"({confidence * 100:.1f}%)"
        )
        cv2.imshow(title, preview)


# ============================================================
# PARAMETRI PREPOZNAVANJA OBLIKA
# ============================================================

MIN_SHAPE_AREA = 250

COARSE_APPROXIMATION_FACTOR = 0.025
FINE_APPROXIMATION_FACTOR = 0.008

MIN_SOLIDITY = 0.78

MIN_ELLIPTICITY = 0.92
MIN_OCTAGONALITY = 0.94

MIN_ELLIPSE_FILL_RATIO = 0.70
MIN_CIRCULARITY_FOR_OCTAGON = 0.70

MIN_ROTATED_ASPECT_RATIO = 0.32


# ============================================================
# UČITAVANJE SLIKE
# ============================================================

def load_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        print(f"Putanja slike nije ispravna: {image_path}")
        sys.exit(-1)

    return image


# ============================================================
# ČIŠĆENJE MASKE
# ============================================================

def clean_mask(mask):
    """
    Blago morfološko čišćenje maske.

    Na slikama sa bliskim objektima veći kernel može spojiti
    znak sa stubom, tablom ili susednim crvenim objektom.
    """

    opening_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    closing_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    cleaned_mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        opening_kernel,
        iterations=1
    )

    cleaned_mask = cv2.morphologyEx(
        cleaned_mask,
        cv2.MORPH_CLOSE,
        closing_kernel,
        iterations=1
    )

    return cleaned_mask


# ============================================================
# NORMALIZOVANI HSV
# ============================================================

def rgb_to_hsv_normalized(image):
    """
    OpenCV zapravo učitava sliku kao BGR.

    HSV_FULL daje:
        H: 0-255
        S: 0-255
        V: 0-255
    """

    hsv_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV_FULL
    )

    hue = hsv_image[:, :, 0]
    saturation = hsv_image[:, :, 1]
    value = hsv_image[:, :, 2]

    return hue, saturation, value


# ============================================================
# MASKA CRVENE BOJE
# ============================================================

def create_red_hue_mask(hue, saturation, value):
    """
    Crvena se nalazi na početku i kraju kružnog Hue opsega.
    """

    hue_condition = (
        ((hue > HUE_RED_HIGH_MIN) & (hue <= 255))
        |
        ((hue >= 0) & (hue < HUE_RED_LOW_MAX))
    )

    saturation_condition = (
        saturation >= SATURATION_MIN
    )

    value_condition = (
        (value >= VALUE_MIN)
        &
        (value <= VALUE_MAX)
    )

    final_condition = (
        hue_condition
        &
        saturation_condition
        &
        value_condition
    )

    hue_mask = np.zeros(
        hue.shape,
        dtype=np.uint8
    )

    hue_mask[final_condition] = 255

    return hue_mask


# ============================================================
# MASKA DRUGIH BOJA
# ============================================================

def create_color_hue_mask(
    hue,
    saturation,
    value,
    hue_min_bound,
    hue_max_bound
):
    hue_condition = (
        (hue >= hue_min_bound)
        &
        (hue <= hue_max_bound)
    )

    saturation_condition = (
        saturation >= SATURATION_MIN
    )

    value_condition = (
        (value >= VALUE_MIN)
        &
        (value <= VALUE_MAX)
    )

    final_condition = (
        hue_condition
        &
        saturation_condition
        &
        value_condition
    )

    hue_mask = np.zeros(
        hue.shape,
        dtype=np.uint8
    )

    hue_mask[final_condition] = 255

    return hue_mask


# ============================================================
# FORMIRANJE SEED SLIKE
# ============================================================

def create_seed_image(
    hue_mask,
    subimage_size=SUBIMAGE_SIZE,
    min_white_pixels=MIN_WHITE_PIXELS
):
    """
    Deli masku na blokove.

    Ako blok ima dovoljno belih piksela, postavlja se seme
    na beli piksel koji je najbliži centru bloka.
    """

    height, width = hue_mask.shape

    seed_image = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    for y_start in range(
        0,
        height,
        subimage_size
    ):
        for x_start in range(
            0,
            width,
            subimage_size
        ):
            y_end = min(
                y_start + subimage_size,
                height
            )

            x_end = min(
                x_start + subimage_size,
                width
            )

            sub_image = hue_mask[
                y_start:y_end,
                x_start:x_end
            ]

            white_pixel_count = cv2.countNonZero(
                sub_image
            )

            if white_pixel_count < min_white_pixels:
                continue

            white_positions = np.argwhere(
                sub_image == 255
            )

            if white_positions.size == 0:
                continue

            center_y = (
                sub_image.shape[0] - 1
            ) / 2.0

            center_x = (
                sub_image.shape[1] - 1
            ) / 2.0

            distances = (
                (
                    white_positions[:, 0]
                    - center_y
                ) ** 2
                +
                (
                    white_positions[:, 1]
                    - center_x
                ) ** 2
            )

            nearest_index = int(
                np.argmin(distances)
            )

            seed_y = (
                y_start
                + int(
                    white_positions[
                        nearest_index,
                        0
                    ]
                )
            )

            seed_x = (
                x_start
                + int(
                    white_positions[
                        nearest_index,
                        1
                    ]
                )
            )

            seed_image[
                seed_y,
                seed_x
            ] = 255

    return seed_image


# ============================================================
# REGION GROWING
# ============================================================

def region_grow(hue_mask, seed_image):
    """
    Zadržava povezane komponente koje sadrže
    najmanje jedno seme.
    """

    binary_mask = (
        hue_mask > 0
    ).astype(np.uint8)

    _, labels = cv2.connectedComponents(
        binary_mask,
        connectivity=8
    )

    grown_regions = np.zeros_like(
        hue_mask
    )

    seed_labels = np.unique(
        labels[seed_image > 0]
    )

    seed_labels = seed_labels[
        seed_labels != 0
    ]

    for label in seed_labels:
        grown_regions[
            labels == label
        ] = 255

    return grown_regions


# ============================================================
# PRIMENA MASKE
# ============================================================

def apply_mask(image, mask):
    return cv2.bitwise_and(
        image,
        image,
        mask=mask
    )


# ============================================================
# FLEYEHOV PRVI AFINI MOMENTNI INVARIJANT
# ============================================================

def calculate_affine_moment_invariant(contour):
    """
    I1 = (mu20 * mu02 - mu11^2) / mu00^4
    """

    moments = cv2.moments(
        contour
    )

    mu00 = moments["m00"]

    if mu00 <= 0:
        return None

    mu20 = moments["mu20"]
    mu02 = moments["mu02"]
    mu11 = moments["mu11"]

    denominator = mu00 ** 4

    if denominator <= 0:
        return None

    invariant = (
        mu20 * mu02
        - mu11 ** 2
    ) / denominator

    if invariant <= 0:
        return None

    return float(invariant)


# ============================================================
# SLIČNOST SA IDEALNIM OBLIKOM
# ============================================================

def calculate_shape_similarity(
    invariant,
    ideal_invariant
):
    if invariant is None:
        return 0.0

    if invariant <= 0 or ideal_invariant <= 0:
        return 0.0

    similarity = min(
        invariant / ideal_invariant,
        ideal_invariant / invariant
    )

    return float(
        np.clip(
            similarity,
            0.0,
            1.0
        )
    )


def calculate_ellipticity(invariant):
    """
    Idealni krug/elipsa:
        I1 = 1 / (16*pi^2)
    """

    ideal_circle_invariant = (
        1.0
        / (
            16.0
            * np.pi ** 2
        )
    )

    return calculate_shape_similarity(
        invariant,
        ideal_circle_invariant
    )


def calculate_octagonality(invariant):
    """
    Idealni osmougao:
        I1 = 1 / (15.932*pi^2)
    """

    ideal_octagon_invariant = (
        1.0
        / (
            15.932
            * np.pi ** 2
        )
    )

    return calculate_shape_similarity(
        invariant,
        ideal_octagon_invariant
    )


# ============================================================
# OSOBINE KONTURE
# ============================================================

def calculate_contour_features(contour):
    original_area = cv2.contourArea(
        contour
    )

    if original_area <= 0:
        return None

    # Fleyeh koristi konveksni omotač da bi dobio
    # punu spoljašnju formu objekta.
    convex_hull = cv2.convexHull(
        contour
    )

    area = cv2.contourArea(
        convex_hull
    )

    perimeter = cv2.arcLength(
        convex_hull,
        True
    )

    if area <= 0 or perimeter <= 0:
        return None

    invariant = calculate_affine_moment_invariant(
        convex_hull
    )

    if invariant is None:
        return None

    ellipticity = calculate_ellipticity(
        invariant
    )

    octagonality = calculate_octagonality(
        invariant
    )

    circularity = (
        4.0
        * np.pi
        * area
    ) / (perimeter ** 2)

    circularity = float(
        np.clip(
            circularity,
            0.0,
            1.0
        )
    )

    hull_area = cv2.contourArea(
        convex_hull
    )

    solidity = (
        original_area / hull_area
        if hull_area > 0
        else 0.0
    )

    # Obični pravougaonik koristi se za crtanje.
    x, y, width, height = cv2.boundingRect(
        convex_hull
    )

    aspect_ratio = (
        width / float(height)
        if height > 0
        else 0.0
    )

    # Rotirani pravougaonik je korisniji kod
    # znakova snimljenih pod uglom.
    rotated_rectangle = cv2.minAreaRect(
        convex_hull
    )

    (_, _), (
        rotated_width,
        rotated_height
    ), rotation_angle = rotated_rectangle

    if rotated_width <= 0 or rotated_height <= 0:
        return None

    rotated_short_side = min(
        rotated_width,
        rotated_height
    )

    rotated_long_side = max(
        rotated_width,
        rotated_height
    )

    rotated_aspect_ratio = (
        rotated_short_side
        / rotated_long_side
        if rotated_long_side > 0
        else 0.0
    )

    coarse_epsilon = (
        COARSE_APPROXIMATION_FACTOR
        * perimeter
    )

    fine_epsilon = (
        FINE_APPROXIMATION_FACTOR
        * perimeter
    )

    coarse_approximation = cv2.approxPolyDP(
        convex_hull,
        coarse_epsilon,
        True
    )

    fine_approximation = cv2.approxPolyDP(
        convex_hull,
        fine_epsilon,
        True
    )

    coarse_vertices = len(
        coarse_approximation
    )

    fine_vertices = len(
        fine_approximation
    )

    # --------------------------------------------------------
    # FITOVANJE ELISE
    # --------------------------------------------------------

    ellipse_fill_ratio = 0.0
    fitted_ellipse = None
    ellipse_axis_ratio = 0.0

    if len(convex_hull) >= 5:
        fitted_ellipse = cv2.fitEllipse(
            convex_hull
        )

        (_, _), (
            ellipse_width,
            ellipse_height
        ), _ = fitted_ellipse

        ellipse_area = (
            np.pi
            * ellipse_width
            * ellipse_height
            / 4.0
        )

        if ellipse_area > 0:
            ellipse_fill_ratio = (
                area / ellipse_area
            )

            ellipse_fill_ratio = float(
                np.clip(
                    ellipse_fill_ratio,
                    0.0,
                    1.5
                )
            )

        ellipse_short_axis = min(
            ellipse_width,
            ellipse_height
        )

        ellipse_long_axis = max(
            ellipse_width,
            ellipse_height
        )

        ellipse_axis_ratio = (
            ellipse_short_axis
            / ellipse_long_axis
            if ellipse_long_axis > 0
            else 0.0
        )

    # --------------------------------------------------------
    # NAJMANJI OPISANI KRUG
    # --------------------------------------------------------

    (_, _), enclosing_radius = (
        cv2.minEnclosingCircle(
            convex_hull
        )
    )

    enclosing_circle_area = (
        np.pi
        * enclosing_radius ** 2
    )

    enclosing_circle_ratio = (
        area / enclosing_circle_area
        if enclosing_circle_area > 0
        else 0.0
    )

    return {
        "area": area,
        "original_area": original_area,
        "perimeter": perimeter,
        "invariant": invariant,
        "ellipticity": ellipticity,
        "octagonality": octagonality,
        "circularity": circularity,
        "solidity": solidity,
        "aspect_ratio": aspect_ratio,
        "rotated_aspect_ratio": rotated_aspect_ratio,
        "rotation_angle": rotation_angle,
        "ellipse_fill_ratio": ellipse_fill_ratio,
        "ellipse_axis_ratio": ellipse_axis_ratio,
        "enclosing_circle_ratio": enclosing_circle_ratio,
        "coarse_vertices": coarse_vertices,
        "fine_vertices": fine_vertices,
        "bounding_box": (
            x,
            y,
            width,
            height
        ),
        "convex_hull": convex_hull,
        "coarse_approximation": coarse_approximation,
        "fine_approximation": fine_approximation,
        "fitted_ellipse": fitted_ellipse
    }


# ============================================================
# KLASIFIKACIJA KRUGA I OSMOUGLA
# ============================================================

def classify_circle_or_octagon(features):
    area = features["area"]
    ellipticity = features["ellipticity"]
    octagonality = features["octagonality"]
    circularity = features["circularity"]
    solidity = features["solidity"]

    coarse_vertices = features[
        "coarse_vertices"
    ]

    fine_vertices = features[
        "fine_vertices"
    ]

    rotated_aspect_ratio = features[
        "rotated_aspect_ratio"
    ]

    ellipse_fill_ratio = features[
        "ellipse_fill_ratio"
    ]

    ellipse_axis_ratio = features[
        "ellipse_axis_ratio"
    ]

    if area < MIN_SHAPE_AREA:
        return "Unknown", "premala površina"

    if solidity < MIN_SOLIDITY:
        return "Unknown", "mala solidity vrednost"

    if rotated_aspect_ratio < MIN_ROTATED_ASPECT_RATIO:
        return "Unknown", "objekat je previše izdužen"

    # ========================================================
    # OSMOUGAO — STOP
    # ========================================================

    octagon_condition = (
        7 <= coarse_vertices <= 10
        and 7 <= fine_vertices <= 13
        and octagonality >= MIN_OCTAGONALITY
        and circularity >= MIN_CIRCULARITY_FOR_OCTAGON
    )

    if octagon_condition:
        return "Stop", "osmougaona geometrija"

    # ========================================================
    # KRUG/ELIPSA — KANDIDAT ZA OGRANIČENJE BRZINE
    # ========================================================

    ellipse_condition = (
        ellipticity >= MIN_ELLIPTICITY
        and ellipse_fill_ratio >= MIN_ELLIPSE_FILL_RATIO
        and ellipse_axis_ratio >= 0.28
        and fine_vertices >= 10
    )

    if ellipse_condition:
        return (
            "Speed limit candidate",
            "kružna ili eliptična geometrija"
        )

    return "Unknown", "geometrijski uslovi nisu ispunjeni"


# ============================================================
# PRONALAŽENJE KANDIDATA
# ============================================================

def analyse_all_contours(red_mask):
    """
    Vraća sve konture sa osobinama i rezultatom
    klasifikacije, uključujući odbačene objekte.
    """

    contours, _ = cv2.findContours(
        red_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    candidates = []

    for contour in contours:
        features = calculate_contour_features(
            contour
        )

        if features is None:
            continue

        sign_type, reason = (
            classify_circle_or_octagon(
                features
            )
        )

        candidates.append({
            "sign_type": sign_type,
            "reason": reason,
            "contour": contour,
            "features": features
        })

    return candidates


def detect_circle_and_octagon_signs(red_mask):
    all_candidates = analyse_all_contours(
        red_mask
    )

    detections = [
        candidate
        for candidate in all_candidates
        if candidate["sign_type"] != "Unknown"
    ]

    return detections


# ============================================================
# CRTANJE REZULTATA
# ============================================================

def draw_shape_detections(
    image,
    detections
):
    output = image.copy()

    for detection in detections:
        sign_type = detection["sign_type"]
        features = detection["features"]

        x, y, width, height = detection.get(
            "cnn_bounding_box",
            features["bounding_box"],
        )

        cnn_class = detection.get("cnn_class")
        cnn_confidence = detection.get("cnn_confidence", 0.0)
        roi_index = detection.get("roi_index", 0)

        accepted = (
            cnn_class is not None
            and cnn_confidence >= CNN_MIN_CONFIDENCE
        )
        box_color = (0, 255, 0) if accepted else (0, 165, 255)

        cv2.rectangle(
            output,
            (x, y),
            (x + width, y + height),
            box_color,
            2
        )

        cv2.drawContours(
            output,
            [features["fine_approximation"]],
            -1,
            (255, 0, 255),
            2
        )

        fitted_ellipse = features["fitted_ellipse"]
        if fitted_ellipse is not None:
            cv2.ellipse(
                output,
                fitted_ellipse,
                (255, 255, 0),
                1
            )

        readable_name = DISPLAY_NAMES.get(cnn_class, str(cnn_class))
        if accepted:
            label = f"ROI {roi_index}: {readable_name} {cnn_confidence * 100:.1f}%"
        elif cnn_class is not None:
            label = f"ROI {roi_index}: Nesigurno {readable_name} {cnn_confidence * 100:.1f}%"
        else:
            label = f"ROI {roi_index}: CNN greska"

        shape_label = f"Oblik: {sign_type}"
        measures_label = (
            f"E={features['ellipticity']:.3f}, "
            f"O={features['octagonality']:.3f}"
        )

        cv2.putText(
            output, label, (x, max(20, y - 42)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.50, box_color, 2, cv2.LINE_AA
        )
        cv2.putText(
            output, shape_label, (x, max(38, y - 23)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.40, box_color, 1, cv2.LINE_AA
        )
        cv2.putText(
            output, measures_label, (x, max(56, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, box_color, 1, cv2.LINE_AA
        )

    return output


# ============================================================
# DIJAGNOSTIČKI ISPIS
# ============================================================

def print_all_contour_candidates(red_mask):
    candidates = analyse_all_contours(
        red_mask
    )

    print("\nSVI PRONAĐENI CRVENI OBJEKTI")
    print("-" * 180)

    if not candidates:
        print("Nije pronađena nijedna odgovarajuća kontura.")
        return

    for index, candidate in enumerate(
        candidates,
        start=1
    ):
        features = candidate["features"]

        print(
            f"{index}. "
            f"rezultat={candidate['sign_type']}, "
            f"razlog={candidate['reason']}, "
            f"area={features['area']:.1f}, "
            f"coarseV={features['coarse_vertices']}, "
            f"fineV={features['fine_vertices']}, "
            f"E={features['ellipticity']:.4f}, "
            f"O={features['octagonality']:.4f}, "
            f"C={features['circularity']:.4f}, "
            f"solidity={features['solidity']:.4f}, "
            f"rotAR={features['rotated_aspect_ratio']:.4f}, "
            f"ellipseAR={features['ellipse_axis_ratio']:.4f}, "
            f"ellipseFill={features['ellipse_fill_ratio']:.4f}"
        )


def print_shape_detections(detections):
    print("\nPREPOZNATI ZNAKOVI")
    print("-" * 120)

    if not detections:
        print(
            "Nisu pronađeni kružni ili osmougaoni znakovi."
        )
        return

    print(
        f"Broj pronađenih znakova: "
        f"{len(detections)}"
    )

    for index, detection in enumerate(
        detections,
        start=1
    ):
        features = detection["features"]

        cnn_class = detection.get("cnn_class", "nije klasifikovano")
        cnn_confidence = detection.get("cnn_confidence", 0.0)

        print(
            f"{index}. "
            f"oblik={detection['sign_type']}, "
            f"CNN={cnn_class}, "
            f"sigurnost={cnn_confidence * 100:.2f}%, "
            f"povrsina={features['area']:.2f}, "
            f"temena="
            f"{features['coarse_vertices']}/"
            f"{features['fine_vertices']}, "
            f"ellipticity="
            f"{features['ellipticity']:.4f}, "
            f"octagonality="
            f"{features['octagonality']:.4f}, "
            f"circularity="
            f"{features['circularity']:.4f}, "
            f"solidity="
            f"{features['solidity']:.4f}, "
            f"ellipse_fill="
            f"{features['ellipse_fill_ratio']:.4f}"
        )


# ============================================================
# SEGMENTACIJA BOJE
# ============================================================

def shadow_highlight_invariant_segmentation(
    image,
    use_cleaning=False,
    color="red"
):
    hue, saturation, value = (
        rgb_to_hsv_normalized(
            image
        )
    )

    color = color.lower()

    if color == "red":
        hue_mask = create_red_hue_mask(
            hue,
            saturation,
            value
        )

    elif color in SEGMENTATION_COLORS:
        hue_mask = create_color_hue_mask(
            hue,
            saturation,
            value,
            SEGMENTATION_COLORS[color][0],
            SEGMENTATION_COLORS[color][1]
        )

    else:
        raise ValueError(
            f"Nepodržana boja: {color}. "
            "Dozvoljene boje su red, blue i yellow."
        )

    seed_image = create_seed_image(
        hue_mask
    )

    grown_mask = region_grow(
        hue_mask,
        seed_image
    )

    if use_cleaning:
        final_mask = clean_mask(
            grown_mask
        )
    else:
        final_mask = grown_mask

    result = apply_mask(
        image,
        final_mask
    )

    return (
        hue_mask,
        seed_image,
        grown_mask,
        final_mask,
        result
    )


# ============================================================
# GLAVNI PROGRAM
# ============================================================

def main():
    if len(sys.argv) != 2:
        print(
            "Upotreba: "
            "python boja.py putanja_do_slike"
        )
        sys.exit(1)

    model = load_cnn_model()

    image = load_image(
        sys.argv[1]
    )

    # Za STOP i znakove ograničenja brzine koristi se
    # segmentacija crvenog oboda.
    (
        hue_mask,
        seed_image,
        grown_mask,
        final_mask,
        color_result
    ) = shadow_highlight_invariant_segmentation(
        image,
        # Region growing već odbacuje male objekte.
        # False sprečava spajanje bliskih objekata.
        use_cleaning=False,
        color="red"
    )

    # Ispis svih kandidata pokazuje zašto je neki
    # objekat prihvaćen ili odbačen.
    print_all_contour_candidates(
        final_mask
    )

    detections = detect_circle_and_octagon_signs(
        final_mask
    )

    detections = classify_detections_with_cnn(
        image,
        detections,
        model
    )

    detections = [
        detection
        for detection in detections
        if detection.get("cnn_class") is not None
        and detection.get("cnn_confidence", 0.0) >= CNN_MIN_CONFIDENCE
    ]

    roi_montage = create_roi_montage(
        detections
    )

    shape_result = draw_shape_detections(
        image,
        detections
    )

    print_shape_detections(
        detections
    )

    # Seed tačke se uvećavaju samo radi prikaza.
    seed_display = cv2.dilate(
        seed_image,
        np.ones(
            (5, 5),
            dtype=np.uint8
        ),
        iterations=1
    )

    cv2.imshow(
        "Originalna slika",
        image
    )

    cv2.imshow(
        "Crvena Hue maska",
        hue_mask
    )

    cv2.imshow(
        "Seed slika",
        seed_display
    )

    cv2.imshow(
        "Region growing maska",
        grown_mask
    )

    cv2.imshow(
        "Konacna crvena maska",
        final_mask
    )

    cv2.imshow(
        "Segmentirana crvena boja",
        color_result
    )

    cv2.imshow(
        "Detekcija oblika i CNN klasifikacija",
        shape_result
    )

    cv2.imshow(
        "Sve ROI oblasti prosledjene neuronskoj mrezi",
        roi_montage
    )

    show_individual_rois(
        detections
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
