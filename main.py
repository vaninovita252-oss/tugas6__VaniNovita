import cv2
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path


# ============================================================
# KONFIGURASI
# ============================================================

PRESENT_FOLDER = Path("data/SIGNATURE PRESENT")
ABSENT_FOLDER = Path("data/SIGNATURE ABSENT")

RESULT_FOLDER = Path("results")

GRAYSCALE_FOLDER = RESULT_FOLDER / "01_GRAYSCALE"
GLOBAL_FOLDER = RESULT_FOLDER / "02_GLOBAL_THRESHOLD"
ADAPTIVE_FOLDER = RESULT_FOLDER / "03_ADAPTIVE_THRESHOLD"
COMPARISON_FOLDER = RESULT_FOLDER / "04_COMPARISON"
OPENING_FOLDER = RESULT_FOLDER / "05_OPENING"
CLOSING_FOLDER = RESULT_FOLDER / "06_CLOSING"
SEGMENTATION_FOLDER = RESULT_FOLDER / "07_SEGMENTATION"
ANALYSIS_FOLDER = RESULT_FOLDER / "08_ANALYSIS"
CLASSIFICATION_FOLDER = RESULT_FOLDER / "09_CLASSIFICATION"


# ============================================================
# PARAMETER THRESHOLDING
# ============================================================

GLOBAL_THRESHOLD_VALUE = 127

ADAPTIVE_BLOCK_SIZE = 11
ADAPTIVE_C = 2


# ============================================================
# PARAMETER MORPHOLOGY
# ============================================================

KERNEL_SIZE = 3


# ============================================================
# PARAMETER KLASIFIKASI
# ============================================================

# Jika foreground ratio >= 2%
# maka dianggap SIGNATURE PRESENT.
#
# Nilai ini dapat disesuaikan setelah melihat hasil pengujian.

CLASSIFICATION_THRESHOLD = 0.02


# ============================================================
# BUAT SEMUA FOLDER
# ============================================================

folders = [
    GRAYSCALE_FOLDER,
    GLOBAL_FOLDER,
    ADAPTIVE_FOLDER,
    COMPARISON_FOLDER,
    OPENING_FOLDER,
    CLOSING_FOLDER,
    SEGMENTATION_FOLDER,
    ANALYSIS_FOLDER,
    CLASSIFICATION_FOLDER
]

for folder in folders:
    folder.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# EXTENSION GAMBAR
# ============================================================

IMAGE_EXTENSIONS = [
    "*.jpg",
    "*.jpeg",
    "*.png",
    "*.bmp",
    "*.tif",
    "*.tiff"
]


# ============================================================
# MENCARI GAMBAR
# ============================================================

def get_images(folder):

    images = []

    if not folder.exists():
        return images

    for extension in IMAGE_EXTENSIONS:

        images.extend(
            folder.glob(extension)
        )

    return sorted(images)


# ============================================================
# SIMPAN GAMBAR
# ============================================================

def save_image(image, path):

    cv2.imwrite(
        str(path),
        image
    )


# ============================================================
# MORPHOLOGICAL OPENING
# ============================================================

def apply_opening(binary):

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            KERNEL_SIZE,
            KERNEL_SIZE
        )
    )

    result = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    return result


# ============================================================
# MORPHOLOGICAL CLOSING
# ============================================================

def apply_closing(binary):

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            KERNEL_SIZE,
            KERNEL_SIZE
        )
    )

    result = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1
    )

    return result


# ============================================================
# MENGHITUNG KUALITAS HASIL THRESHOLD
# ============================================================
#
# Fungsi ini digunakan untuk menentukan:
#
# GLOBAL lebih baik
# atau
# ADAPTIVE lebih baik
#
# Penilaian dilakukan berdasarkan:
#
# 1. foreground ratio
# 2. komponen terbesar
# 3. jumlah komponen kecil/noise
#
# Tidak menggunakan label PRESENT/ABSENT.
# ============================================================

def calculate_quality(binary):

    total_pixels = binary.shape[0] * binary.shape[1]

    foreground_pixels = cv2.countNonZero(
        binary
    )

    foreground_ratio = (
        foreground_pixels /
        total_pixels
    )


    # --------------------------------------------------------
    # CONNECTED COMPONENT
    # --------------------------------------------------------

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8
    )


    component_areas = []

    for i in range(1, num_labels):

        area = stats[i, cv2.CC_STAT_AREA]

        component_areas.append(area)


    # Tidak ada foreground
    if len(component_areas) == 0:

        largest_component_ratio = 0

        small_components = 0

    else:

        largest_area = max(
            component_areas
        )

        largest_component_ratio = (
            largest_area /
            total_pixels
        )

        small_components = sum(
            1
            for area in component_areas
            if area < 20
        )


    # --------------------------------------------------------
    # SCORE
    # --------------------------------------------------------
    #
    # Foreground yang terlalu sedikit:
    # kemungkinan tidak mendapatkan objek.
    #
    # Foreground yang terlalu banyak:
    # kemungkinan noise/background ikut masuk.
    #
    # Ratio yang lebih masuk akal mendapatkan score lebih tinggi.
    # --------------------------------------------------------

    if foreground_ratio < 0.001:

        ratio_score = 0

    elif foreground_ratio <= 0.20:

        ratio_score = 1

    else:

        ratio_score = max(
            0,
            1 - (
                foreground_ratio - 0.20
            )
        )


    # Komponen terbesar menunjukkan apakah
    # objek utama cukup terhubung.

    component_score = min(
        largest_component_ratio * 20,
        1
    )


    # Penalti noise.

    noise_penalty = min(
        small_components / 100,
        1
    )


    score = (
        (ratio_score * 0.50)
        +
        (component_score * 0.50)
        -
        (noise_penalty * 0.20)
    )


    return {
        "foreground_pixels": foreground_pixels,
        "foreground_ratio": foreground_ratio,
        "largest_component_ratio": largest_component_ratio,
        "small_components": small_components,
        "score": score
    }


# ============================================================
# NAMA FILE TANPA EXTENSION
# ============================================================

def clean_name(path):

    return path.stem


# ============================================================
# AMBIL DATA
# ============================================================

present_images = get_images(
    PRESENT_FOLDER
)

absent_images = get_images(
    ABSENT_FOLDER
)


# ============================================================
# INFORMASI AWAL
# ============================================================

print()
print("=" * 80)
print("              SISTEM DETEKSI TANDA TANGAN")
print("=" * 80)
print()

print(
    "SIGNATURE PRESENT :",
    len(present_images),
    "citra"
)

print(
    "SIGNATURE ABSENT  :",
    len(absent_images),
    "citra"
)

print()


# ============================================================
# CEK DATA
# ============================================================

if len(present_images) == 0:

    print(
        "ERROR: Folder SIGNATURE PRESENT kosong."
    )

    print(
        "Masukkan gambar ke:"
    )

    print(
        PRESENT_FOLDER
    )

    exit()


if len(absent_images) == 0:

    print(
        "ERROR: Folder SIGNATURE ABSENT kosong."
    )

    print(
        "Masukkan gambar ke:"
    )

    print(
        ABSENT_FOLDER
    )

    exit()


# ============================================================
# GABUNGKAN DATA PRESENT + ABSENT
# ============================================================

dataset = []


for image_path in present_images:

    dataset.append(
        {
            "path": image_path,
            "actual": "SIGNATURE PRESENT"
        }
    )


for image_path in absent_images:

    dataset.append(
        {
            "path": image_path,
            "actual": "SIGNATURE ABSENT"
        }
    )


# ============================================================
# PENAMPUNG HASIL
# ============================================================

results = []


# ============================================================
# PROSES SEMUA CITRA
# ============================================================

print("=" * 80)
print("                    PROSES CITRA")
print("=" * 80)
print()


for number, item in enumerate(
    dataset,
    start=1
):

    image_path = item["path"]

    actual_label = item["actual"]

    print(
        f"[{number}/{len(dataset)}] "
        f"{image_path.name}"
    )


    # ========================================================
    # BACA CITRA
    # ========================================================

    image = cv2.imread(
        str(image_path)
    )


    if image is None:

        print(
            "    ERROR: gambar tidak dapat dibaca."
        )

        continue


    # ========================================================
    # GRAYSCALE
    # ========================================================

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )


    # ========================================================
    # GLOBAL THRESHOLD
    # ========================================================

    _, global_threshold = cv2.threshold(
        gray,
        GLOBAL_THRESHOLD_VALUE,
        255,
        cv2.THRESH_BINARY_INV
    )


    # ========================================================
    # ADAPTIVE THRESHOLD
    # ========================================================

    adaptive_threshold = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        ADAPTIVE_BLOCK_SIZE,
        ADAPTIVE_C
    )


    # ========================================================
    # ANALISIS GLOBAL
    # ========================================================

    global_quality = calculate_quality(
        global_threshold
    )


    # ========================================================
    # ANALISIS ADAPTIVE
    # ========================================================

    adaptive_quality = calculate_quality(
        adaptive_threshold
    )


    # ========================================================
    # PILIH THRESHOLD TERBAIK
    # ========================================================

    if (
        global_quality["score"]
        >=
        adaptive_quality["score"]
    ):

        selected_method = "GLOBAL THRESHOLD"

        selected_threshold = global_threshold

        selected_quality = global_quality

    else:

        selected_method = "ADAPTIVE THRESHOLD"

        selected_threshold = adaptive_threshold

        selected_quality = adaptive_quality


    # ========================================================
    # OPENING
    # ========================================================

    opening = apply_opening(
        selected_threshold
    )


    # ========================================================
    # CLOSING
    # ========================================================

    closing = apply_closing(
        opening
    )


    # ========================================================
    # HASIL SEGMENTASI
    # ========================================================

    segmentation = closing


    # ========================================================
    # FOREGROUND PIXEL
    # ========================================================

    foreground_pixels = cv2.countNonZero(
        segmentation
    )


    # ========================================================
    # TOTAL PIXEL
    # ========================================================

    total_pixels = (
        segmentation.shape[0]
        *
        segmentation.shape[1]
    )


    # ========================================================
    # FOREGROUND RATIO
    # ========================================================

    foreground_ratio = (
        foreground_pixels /
        total_pixels
    )


    foreground_percentage = (
        foreground_ratio *
        100
    )


    # ========================================================
    # KLASIFIKASI
    # ========================================================

    if (
        foreground_ratio
        >=
        CLASSIFICATION_THRESHOLD
    ):

        prediction = "SIGNATURE PRESENT"

    else:

        prediction = "SIGNATURE ABSENT"


    # ========================================================
    # CEK BENAR / SALAH
    # ========================================================

    if prediction == actual_label:

        status = "CORRECT"

    else:

        status = "INCORRECT"


    # ========================================================
    # NAMA
    # ========================================================

    name = clean_name(
        image_path
    )


    # ========================================================
    # SIMPAN GRAYSCALE
    # ========================================================

    save_image(
        gray,
        GRAYSCALE_FOLDER /
        f"{name}_grayscale.png"
    )


    # ========================================================
    # SIMPAN GLOBAL
    # ========================================================

    save_image(
        global_threshold,
        GLOBAL_FOLDER /
        f"{name}_global.png"
    )


    # ========================================================
    # SIMPAN ADAPTIVE
    # ========================================================

    save_image(
        adaptive_threshold,
        ADAPTIVE_FOLDER /
        f"{name}_adaptive.png"
    )


    # ========================================================
    # SIMPAN OPENING
    # ========================================================

    save_image(
        opening,
        OPENING_FOLDER /
        f"{name}_opening.png"
    )


    # ========================================================
    # SIMPAN CLOSING
    # ========================================================

    save_image(
        closing,
        CLOSING_FOLDER /
        f"{name}_closing.png"
    )


    # ========================================================
    # SIMPAN SEGMENTASI
    # ========================================================

    save_image(
        segmentation,
        SEGMENTATION_FOLDER /
        f"{name}_segmentation.png"
    )


    # ========================================================
    # SIMPAN DATA HASIL
    # ========================================================

    results.append(
        {
            "name": image_path.name,

            "actual": actual_label,

            "prediction": prediction,

            "status": status,

            "gray": gray,

            "global": global_threshold,

            "adaptive": adaptive_threshold,

            "selected": selected_threshold,

            "opening": opening,

            "closing": closing,

            "segmentation": segmentation,

            "selected_method":
                selected_method,

            "global_score":
                global_quality["score"],

            "adaptive_score":
                adaptive_quality["score"],

            "global_foreground":
                global_quality["foreground_pixels"],

            "adaptive_foreground":
                adaptive_quality["foreground_pixels"],

            "foreground_pixels":
                foreground_pixels,

            "foreground_percentage":
                foreground_percentage
        }
    )


    print(
        "    Global Score   :",
        f"{global_quality['score']:.4f}"
    )

    print(
        "    Adaptive Score :",
        f"{adaptive_quality['score']:.4f}"
    )

    print(
        "    Terpilih       :",
        selected_method
    )

    print(
        "    Foreground     :",
        foreground_pixels,
        "px"
    )

    print(
        "    Ratio          :",
        f"{foreground_percentage:.2f}%"
    )

    print(
        "    Prediksi       :",
        prediction
    )

    print(
        "    Status         :",
        status
    )

    print()


# ============================================================
# BUAT GAMBAR COMPARISON
# ============================================================

print("=" * 80)
print("                 MEMBUAT COMPARISON")
print("=" * 80)
print()


for number, result in enumerate(
    results,
    start=1
):

    gray = result["gray"]

    global_threshold = result["global"]

    adaptive_threshold = result["adaptive"]

    selected = result["selected"]

    opening = result["opening"]

    closing = result["closing"]

    actual = result["actual"]

    prediction = result["prediction"]

    status = result["status"]

    selected_method = result[
        "selected_method"
    ]

    global_score = result[
        "global_score"
    ]

    adaptive_score = result[
        "adaptive_score"
    ]

    foreground_pixels = result[
        "foreground_pixels"
    ]

    foreground_percentage = result[
        "foreground_percentage"
    ]


    # ========================================================
    # FIGURE
    # ========================================================

    fig = plt.figure(
        figsize=(13, 15),
        facecolor="white"
    )


    grid = fig.add_gridspec(
        7,
        6,
        height_ratios=[
            0.7,
            2.2,
            2.0,
            1.7,
            2.0,
            2.0,
            2.0
        ],
        hspace=0.8,
        wspace=0.25
    )


    # ========================================================
    # JUDUL
    # ========================================================

    ax_title = fig.add_subplot(
        grid[0, :]
    )

    ax_title.axis("off")

    ax_title.text(
        0.5,
        0.65,
        "ANALISIS DETEKSI TANDA TANGAN",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold"
    )

    ax_title.text(
        0.5,
        0.15,
        result["name"],
        ha="center",
        va="center",
        fontsize=11
    )


    # ========================================================
    # GRAYSCALE
    # ========================================================

    ax = fig.add_subplot(
        grid[1, :]
    )

    ax.imshow(
        gray,
        cmap="gray"
    )

    ax.set_title(
        "GRAYSCALE",
        fontsize=12,
        fontweight="bold"
    )

    ax.axis("off")


    # ========================================================
    # GLOBAL
    # ========================================================

    ax = fig.add_subplot(
        grid[2, 0:3]
    )

    ax.imshow(
        global_threshold,
        cmap="gray"
    )

    ax.set_title(
        "GLOBAL THRESHOLD",
        fontsize=11,
        fontweight="bold"
    )

    ax.axis("off")


    # ========================================================
    # ADAPTIVE
    # ========================================================

    ax = fig.add_subplot(
        grid[2, 3:6]
    )

    ax.imshow(
        adaptive_threshold,
        cmap="gray"
    )

    ax.set_title(
        "ADAPTIVE THRESHOLD",
        fontsize=11,
        fontweight="bold"
    )

    ax.axis("off")


    # ========================================================
    # PERBANDINGAN
    # ========================================================

    ax = fig.add_subplot(
        grid[3, :]
    )

    ax.axis("off")


    comparison_text = (
        "PERBANDINGAN THRESHOLDING\n\n"

        f"Global Score   : {global_score:.4f}\n"
        f"Adaptive Score : {adaptive_score:.4f}\n\n"

        f"METODE TERBAIK : {selected_method}\n\n"

        "Metode dengan hasil segmentasi yang lebih baik "
        "dipilih untuk proses morphology."
    )


    ax.text(
        0.5,
        0.5,
        comparison_text,
        ha="center",
        va="center",
        fontsize=10,
        bbox=dict(
            boxstyle="round,pad=0.8",
            facecolor="white",
            edgecolor="black"
        )
    )


    # ========================================================
    # THRESHOLD TERPILIH
    # ========================================================

    ax = fig.add_subplot(
        grid[4, 0:2]
    )

    ax.imshow(
        selected,
        cmap="gray"
    )

    ax.set_title(
        f"THRESHOLD TERPILIH\n{selected_method}",
        fontsize=10,
        fontweight="bold"
    )

    ax.axis("off")


    # ========================================================
    # OPENING
    # ========================================================

    ax = fig.add_subplot(
        grid[4, 2:4]
    )

    ax.imshow(
        opening,
        cmap="gray"
    )

    ax.set_title(
        "OPENING",
        fontsize=11,
        fontweight="bold"
    )

    ax.axis("off")


    # ========================================================
    # CLOSING
    # ========================================================

    ax = fig.add_subplot(
        grid[4, 4:6]
    )

    ax.imshow(
        closing,
        cmap="gray"
    )

    ax.set_title(
        "CLOSING",
        fontsize=11,
        fontweight="bold"
    )

    ax.axis("off")


    # ========================================================
    # HASIL SEGMENTASI
    # ========================================================

    ax = fig.add_subplot(
        grid[5, :]
    )

    ax.axis("off")


    analysis_text = (
        "HASIL SEGMENTASI\n\n"

        f"Foreground Pixel : "
        f"{foreground_pixels} px\n\n"

        f"Foreground Ratio : "
        f"{foreground_percentage:.2f}%\n\n"

        "ATURAN KLASIFIKASI\n"

        f"Ratio >= "
        f"{CLASSIFICATION_THRESHOLD * 100:.2f}% "
        f"→ SIGNATURE PRESENT\n"

        f"Ratio < "
        f"{CLASSIFICATION_THRESHOLD * 100:.2f}% "
        f"→ SIGNATURE ABSENT"
    )


    ax.text(
        0.5,
        0.5,
        analysis_text,
        ha="center",
        va="center",
        fontsize=11,
        bbox=dict(
            boxstyle="round,pad=0.8",
            facecolor="white",
            edgecolor="black"
        )
    )


    # ========================================================
    # HASIL PENGUJIAN
    # ========================================================

    ax = fig.add_subplot(
        grid[6, :]
    )

    ax.axis("off")


    result_text = (
        "HASIL PENGUJIAN\n\n"

        f"Label Aktual : {actual}\n\n"

        f"Prediksi     : {prediction}\n\n"

        f"Status       : {status}"
    )


    ax.text(
        0.5,
        0.5,
        result_text,
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=1.0",
            facecolor="white",
            edgecolor="black",
            linewidth=1.5
        )
    )


    # ========================================================
    # LAYOUT
    # ========================================================

    plt.tight_layout(
        rect=[
            0.03,
            0.02,
            0.97,
            0.98
        ]
    )


    # ========================================================
    # NAMA OUTPUT
    # ========================================================

    if actual == "SIGNATURE PRESENT":

        prefix = "PRESENT"

    else:

        prefix = "ABSENT"


    output_name = (
        f"{number:02d}_"
        f"{prefix}_"
        f"{clean_name(Path(result['name']))}"
        "_comparison.png"
    )


    output_path = (
        COMPARISON_FOLDER /
        output_name
    )


    # ========================================================
    # SIMPAN
    # ========================================================

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close()


# ============================================================
# BUAT HASIL ANALISIS TXT
# ============================================================

analysis_file = (
    ANALYSIS_FOLDER /
    "hasil_analisis.txt"
)


with open(
    analysis_file,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "=" * 125 + "\n"
    )

    file.write(
        "                    HASIL ANALISIS DETEKSI TANDA TANGAN\n"
    )

    file.write(
        "=" * 125 + "\n\n"
    )


    # --------------------------------------------------------
    # TAHAP
    # --------------------------------------------------------

    file.write(
        "ALUR PENGOLAHAN\n"
    )

    file.write(
        "-" * 100 + "\n"
    )

    file.write(
        "1. Grayscale\n"
    )

    file.write(
        "2. Global Threshold\n"
    )

    file.write(
        "3. Adaptive Threshold\n"
    )

    file.write(
        "4. Perbandingan Global dan Adaptive\n"
    )

    file.write(
        "5. Memilih threshold terbaik\n"
    )

    file.write(
        "6. Opening pada threshold terpilih\n"
    )

    file.write(
        "7. Closing pada hasil Opening\n"
    )

    file.write(
        "8. Segmentasi\n"
    )

    file.write(
        "9. Perhitungan Foreground Pixel\n"
    )

    file.write(
        "10. Perhitungan Foreground Ratio\n"
    )

    file.write(
        "11. Klasifikasi Signature Present / Signature Absent\n\n"
    )


    # --------------------------------------------------------
    # PARAMETER
    # --------------------------------------------------------

    file.write(
        "PARAMETER\n"
    )

    file.write(
        "-" * 100 + "\n"
    )

    file.write(
        f"Global Threshold : "
        f"{GLOBAL_THRESHOLD_VALUE}\n"
    )

    file.write(
        f"Adaptive Block Size : "
        f"{ADAPTIVE_BLOCK_SIZE}\n"
    )

    file.write(
        f"Adaptive C : "
        f"{ADAPTIVE_C}\n"
    )

    file.write(
        f"Kernel Morphology : "
        f"{KERNEL_SIZE} x {KERNEL_SIZE}\n"
    )

    file.write(
        f"Classification Threshold : "
        f"{CLASSIFICATION_THRESHOLD * 100:.2f}%\n\n"
    )


    # --------------------------------------------------------
    # TABEL PRESENT DAN ABSENT (DIPISAH)
    # --------------------------------------------------------

    def tulis_tabel(kategori, data):
        file.write("=" * 125 + "\n")
        file.write(f"                 HASIL ANALISIS {kategori}\n")
        file.write("=" * 125 + "\n")

        file.write(
            f"{'No':<4}"
            f"{'Nama Citra':<35}"
            f"{'Aktual':<20}"
            f"{'Global':>10}"
            f"{'Adaptive':>10}"
            f"{'Terpilih':<22}"
            f"{'FG Pixel':>12}"
            f"{'Ratio':>10}"
            f"{'Prediksi':<20}"
            f"{'Status':<10}\n"
        )
        file.write("-" * 125 + "\n")

        for number, result in enumerate(data, start=1):
            name = result["name"]
            if len(name) > 32:
                name = name[:29] + "..."

            actual = result["actual"].replace("SIGNATURE ", "")
            prediction = result["prediction"].replace("SIGNATURE ", "")

            file.write(
                f"{number:<4}"
                f"{name:<35}"
                f"{actual:<20}"
                f"{result['global_score']:>10.4f}"
                f"{result['adaptive_score']:>10.4f}"
                f"{result['selected_method']:<22}"
                f"{result['foreground_pixels']:>12}"
                f"{result['foreground_percentage']:>9.2f}%"
                f"{prediction:<20}"
                f"{result['status']:<10}\n"
            )

        file.write("=" * 125 + "\n\n")

    present_results = [
        result for result in results
        if result["actual"] == "SIGNATURE PRESENT"
    ]
    absent_results = [
        result for result in results
        if result["actual"] == "SIGNATURE ABSENT"
    ]

    tulis_tabel("SIGNATURE PRESENT", present_results)
    tulis_tabel("SIGNATURE ABSENT", absent_results)


    # --------------------------------------------------------
    # AKURASI
    # --------------------------------------------------------

    correct = sum(
        1
        for result in results
        if result["status"] == "CORRECT"
    )


    incorrect = sum(
        1
        for result in results
        if result["status"] == "INCORRECT"
    )


    total = len(results)


    if total > 0:

        accuracy = (
            correct /
            total
        ) * 100

    else:

        accuracy = 0


    file.write("\n")

    file.write(
        "RINGKASAN PENGUJIAN\n"
    )

    file.write(
        "-" * 100 + "\n"
    )

    file.write(
        f"Jumlah PRESENT : "
        f"{len(present_images)}\n"
    )

    file.write(
        f"Jumlah ABSENT  : "
        f"{len(absent_images)}\n"
    )

    file.write(
        f"Total Citra    : "
        f"{total}\n"
    )

    file.write(
        f"Benar          : "
        f"{correct}\n"
    )

    file.write(
        f"Salah          : "
        f"{incorrect}\n"
    )

    file.write(
        f"Akurasi        : "
        f"{accuracy:.2f}%\n\n"
    )


    # --------------------------------------------------------
    # PENJELASAN
    # --------------------------------------------------------

    file.write(
        "PENJELASAN\n"
    )

    file.write(
        "-" * 100 + "\n"
    )

    file.write(
        "Grayscale mengubah citra menjadi satu kanal "
        "intensitas agar proses segmentasi lebih mudah.\n\n"
    )

    file.write(
        "Global Threshold menggunakan satu nilai threshold "
        "yang sama pada seluruh area citra.\n\n"
    )

    file.write(
        "Adaptive Threshold menentukan nilai threshold "
        "berdasarkan kondisi lokal citra.\n\n"
    )

    file.write(
        "Kedua metode thresholding dibandingkan terlebih "
        "dahulu. Metode dengan nilai kualitas segmentasi "
        "lebih tinggi dipilih sebagai threshold terbaik.\n\n"
    )

    file.write(
        "Hanya threshold yang terpilih yang diteruskan "
        "ke proses Opening dan Closing.\n\n"
    )

    file.write(
        "Opening digunakan untuk mengurangi noise atau "
        "objek kecil yang tidak diperlukan.\n\n"
    )

    file.write(
        "Closing digunakan untuk menutup celah kecil "
        "pada objek hasil segmentasi.\n\n"
    )

    file.write(
        "Foreground Pixel merupakan jumlah piksel yang "
        "terdeteksi sebagai objek foreground setelah "
        "proses Opening dan Closing.\n\n"
    )

    file.write(
        "Foreground Ratio merupakan perbandingan jumlah "
        "foreground pixel dengan seluruh pixel citra.\n\n"
    )

    file.write(
        f"Jika Foreground Ratio >= "
        f"{CLASSIFICATION_THRESHOLD * 100:.2f}% maka "
        f"hasil dikategorikan sebagai SIGNATURE PRESENT.\n\n"
    )

    file.write(
        f"Jika Foreground Ratio < "
        f"{CLASSIFICATION_THRESHOLD * 100:.2f}% maka "
        f"hasil dikategorikan sebagai SIGNATURE ABSENT.\n"
    )


# ============================================================
# HASIL KLASIFIKASI TXT
# ============================================================

classification_file = (
    CLASSIFICATION_FOLDER /
    "hasil_klasifikasi.txt"
)


with open(
    classification_file,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "=" * 100 + "\n"
    )

    file.write(
        "                  HASIL KLASIFIKASI TANDA TANGAN\n"
    )

    file.write(
        "=" * 100 + "\n\n"
    )


    file.write(
        "ATURAN:\n"
    )

    file.write(
        f"Ratio >= "
        f"{CLASSIFICATION_THRESHOLD * 100:.2f}% "
        f"-> SIGNATURE PRESENT\n"
    )

    file.write(
        f"Ratio < "
        f"{CLASSIFICATION_THRESHOLD * 100:.2f}% "
        f"-> SIGNATURE ABSENT\n\n"
    )


    for number, result in enumerate(
        results,
        start=1
    ):

        file.write(
            f"{number}. {result['name']}\n"
        )

        file.write(
            f"   Label Aktual : "
            f"{result['actual']}\n"
        )

        file.write(
            f"   Threshold    : "
            f"{result['selected_method']}\n"
        )

        file.write(
            f"   Foreground   : "
            f"{result['foreground_pixels']} px\n"
        )

        file.write(
            f"   Ratio        : "
            f"{result['foreground_percentage']:.2f}%\n"
        )

        file.write(
            f"   Prediksi     : "
            f"{result['prediction']}\n"
        )

        file.write(
            f"   Status       : "
            f"{result['status']}\n"
        )

        file.write("\n")


# ============================================================
# SELESAI
# ============================================================

print()
print("=" * 80)
print("                    PROGRAM SELESAI")
print("=" * 80)
print()

print(
    "Present :",
    len(present_images),
    "citra"
)

print(
    "Absent  :",
    len(absent_images),
    "citra"
)

print(
    "Total   :",
    len(results),
    "citra"
)

print()

print(
    "Hasil comparison:"
)

print(
    COMPARISON_FOLDER
)

print()

print(
    "Hasil analisis:"
)

print(
    analysis_file
)

print()

print(
    "Hasil klasifikasi:"
)

print(
    classification_file
)

print()