from pdf2image import convert_from_path
import os
import cv2
import matplotlib.pyplot as plt
from skimage import measure, morphology
from skimage.measure import regionprops


def konvertiranje_od_PDF_vo_sliki(pateka_na_PDF, denstinacisi_folder):
    images = convert_from_path(pateka_na_PDF)
    os.makedirs(denstinacisi_folder, exist_ok=True)

    for i in range(len(images)):
        jpg_tmp_filename = os.path.join(denstinacisi_folder, f"page_{i + 1}.jpg")
        images[i].save(jpg_tmp_filename, 'JPEG')


def cistenje_direktorium(direktorium_za_cistenje):
    for datoteka in os.listdir(direktorium_za_cistenje):
        pateka_do_datoteka = os.path.join(direktorium_za_cistenje, datoteka)
        if os.path.isfile(pateka_do_datoteka):
            os.remove(pateka_do_datoteka)


parametar_1 = 84
parametar_2 = 250
parametar_3 = 100
parametar_4 = 18

cistenje_direktorium('inputs')

pateka_na_PDF = "pdftest1.pdf"
denstinacisi_folder = "inputs"

konvertiranje_od_PDF_vo_sliki(pateka_na_PDF, denstinacisi_folder)

br_na_strani = len(convert_from_path(pateka_na_PDF))

cistenje_direktorium('outputs')

for i in range(1, br_na_strani + 1):
    img = cv2.imread(f'./inputs/page_{i}.jpg', 0)
    img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]

    blobs = img == 255
    blobs_labels = measure.label(blobs, background=1)

    vkupna_plostina = 0
    br = 0

    for region in regionprops(blobs_labels):
        if region.area > 10:
            vkupna_plostina += region.area
            br += 1

    if br > 0:
        prosek_od_plostini = vkupna_plostina / br
    else:
        prosek_od_plostini = 0

    mali_vrednosti_a4_Format = ((prosek_od_plostini / parametar_1) * parametar_2) + parametar_3

    probna_verzija = morphology.remove_small_objects(blobs_labels, mali_vrednosti_a4_Format)

    plt.imsave(f'probna_verzija{i}.png', probna_verzija)
    img = cv2.imread(f'probna_verzija{i}.png', 0)
    os.remove(f'probna_verzija{i}.png')

    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    cv2.imwrite(f'./outputs/output_{i}.png', img)
