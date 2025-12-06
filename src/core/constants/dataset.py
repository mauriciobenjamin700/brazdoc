from os.path import abspath, dirname, join

ROOT = dirname(dirname(dirname(dirname(__file__))))

DATASET_PATH = abspath(join(ROOT, "data"))

CPF_IMAGE_FOLDERS = [
    join(DATASET_PATH, "CPF_Frente"),
    join(DATASET_PATH, "CPF_Verso"),
]
RG_IMAGE_FOLDERS = [
    join(DATASET_PATH, "RG_Aberto"),
    join(DATASET_PATH, "RG_Frente"),
    join(DATASET_PATH, "RG_Verso"),
]
CNH_IMAGE_FOLDERS = [
    join(DATASET_PATH, "CNH_Aberto"),
    join(DATASET_PATH, "CNH_Frente"),
    join(DATASET_PATH, "CNH_Verso"),
]

IMAGE_EXTENSION = ".jpg"
