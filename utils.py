import zipfile
from collections import OrderedDict
import os
import sys
import logging
import traceback
import numpy as np


def create_dummy_prediction(args, eval_ds, predictions):
    prediction_csv_path = f'{args.output_folder}/prediction.csv'
    # all image names in the database and queries
    database_name = np.asarray([os.path.splitext(os.path.basename(path))[0] for path in eval_ds.database_paths])
    queries_name = np.asarray([os.path.splitext(os.path.basename(path))[0] for path in eval_ds.queries_paths])
    predictions_str = np.asarray([[database_name[pred] for pred in preds[:20]] for preds in predictions])
    queries_str = np.asarray(queries_name.reshape(-1, 1))

    logging.debug(f"==> We create a new prediction csv file at : {prediction_csv_path}")
    # save the dummy predictions
    np.savetxt(prediction_csv_path, np.concatenate([queries_str, predictions_str], axis=1), fmt='%s')
    zip_file_path = prediction_csv_path.replace('.csv', '.zip')

    logging.debug(
        f"==> Package CSV files {os.path.basename(prediction_csv_path)} into zip format: "
        f"{os.path.basename(zip_file_path)}")

    with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(prediction_csv_path, arcname=os.path.basename(prediction_csv_path))
    logging.info('Prediction completed!')
    logging.info(
        "Please upload the .zip file to the official website: https://codalab.lisn.upsaclay.fr/competitions/865 "
        "to obtain the experimental results")


def clean_state_dict(state_dict):
    # 'clean' checkpoint by removing .module prefix from state dict if it exists from parallel training
    cleaned_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        cleaned_state_dict[name] = v
    return cleaned_state_dict


def setup_logging(output_folder: str, exist_ok: bool = False, console: str = "debug",
                  info_filename: str = "info.log", debug_filename: str = "debug.log"):
    if not exist_ok and os.path.exists(output_folder):
        raise FileExistsError(f"{output_folder} already exists!")
    os.makedirs(output_folder, exist_ok=True)
    base_formatter = logging.Formatter('%(asctime)s   %(message)s', "%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger('')
    logger.setLevel(logging.DEBUG)

    if info_filename is not None:
        info_file_handler = logging.FileHandler(f'{output_folder}/{info_filename}')
        info_file_handler.setLevel(logging.INFO)
        info_file_handler.setFormatter(base_formatter)
        logger.addHandler(info_file_handler)

    if debug_filename is not None:
        debug_file_handler = logging.FileHandler(f'{output_folder}/{debug_filename}')
        debug_file_handler.setLevel(logging.DEBUG)
        debug_file_handler.setFormatter(base_formatter)
        logger.addHandler(debug_file_handler)

    if console is not None:
        console_handler = logging.StreamHandler()
        if console == "debug":
            console_handler.setLevel(logging.DEBUG)
        if console == "info":
            console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(base_formatter)
        logger.addHandler(console_handler)

    def my_handler(type_, value, tb):
        logger.info("\n" + "".join(traceback.format_exception(type, value, tb)))
        logging.info("Experiment finished (with some errors)")

    sys.excepthook = my_handler
