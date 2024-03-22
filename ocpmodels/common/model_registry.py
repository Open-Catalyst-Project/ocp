import logging
import os
import shutil
from typing import Optional

import urllib3

model_registry = {
    "CGCNN 200k": "https://dl.fbaipublicfiles.com/opencatalystproject/models/2020_11/s2ef/cgcnn_200k.pt",
    "CGCNN 2M": "https://dl.fbaipublicfiles.com/opencatalystproject/models/2020_11/s2ef/cgcnn_2M.pt",
    "CGCNN 20M": "https://dl.fbaipublicfiles.com/opencatalystproject/models/2020_11/s2ef/cgcnn_20M.pt",
    "CGCNN All": "https://dl.fbaipublicfiles.com/opencatalystproject/models/2020_11/s2ef/cgcnn_all.pt",
    "DimeNet 200k": "https://dl.fbaipublicfiles.com/opencatalystproject/models/2020_11/s2ef/dimenet_200k.pt",
    "DimeNet 2M": "https://dl.fbaipublicfiles.com/opencatalystproject/models/2020_11/s2ef/dimenet_2M.pt",
    "SchNet 200k": "https://dl.fbaipublicfiles.com/opencatalystproject/models/2020_11/s2ef/schnet_200k.pt",
    "SchNet 2M": "https://dl.fbaipublicfiles.com/opencatalystproject/models/2020_11/s2ef/schnet_2M.pt",
    "SchNet 20M": "https://dl.fbaipublicfiles.com/opencatalystproject/models/2020_11/s2ef/schnet_20M.pt",
    "SchNet All": "https://dl.fbaipublicfiles.com/opencatalystproject/models/2020_11/s2ef/schnet_all_large.pt",
    "DimeNet++ 200k": "https://dl.fbaipublicfiles.com/opencatalystproject/models/2021_02/s2ef/dimenetpp_200k.pt",
    "DimeNet++ 2M": "https://dl.fbaipublicfiles.com/opencatalystproject/models/2021_02/s2ef/dimenetpp_2M.pt",
    "DimeNet++ 20M": "https://dl.fbaipublicfiles.com/opencatalystproject/models/2021_02/s2ef/dimenetpp_20M.pt",
    "DimeNet++ All": "https://dl.fbaipublicfiles.com/opencatalystproject/models/2021_02/s2ef/dimenetpp_all.pt",
    "SpinConv 2M": "https://dl.fbaipublicfiles.com/opencatalystproject/models/2021_12/s2ef/spinconv_force_centric_2M.pt",
    "SpinConv All": "https://dl.fbaipublicfiles.com/opencatalystproject/models/2021_08/s2ef/spinconv_force_centric_all.pt",
    "GemNet-dT 2M": "https://dl.fbaipublicfiles.com/opencatalystproject/models/2021_12/s2ef/gemnet_t_direct_h512_2M.pt",
    "GemNet-dT All": "https://dl.fbaipublicfiles.com/opencatalystproject/models/2021_08/s2ef/gemnet_t_direct_h512_all.pt",
    "PaiNN All": "https://github.com/Open-Catalyst-Project/ocp/blob/main/configs/s2ef/all/painn/painn_nb6_scaling_factors.pt",
    "GemNet-OC 2M": "https://github.com/Open-Catalyst-Project/ocp/blob/481f3a5a92dc787384ddae9fe3f50f5d932712fd/configs/s2ef/all/gemnet/scaling_factors/gemnet-oc.pt",
    "GemNet-OC All": "https://github.com/Open-Catalyst-Project/ocp/blob/481f3a5a92dc787384ddae9fe3f50f5d932712fd/configs/s2ef/all/gemnet/scaling_factors/gemnet-oc.pt",
    "GemNet-OC All+MD": "https://github.com/Open-Catalyst-Project/ocp/blob/481f3a5a92dc787384ddae9fe3f50f5d932712fd/configs/s2ef/all/gemnet/scaling_factors/gemnet-oc.pt",
    "GemNet-OC-Large All+MD": "https://github.com/Open-Catalyst-Project/ocp/blob/481f3a5a92dc787384ddae9fe3f50f5d932712fd/configs/s2ef/all/gemnet/scaling_factors/gemnet-oc-large.pt",
    "SCN 2M": "https://dl.fbaipublicfiles.com/opencatalystproject/models/2023_03/s2ef/scn_t1_b1_s2ef_2M.pt",
    "SCN-t4-b2 2M": "https://dl.fbaipublicfiles.com/opencatalystproject/models/2023_03/s2ef/scn_t4_b2_s2ef_2M.pt",
    "SCN All+MD": "https://dl.fbaipublicfiles.com/opencatalystproject/models/2023_03/s2ef/scn_all_md_s2ef.pt",
    "eSCN-L4-M2-Lay12 2M": "https://dl.fbaipublicfiles.com/opencatalystproject/models/2023_03/s2ef/escn_l4_m2_lay12_2M_s2ef.pt",
    "eSCN-L6-M2-Lay12 2M": "https://dl.fbaipublicfiles.com/opencatalystproject/models/2023_03/s2ef/escn_l6_m2_lay12_2M_s2ef.pt",
    "eSCN-L6-M2-Lay12All+MD": "https://dl.fbaipublicfiles.com/opencatalystproject/models/2023_03/s2ef/escn_l6_m2_lay12_all_md_s2ef.pt",
    "eSCN-L6-M3-Lay20All+MD": "https://dl.fbaipublicfiles.com/opencatalystproject/models/2023_03/s2ef/escn_l6_m3_lay20_all_md_s2ef.pt",
    "EquiformerV2 (83M) 2M": "https://dl.fbaipublicfiles.com/opencatalystproject/models/2023_06/oc20/s2ef/eq2_83M_2M.pt",
    "EquiformerV2 (31M) All+MD": "https://dl.fbaipublicfiles.com/opencatalystproject/models/2023_06/oc20/s2ef/eq2_31M_ec4_allmd.pt",
    "EquiformerV2 (153M) All+MD": "https://dl.fbaipublicfiles.com/opencatalystproject/models/2023_06/oc20/s2ef/eq2_153M_ec4_allmd.pt",
    "SchNet All forceonly": "https://dl.fbaipublicfiles.com/opencatalystproject/models/2020_11/s2ef/schnet_all_forceonly.pt",
    "DimeNet++ All forceonly": "https://dl.fbaipublicfiles.com/opencatalystproject/models/2020_11/s2ef/dimenetpp_all_forceonly.pt",
    "DimeNet++-Large All": "https://dl.fbaipublicfiles.com/opencatalystproject/models/2021_02/s2ef/dimenetpp_large_all_forceonly.pt",
    "DimeNet++ 20M+Rattled": "https://dl.fbaipublicfiles.com/opencatalystproject/models/2021_02/s2ef/dimenetpp_20M_rattled_forceonly.pt",
    "DimeNet++ 20M+MD": "https://dl.fbaipublicfiles.com/opencatalystproject/models/2021_02/s2ef/dimenetpp_20M_md_forceonly.pt",
    "CGCNN 10k is2re": "https://dl.fbaipublicfiles.com/opencatalystproject/models/2021_02/is2re/cgcnn_10k.pt",
    "CGCNN 100k is2re": "https://dl.fbaipublicfiles.com/opencatalystproject/models/2021_02/is2re/cgcnn_100k.pt",
    "CGCNN All is2re": "https://dl.fbaipublicfiles.com/opencatalystproject/models/2021_02/is2re/cgcnn_all.pt",
    "DimeNet 10k is2re": "https://dl.fbaipublicfiles.com/opencatalystproject/models/2020_11/is2re/dimenet_10k.pt",
    "DimeNet 100k is2re": "https://dl.fbaipublicfiles.com/opencatalystproject/models/2020_11/is2re/dimenet_100k.pt",
    "DimeNet All is2re": "https://dl.fbaipublicfiles.com/opencatalystproject/models/2020_11/is2re/dimenet_all.pt",
    "SchNet 10k is2re": "https://dl.fbaipublicfiles.com/opencatalystproject/models/2021_02/is2re/schnet_10k.pt",
    "SchNet 100k is2re": "https://dl.fbaipublicfiles.com/opencatalystproject/models/2021_02/is2re/schnet_100k.pt",
    "SchNet All is2re": "https://dl.fbaipublicfiles.com/opencatalystproject/models/2021_02/is2re/schnet_all.pt",
    "DimeNet++ 10k is2re": "https://dl.fbaipublicfiles.com/opencatalystproject/models/2021_02/is2re/dimenetpp_10k.pt",
    "DimeNet++ 100k is2re": "https://dl.fbaipublicfiles.com/opencatalystproject/models/2021_02/is2re/dimenetpp_100k.pt",
    "DimeNet++ All is2re": "https://dl.fbaipublicfiles.com/opencatalystproject/models/2021_02/is2re/dimenetpp_all.pt",
    "PaiNNAll": "https://github.com/Open-Catalyst-Project/ocp/blob/main/configs/s2ef/all/painn/painn_nb6_scaling_factors.pt",
    "GemNet-dTOC22": "https://dl.fbaipublicfiles.com/opencatalystproject/models/2022_09/oc22/s2ef/gndt_oc22_all_s2ef.pt",
    "GemNet-OCOC22": "https://dl.fbaipublicfiles.com/opencatalystproject/models/2022_09/oc22/s2ef/gnoc_oc22_all_s2ef.pt",
    "GemNet-OCOC20+OC22": "https://dl.fbaipublicfiles.com/opencatalystproject/models/2022_09/oc22/s2ef/gnoc_oc22_oc20_all_s2ef.pt",
    "GemNet-OC enforce_max_neighbors_strictly=False": "https://dl.fbaipublicfiles.com/opencatalystproject/models/2023_05/oc22/s2ef/gnoc_oc22_oc20_all_s2ef.pt",
    "GemNet-OCOC20->OC22": "https://dl.fbaipublicfiles.com/opencatalystproject/models/2022_09/oc22/s2ef/gnoc_finetune_all_s2ef.pt",
    "EquiformerV2 lambda_E$=4, lambda_F$=100 OC22": "https://dl.fbaipublicfiles.com/opencatalystproject/models/2023_10/oc22/s2ef/eq2_121M_e4_f100_oc22_s2ef.pt",
    "SchNet": "https://dl.fbaipublicfiles.com/dac/checkpoints_20231018/Schnet.pt",
    "DimeNet++": "https://dl.fbaipublicfiles.com/dac/checkpoints_20231018/DimenetPP.pt",
    "PaiNN": "https://dl.fbaipublicfiles.com/dac/checkpoints_20231018/PaiNN.pt",
    "GemNet-OC": "https://dl.fbaipublicfiles.com/dac/checkpoints_20231018/Gemnet-OC.pt",
    "eSCN": "https://dl.fbaipublicfiles.com/dac/checkpoints_20231018/eSCN.pt",
    "EquiformerV2": "https://dl.fbaipublicfiles.com/dac/checkpoints_20231116/eqv2_31M.pt",
    "EquiformerV2 (Large)": "https://dl.fbaipublicfiles.com/dac/checkpoints_20231018/Equiformer_V2_Large.pt",
    "Gemnet-OC (Direct)": "https://dl.fbaipublicfiles.com/dac/checkpoints_20231018/Gemnet-OC_Direct.pt",
    "eSCN (Direct)": "https://dl.fbaipublicfiles.com/dac/checkpoints_20231018/eSCN_Direct.pt",
    "EquiformerV2 (Direct)": "https://dl.fbaipublicfiles.com/dac/checkpoints_20231018/Equiformer_V2_Direct.pt",
}


def model_name_to_local_file(
    model_name: str, local_cache: str
) -> Optional[str]:
    logging.info(f"Checking local cache: {local_cache} for model {model_name}")
    if model_name not in model_registry:
        logging.error(f"Not a valid model name '{model_name}'")
        return None
    if not os.path.exists(local_cache):
        os.makedirs(local_cache, exist_ok=True)
    if not os.path.exists(local_cache):
        logging.error(f"Failed to create local cache folder '{local_cache}'")
        return None
    model_url = model_registry[model_name]
    local_path = os.path.join(local_cache, os.path.basename(model_url))

    # download the file
    if not os.path.isfile(local_path):
        local_path_tmp = (
            local_path + ".tmp"
        )  # download to a tmp file in case we fail
        http = urllib3.PoolManager()
        with open(local_path_tmp, "wb") as out:
            r = http.request("GET", model_url, preload_content=False)
            shutil.copyfileobj(r, out)
        shutil.move(local_path_tmp, local_path)
    return local_path
