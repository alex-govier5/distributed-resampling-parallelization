from src.bump.bump import Bump
from src.utils.dataframe import select_df


def collect_bumps(dataset, phi, threshold=0.8):
    bumps = []

    n_samples = dataset.count()
    i_start = 0
    i_end = 0

    for i in range(n_samples):
        if phi[i] >= threshold:
            if i == 0 or phi[i - 1] < threshold:
                i_start = i
            if i + 1 == n_samples or phi[i + 1] < threshold:
                i_end = i
                bumps.append(Bump("rare", select_df(dataset, i_start, i_end)))
        else:
            if i == 0 or phi[i - 1] >= threshold:
                i_start = i
            if i + 1 == n_samples or phi[i + 1] >= threshold:
                i_end = i
                bumps.append(Bump("normal", select_df(dataset, i_start, i_end)))

    return bumps

# NEW
# def collect_bumps(dataset, phi, threshold=0.8):
#     bumps = []
#     n_samples = dataset.count()
#     i_start = 0
#     current_type = "normal" if phi[0] < threshold else "rare"

#     for i in range(1, n_samples):
#         is_rare = phi[i] >= threshold
#         if (is_rare and current_type == "normal") or (not is_rare and current_type == "rare"):
#             bumps.append(Bump(current_type, select_df(dataset, i_start, i - 1)))
#             i_start = i
#             current_type = "rare" if is_rare else "normal"

#     bumps.append(Bump(current_type, select_df(dataset, i_start, n_samples - 1)))

#     return bumps

def get_rare_bumps(bumps):
    return [bump for bump in bumps if bump.type == "rare"]


def get_normal_bumps(bumps):
    return [bump for bump in bumps if bump.type == "normal"]
