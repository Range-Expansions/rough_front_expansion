import numpy as np
import pandas as pd
import skimage as ski
import skimage.morphology
import skimage.measure


def get_wall_df(sim, ii, jj, expansion_size=2, single_sector=False, collision_offset=5):
    """Returns a list of domain wall positions. Could be improved, terrible at dealing with when walls collide."""
    frozen_field = np.asarray(sim.lattice)

    frozen_pops = np.zeros((frozen_field.shape[0], frozen_field.shape[1], sim.num_strains), dtype=np.bool)
    for i in range(sim.num_strains):
        frozen_pops[:, :, i] = (frozen_field == i)

    expanded_pops = np.zeros_like(frozen_pops)

    expander = ski.morphology.disk(expansion_size)

    for i in range(sim.num_strains):
        cur_slice = frozen_pops[:, :, i]
        expanded_pops[:, :, i] = ski.morphology.binary_dilation(cur_slice, selem=expander)

    walls = expanded_pops[:, :, ii] * expanded_pops[:, :, jj]

    labeled_walls = ski.measure.label(walls, connectivity=2)

    if single_sector:
        if labeled_walls.max() == 1:  # Walls collided...need to split the data
            print 'Domain walls must have collided...fixing the labels...'
            r, c = np.where(walls)
            max_c = np.max(c)
            new_label_image = walls[:, max_c - collision_offset]
            labeled_walls = ski.measure.label(walls, connectivity=2)

    df_list = []

    for cur_label in range(1, np.max(labeled_walls) + 1):
        r, c = np.where(labeled_walls == cur_label)

        df = pd.DataFrame(data={'i': ii, 'j': jj, 'label_num': cur_label, 'r': r, 'c': c})

        # Group the df so that there is only one y for each x

        df_list.append(df)
    if len(df_list) == 0:
        return None
    return pd.concat(df_list)