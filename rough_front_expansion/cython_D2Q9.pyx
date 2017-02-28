#cython: profile=False
#cython: boundscheck=False
#cython: initializedcheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: cdivision=True

import numpy as np
cimport numpy as np
import skimage as ski
import skimage.morphology
import skimage.measure
import matplotlib.pyplot as plt

from cython_gsl cimport *
from cpython cimport bool

import pandas as pd


# The lattice we are using; right now, square, but should probably upgrade to a 9-point lattice
cdef int[:] cx = np.array([1, 0, -1, 0, 1, -1, -1,  1], dtype=np.int32)
cdef int[:] cy = np.array([0, 1, 0, -1, 1,  1, -1, -1], dtype=np.int32)
dist = np.array([1., 1., 1., 1., 1./np.sqrt(2), 1./np.sqrt(2), 1./np.sqrt(2), 1./np.sqrt(2)], dtype=np.double)
cdef double[:] weights = dist/np.sum(dist)
cdef int NUM_LATTICE_NEIGHBORS = 8

cdef class Rough_Front(object):

    cdef public:
        int nx
        int ny
        int num_strains
        int[:] ic
        double[:] v
        int[:, :] lattice
        list strain_positions_x
        list strain_positions_y
        list strain_labels
        int[:] N
        double[:] weights
        int[:] strain_array
        int max_iterations
        int iterations_run

        bool debug

    cdef public unsigned long int seed
    cdef gsl_rng *random_generator

    def __cinit__(self, unsigned long int seed = 0, **kwargs):
        self.seed = seed
        cdef gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937)
        gsl_rng_set(r, self.seed)
        self.random_generator = r

    def __dealloc__(self):
        gsl_rng_free(self.random_generator)

    def __init__(self, nx=100, ny=100, num_strains=2, ic = None, v=None, debug=False, **kwargs):
        self.nx = nx # Dimension of the lattice in the x (row) direction
        self.ny = ny # Dimension of the lattice in the y (column) direction

        self.debug = debug

        self.num_strains = num_strains # Number of different strains in the simulation
        if v is None:
            self.v = np.ones(num_strains, dtype=np.double) # Growth velocities of each strain
        else:
            self.v = v

        if ic is None:
            self.ic =  np.random.randint(0, num_strains, size=nx) # The initial condition of the lattice (along a line)
        else:
            self.ic = ic

        self.lattice = -1*np.ones((self.nx, self.ny), dtype=np.int32)
        self.lattice[:, 0] = self.ic

        # Get the original location of the interface
        background = (np.asarray(self.lattice) == -1)
        # Need to use 8-point dilation now
        selem = ski.morphology.square(3)
        grown = ski.morphology.binary_dilation(background, selem=selem)  # 8-point dilation
        interface = grown != background
        interface = np.where(interface)

        interface_loc = np.stack(interface, axis=1).astype(np.int32)

        self.strain_positions_x = [] # Locations of each type of strain
        self.strain_positions_y = []
        self.strain_labels = [] # A unique, position-based label for each strain
        for _ in range(num_strains):
            self.strain_positions_x.append([])
            self.strain_positions_y.append([])
            self.strain_labels.append([])

        self.N = np.zeros(num_strains, dtype=np.int32) # Number of each type of strain at the interface
        for cur_loc in interface_loc:
            cur_strain = self.lattice[cur_loc[0], cur_loc[1]]
            self.N[cur_strain] += 1

            self.strain_positions_x[cur_strain].append(cur_loc[0])
            self.strain_positions_y[cur_strain].append(cur_loc[1])

            self.strain_labels[cur_strain].append(cur_loc[1] * nx + cur_loc[0])

        self.weights = np.zeros(num_strains, dtype=np.double) # The weight to draw each type of strain
        self.strain_array = np.arange(num_strains, dtype=np.int32) # The name of every strain; i.e. 0->num_strains - 1

        # Get the maximum possible number of iterations...don't want to run longer than that!
        self.max_iterations = np.sum(np.asarray(self.lattice) == -1)
        self.iterations_run = 0


    cdef void get_nearby_empty_locations(self, int cur_x, int cur_y,
                                    int *x_choices, int *y_choices,
                                    int *num_choices) nogil:
        """
        x_choices, y_choices: pointers to an array with the size of num_neighbors
        num_choices: the number of possible choices
        """

        cdef int n
        cdef int cur_cx, cur_cy, streamed_x, streamed_y
        cdef int neighboring_strain

        cdef int temp_num_choices = 0

        for n in range(NUM_LATTICE_NEIGHBORS):
            cur_cx = cx[n]
            cur_cy = cy[n]

            streamed_x = cur_x + cur_cx
            streamed_y = cur_y + cur_cy

            # Periodic BC's in x, not y
            if streamed_x == self.nx:
                streamed_x = 0

            if streamed_x == -1:
                streamed_x = self.nx - 1

            if not (streamed_y == self.ny or streamed_y == -1):

                neighboring_strain = self.lattice[streamed_x, streamed_y]

                if neighboring_strain == -1:
                    x_choices[temp_num_choices] = streamed_x
                    y_choices[temp_num_choices] = streamed_y
                    temp_num_choices += 1

        num_choices[0] = temp_num_choices

    cdef void get_nearby_locations(self, int cur_x, int cur_y,
                                   int *x_choices, int *y_choices,
                                   int *num_choices) nogil:
        """
        x_choices, y_choices: pointers to an array with the size of num_neighbors
        num_choices: the number of possible choices.

        Gets all nearby locations.
        """

        cdef int n
        cdef int cur_cx, cur_cy, streamed_x, streamed_y
        cdef int neighboring_strain

        cdef int temp_num_choices = 0

        for n in range(NUM_LATTICE_NEIGHBORS):
            cur_cx = cx[n]
            cur_cy = cy[n]

            streamed_x = cur_x + cur_cx
            streamed_y = cur_y + cur_cy

            # Periodic BC's in x, not y
            if streamed_x == self.nx:
                streamed_x = 0

            if streamed_x == -1:
                streamed_x = self.nx - 1

            if not (streamed_y == self.ny or streamed_y == -1):

                x_choices[temp_num_choices] = streamed_x
                y_choices[temp_num_choices] = streamed_y
                temp_num_choices += 1

        num_choices[0] = temp_num_choices

    cdef unsigned int weighted_choice(self, double[:] normalized_weights) nogil:
        cdef double rand_num = gsl_rng_uniform(self.random_generator)

        cdef double cur_sum = 0
        cdef unsigned int index = 0

        cdef double normalized_sum = 0

        for index in range(normalized_weights.shape[0]):
            cur_sum += normalized_weights[index]

            if cur_sum > rand_num:
                return index

    def run(self, int num_iterations):

        cdef double[:] normalized_weights = self.weights.copy()
        cdef int iteration
        cdef double sum_of_weights
        cdef int strain
        cdef double cur_weight
        cdef int chosen_type
        cdef int random_index
        cdef int[:] cur_loc
        cdef random_choice
        cdef int[:] new_loc
        cdef int num_free, new_label
        cdef int l
        cdef int[:] loc
        cdef int neighbor_id
        cdef int two_d_index
        cdef int index_to_remove

        cdef int choice_index

        cdef int[8] x_choices = np.zeros(8, dtype=np.int32) # I don't know how to declare a final constant in cython...lol
        cdef int[8] y_choices = np.zeros(8, dtype=np.int32)

        cdef int[8] neighbor_x_choices = np.zeros(8, dtype=np.int32)
        cdef int[8] neighbor_y_choices = np.zeros(8, dtype=np.int32)

        cdef int num_choices = 0
        cdef int num_neighbors = 0

        cdef int new_loc_x, new_loc_y, cur_loc_x, cur_loc_y

        if self.iterations_run < self.max_iterations:
            for iteration in range(num_iterations):
                # Choose what type of cell to divide

                if self.debug:
                    print 'POINT A'

                sum_of_weights = 0
                for strain in range(self.num_strains):
                    cur_weight = self.N[strain] * self.v[strain]

                    self.weights[strain] = cur_weight
                    sum_of_weights += cur_weight
                for strain  in range(self.num_strains):
                    normalized_weights[strain] = self.weights[strain] / sum_of_weights

                choice_index = self.weighted_choice(normalized_weights)
                chosen_type = self.strain_array[choice_index]

                # Now that we have the type to choose, choose that type at random
                random_index = gsl_rng_uniform_int(self.random_generator, self.N[chosen_type])

                cur_loc_x = self.strain_positions_x[chosen_type][random_index]
                cur_loc_y = self.strain_positions_y[chosen_type][random_index]

                # Check where you can reproduce
                self.get_nearby_empty_locations(cur_loc_x, cur_loc_y,
                                                &x_choices[0], &y_choices[0], &num_choices)

                random_choice = gsl_rng_uniform_int(self.random_generator, num_choices)
                new_loc_x = x_choices[random_choice]
                new_loc_y = y_choices[random_choice]

                # Now update the lattice
                self.lattice[new_loc_x, new_loc_y] = chosen_type

                # Uh oh... you have to check if *you* are on the interface now!
                self.get_nearby_empty_locations(new_loc_x, new_loc_y,
                                                &x_choices[0], &y_choices[0], &num_choices)

                if num_choices != 0: # Need to update the front
                    new_label = new_loc_y * self.nx + new_loc_x

                    self.strain_labels[chosen_type].append(new_label)
                    self.strain_positions_x[chosen_type].append(new_loc_x)
                    self.strain_positions_y[chosen_type].append(new_loc_y)

                    self.N[chosen_type] += 1

                if self.debug:
                    print 'POINT B'

                # Now update who is on the edge of the interface.
                # We have to check who is on the four squares around the interface
                self.get_nearby_locations(new_loc_x, new_loc_y,
                                          &x_choices[0], &y_choices[0], &num_neighbors)

                for l in range(num_neighbors):
                    neighbor_loc_x = x_choices[l]
                    neighbor_loc_y = y_choices[l]

                    neighbor_id = self.lattice[neighbor_loc_x, neighbor_loc_y]
                    if neighbor_id != -1:
                        two_d_index = neighbor_loc_y * self.nx + neighbor_loc_x
                        if two_d_index in self.strain_labels[neighbor_id]:
                            self.get_nearby_empty_locations(neighbor_loc_x, neighbor_loc_y,
                                                            &neighbor_x_choices[0], &neighbor_y_choices[0], &num_choices)

                            if num_choices == 0:
                                # Remove from interface
                                index_to_remove = self.strain_labels[neighbor_id].index(two_d_index)
                                del self.strain_labels[neighbor_id][index_to_remove]
                                del self.strain_positions_x[neighbor_id][index_to_remove]
                                del self.strain_positions_y[neighbor_id][index_to_remove]

                                self.N[neighbor_id] -= 1

                self.iterations_run += 1
                if self.iterations_run == self.max_iterations:
                    print 'Ran for the maximum number of iterations! Done.'
                    break
        else:
            print 'I already ran for the maximum number of iterations! Done.'

    def get_wall_df(self, ii, jj, expansion_size = 3, single_sector=False, collision_offset=5):

        frozen_field = np.asarray(self.lattice)

        frozen_pops = np.zeros((frozen_field.shape[0], frozen_field.shape[1], self.num_strains), dtype=np.bool)
        for i in range(self.num_strains):
            frozen_pops[:, :, i] = (frozen_field == i)

        expanded_pops = np.zeros_like(frozen_pops)

        expander = ski.morphology.disk(expansion_size)

        for i in range(self.num_strains):
            cur_slice = frozen_pops[:, :, i]
            expanded_pops[:, :, i] = ski.morphology.binary_dilation(cur_slice, selem=expander)

        walls = expanded_pops[:, :, ii] * expanded_pops[:, :, jj]

        labeled_walls = ski.measure.label(walls, connectivity=2)

        if single_sector:
            if labeled_walls.max() == 1: # Walls collided...need to split the data
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