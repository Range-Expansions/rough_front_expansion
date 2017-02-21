import numpy as np
import skimage as ski
import skimage.morphology

# The lattice we are using; right now, square, but should probably upgrade to a 9-point lattice
cx = np.array([1, 0, -1, 0], dtype=np.int32)
cy = np.array([0, 1, 0, -1], dtype=np.int32)
num_neighbors = 4

class Rough_Front(object):

    def __init__(self, nx, ny, num_strains=2, ic = None, v=None):
        self.nx = nx
        self.ny = ny

        self.num_strains = num_strains
        if self.v is None:
            self.v = np.ones(num_strains, dtype=np.double)
        else:
            self.v = v

        if ic is None:
            self.ic =  np.random.randint(0, num_strains, size=nx)
        else:
            self.ic = ic

        self.lattice = -1*np.ones((self.ny, self.ny), dtype=np.int)
        self.lattice[:, 0] = self.ic

        # Get the original location of the interface
        background = (self.lattice == -1)
        grown = ski.morphology.binary_dilation(background)  # Cross dilation
        interface = grown != background
        interface = np.where(interface)

        interface_loc = np.stack(interface, axis=1)

        self.strain_positions = [] # Locations of each type of strain
        self.strain_labels = [] # A unique, position-based label for each strain
        for _ in range(num_strains):
            self.strain_positions.append([])
            self.strain_labels.append([])

        self.N = np.zeros(num_strains) # Number of each type of strain at the interface
        for cur_loc in interface_loc:
            cur_strain = self.lattice[cur_loc[0], cur_loc[1]]
            self.N[cur_strain] += 1

            self.strain_positions[cur_strain].append(cur_loc)

            self.strain_labels[cur_strain].append(cur_loc[1] * nx + cur_loc[0])

        self.weights = np.zeros(num_strains, dtype=np.double) # The weight to draw each type of strain
        self.strain_array = np.arange(num_strains, dtype=np.int) # The name of every strain; i.e. 0->num_strains - 1

    def get_nearby_empty_locations(self, cur_loc):

        cur_x = cur_loc[0]
        cur_y = cur_loc[1]

        num_choices = 0
        choices_to_occupy = []

        for n in range(num_neighbors):
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
                    num_choices += 1
                    choices_to_occupy.append(np.array([streamed_x, streamed_y]))

        return num_choices, choices_to_occupy

    def get_nearby_locations(self, cur_loc):

        cur_x = cur_loc[0]
        cur_y = cur_loc[1]

        choices_to_occupy = []

        for n in range(num_neighbors):
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
                choices_to_occupy.append(np.array([streamed_x, streamed_y]))

        return choices_to_occupy


    def run(self, num_iterations):

        for iteration in range(num_iterations):
            # Choose what type of cell to divide

            sum_of_weights = 0
            for strain in range(self.num_strains):
                cur_weight = self.N[strain] * self.v[strain]

                self.weights[strain] = cur_weight
                sum_of_weights += cur_weight
            normalized_weights = self.weights / sum_of_weights

            chosen_type = np.random.choice(self.strain_array, p=normalized_weights)

            # Now that we have the type to choose, choose that type at random
            random_index = np.random.randint(0, self.N[chosen_type])

            cur_loc = self.strain_positions[chosen_type][random_index]

            # Check where you can reproduce
            num_choices, choices_to_occupy = self.get_nearby_empty_locations(cur_loc)

            if num_choices == 0:
                print 'Something bad has happened...'
                print cur_loc
                print cur_loc[1] * self.nx + cur_loc[0]

            random_choice = np.random.randint(0, num_choices)
            new_loc = choices_to_occupy[random_choice]

            # Now update the lattice
            self.lattice[new_loc[0], new_loc[1]] = chosen_type

            # Uh oh... you have to check if *you* are on the interface now!
            num_free, _ = self.get_nearby_empty_locations(new_loc)
            if num_free != 0:
                new_label = new_loc[1] * self.nx + new_loc[0]

                self.strain_labels[chosen_type].append(new_label)
                self.strain_positions[chosen_type].append(new_loc)
                self.N[chosen_type] += 1

            # Now update who is on the edge of the interface.
            # We have to check who is on the four squares around the interface
            new_neighbors_loc = self.get_nearby_locations(new_loc)

            for l in new_neighbors_loc:
                neighbor_id = self.lattice[l[0], l[1]]
                if neighbor_id != -1:
                    two_d_index = l[1] * self.nx + l[0]
                    if two_d_index in self.strain_labels[neighbor_id]:
                        num_free, _ = self.get_nearby_empty_locations(l)
                        if num_free == 0:
                            # Remove from interface
                            index_to_remove = self.strain_labels[neighbor_id].index(two_d_index)
                            del self.strain_labels[neighbor_id][index_to_remove]
                            del self.strain_positions[neighbor_id][index_to_remove]
                            self.N[neighbor_id] -= 1