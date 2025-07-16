"""
Package for simulating the metabolic ecology of microbes. This package includes the classes:
    - Microbes: holds properties of microbial metabolism and methods to simulate the metabolism.
    - Culture: holds properties of the culture and methods to simulate the culture.
"""

####
# IMPORTS
####
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

####
# COLOUR SCHEMES
####
# colour scheme
def lighten_color(hex_color, factor=0.5):
    """
    Lighten a hex color string while retaining its alpha channel.

    Parameters:
    :param hex_color: Color in hex format (e.g., #RRGGBB or #RRGGBBAA).
    :param factor: Amount to lighten (0 = no change, 1 = white).

    Returns:
    :returns: Hex color string of the lightened color.
    """
    # Convert hex color to RGBA
    hex_color = hex_color.lstrip('#')
    rgba = tuple(int(hex_color[i:i + 2], 16) for i in range(0, len(hex_color), 2))

    # Extract RGB and alpha
    rgb = rgba[:3]
    alpha = rgba[3] if len(rgba) == 4 else 255

    # Lighten the RGB values
    light_rgb = tuple(min(255, int(c + (255 - c) * factor)) for c in rgb)

    # Return new hex color with alpha
    return "#{:02x}{:02x}{:02x}{:02x}".format(*light_rgb, alpha)

def generate_colors(n):
    """
    Generate a dark color scheme starting with predefined dark colors.
    Extends to an infinite sequence of dark colors if needed.

    Parameters:
    :param n: Number of desired colors (positive integer).

    :returns: Tuple containing (dark_colors, light_colors), where:
              - dark_colors: List of n dark colors.
              - light_colors: List of n corresponding lightened colors.
    """
    # Initial set of dark colors
    dark_colors = ['#aa0000ff', '#214478ff', '#008000ff', '#d4aa00ff','#d45500ff', '#800066ff', '#784421ff', '#37c7c8ff']
    length = len(dark_colors)

    # If n exceeds the number of initial colors, generate more colors
    if n > length:
        remaining = n - length
        for i in range(remaining):
            # Generate evenly spaced hues with low lightness/brightness to keep them "dark"
            hue = (i / remaining) % 1  # Rotate through hue spectrum
            rgb = mcolors.hsv_to_rgb((hue, 0.8, 0.5))  # Saturation=0.8, Value=0.5 for darkness
            hex_color = mcolors.to_hex(rgb, keep_alpha=False)
            dark_colors.append(hex_color)
    dark_colors=dark_colors[:n]

    # Generate lightened colors
    light_colors = [lighten_color(color, factor=0.6) for color in dark_colors]

    return dark_colors, light_colors


####
# IDEAL MICROBE CLASS
####
# Keeps record of enzyme budgets, their saturation, and possible dynamics

class Microbe:

    ##
    # INITIALIZATION AND MODIFICATION
    ##

    # Initialization function
    def __init__(self, met_noise, mich_ment, react_rate, extreme_path, stoich_int, stoich_biomass, stoich_ext=None, stoich_cost=None, enz_total=1, exp_pot=None, met_int_steady=None, met_int_units=None, enz_units=None, fba_approach=False):
        """
        Initialization of the Microbe object that includes metabolic parameters and methods to simulate the metabolism.

        :param met_noise:
            Metabolic noise T (positive float).
            FBA-like approach: take the reference value for E. coli.
            Units: num. cell divisions/time (natural), num. cell divisions/hour (fba).
        :type met_noise: float
        :param mich_ment:
            Rescaled Michaelis-Menten constants k_i0 for each reaction i (1D ndarray, positive floats).
            When the enzymes have intracellular metabolites as substrate, the value of k_i0*(steady intracellular metabolite levels
            taken to appropriate stoichiometric power) gives the true Michaelis-Menten constant.
            FBA-like approach: choose a typical level of k_i0 and set all k_i0 parameters to this value.
            Units: molar fraction (natural), molar fraction (fba).
        :type mich_ment: numpy.ndarray
        :param react_rate:
            Michaelis-Menten reaction rates v_i for each reaction i (1D ndarray, positive floats).
            FBA-like approach: choose v_i to be equal to the maximal uptake reaction rate for non-biomass reactions and
            to be equal to the typical maximal division rate of microbes for the biomass reaction.
            Units: num. enzymes dissociated/time (natural), mmol/(l*hr) for non-biomass reactions and 1/hr for biomass reaction (fba).
        :type react_rate: numpy.ndarray
        :param extreme_path:
            A matrix with rows p corresponding to distinct extreme pathways X_pi (2D ndarray, non-negative floats).
            All entries must be non-negative, which can be achieved by splitting all reversible reactions into two irreversible reactions.
            FBA-like approach: normalize the extreme pathways by constant carbon input rate measured in mmol/(l*hr).
            Units: num. enzymes dissociated/time (natural), mmol/(l*hr) for non-biomass reactions and 1/hr for biomass reaction (fba).
        :type extreme_path: numpy.ndarray
        :param stoich_int:
            Stoichiometric matrix S_bi for intracellular metabolites b (row) and reactions i (column) (2D ndarray, floats or integers).
            Any reversible reactions must be split into two irreversible reactions.
            This matrix must be used in the preprocessing step that calculates the extreme pathways.
            We use the convention of positive (resp. negative) value S_bi when an intracellular metabolites b appears as a product (resp. reactant) in reaction i.
            Units: num. intracellular metabolite molecules (natural), num. intracellular metabolite molecules for non-biomass reactions and mmol/g for biomass reaction (fba).
        :type stoich_int: numpy.ndarray
        :param stoich_biomass:
            Stoichiometric coefficients s_i+ for the biomass production in each reaction i (1D ndarray, non-negative floats).
            Natural approach: typical values 0 (resp. 1) if the reaction contributes (resp. does not contribute) to biomass production (absolute natural units), respectively proportions of 1 cell produced (relative natural units).
            FBA-like approach: specify 0 for non-biomass reactions and 1 for biomass reaction, the vector is further normalized during initialization.
            Units: num. biomass macromolecules produced (natural - absolute), proportion of cell produced (natural - relative), binary values identifying biomass reaction (fba).
        :type stoich_biomass: numpy.ndarray
        :param stoich_ext:
            Stoichiometric matrix s_ai for external metabolites a (row) and reactions i (column) (2D ndarray, floats or integers, default: None).
            If stoich_ext=None, the matrix is inferred from the stoich_int matrix by identifying (unique!) import/export reactions.
            Import/export reactions correspond to columns of S_bi that contain a single nonzero value (typically +1 for import, -1 for export).
            In such a case, the indices of extracellular metabolites follow the order of intracellular metabolites b taken up by an import/export reaction.
            When simulations with more than one Microbes are done, stoich_ext should be specified explicitly for all extracellular metabolites
            in the culture (even those unused by the microbe!) to avoid confusion due to inconsistent indexing of extracellular metabolites.
            Similarly, if import/export reactions are not unique (including cases where a single extracellular metabolite can be both imported and exported), stoich_ext should be specified explicitly.
            FBA-like approach: Specify +1 for import reactions, -1 for export reactions, and 0 otherwise; extracellular metabolites not affected by this microbe are represented by rows of 0.
            Units: num. extracellular metabolite molecules (natural), num. extracellular metabolite molecules (fba).
        :type stoich_ext: Union[numpy.ndarray, None]
        :param stoich_cost:
            Stoichiometric coefficients s_i- for the maintenance cost of each reaction i (1D ndarray, non-negative floats, default: None).
            Represents the effective number of degraded enzymes per flux in reaction i.
            If stoich_cost=None, the default zero vector is used and maintenance costs are ignored.
            FBA-like approach: Use the default value. If this vector becomes known in the future, appropriately normalize by enz_total.
            Units: num. degraded enzymes (natural), num. degraded enzymes (fba).
        :type stoich_cost: Union[numpy.ndarray, None]
        :param enz_total:
            The maximal number of enzymes Phi in a microbial cell (positive integer, default: 1).
            Natural-like approach: The default value None gets converted to 1 (relative natural units); otherwise, total number of enzymes in a cell should be used (absolute natural units).
            Absolute natural units must be used for stochastic simulations.
            FBA-like approach: Any value is changed to a default normalization of the FBA-like approach.
            Units: enzyme units (natural), dimensionless normalization (fba).
        :type enz_total: int
        :param exp_pot:
            Expression potential nu_p0 of enzymes along an extreme pathway p (1D ndarray, floats, default: None).
            If exp_pot=None, the noisy FBA is used and the expression potentials are set to the growth rates contributed by each pathway p.
            FBA-like approach: Use the default value.
            Units: num. cell divisions/time (natural), num. cell divisions/time (fba).
        :type exp_pot: Union[numpy.ndarray, None]
        :param met_int_steady:
            Stationary levels y_b0 of intracellular metabolites b (1D ndarray, positive floats, default: None).
            If met_int_steady=None, the stationary levels are set to 1.
            This parameter is only used in the dynamic proteome model.
            Units: intracellular metabolite molecule units (natural), irrelevant parameter (fba).
        :type met_int_steady: Union[numpy.ndarray, None]
        :param met_int_units:
            Specified conversion coefficients for the units of the intracellular metabolites b (1D ndarray, positive floats, default: None).
            If met_int_units=None, the default values are set to 1.
            Each entry specifies how many units of the intracellular metabolite b corresponds to one molecule.
            FBA-like approach: Use the default value.
            Units: e.g. g per 1 cell (natural), dimensionless parameter (fba).
        :type met_int_units: Union[numpy.ndarray, None]
        :param enz_units:
            Specified conversion coefficients for the units of the enzymes i (1D ndarray, positive floats, default: None).
            If enz_units=None, the default values are set to 1.
            Each entry specifies how many units of the enzyme i corresponds to one enzyme molecule.
            FBA-like approach: Use the default value.
            Units: e.g. g per 1 cell (natural), dimensionless parameter (fba).
        :type enz_units: Union[numpy.ndarray, None]
        :param fba_approach:
            Boolean value specifying whether the FBA-like units approach is used (True) or the natural units approach is used (False).
            See the description of appropriate input parameters for each approach. Notice that the application of the
            dynamic proteome model or the consumer-metabolism-resource model does not work correctly with the FBA-like approach.
            This is because the stoichiometric entries of the biomass reaction are not dimensionless.
        :type fba_approach: bool
        Defines class variables of the same name and type as input, as well as additional class variables:
            - stoich_int_in: Negative (input) stoichiometries of intracellular metabolites (ndarray with shape (self.int_num,)).
            - stoich_int_out: Positive (output) stoichiometries of intracellular metabolites (ndarray with shape (self.int_num,)).
            - stoich_ext_in: Negative (input) stoichiometries of extracellular metabolites (ndarray with shape (self.ext_num,)).
            - stoich_ext_out: Positive (output) stoichiometries of extracellular metabolites (ndarray with shape (self.ext_num,)).
            - react_num: Number of reactions (int).
            - int_num: Number of intracellular metabolites (int).
            - ext_num: Number of extracellular metabolites (int).
            - path_num: Number of extreme pathways (int).
            - class_num: Number of equivalence classes (int).
            - class_to_paths: List of equivalence classes, each of which is a list of pathway indices (list of lists).
            - path_to_class: Mapping from pathway indices to their equivalence class indices (1D ndarray of shape (path_num,)).
            - stoich_class: Matrix of stoichiometries of a representative pathway for each class (ndarray with shape (ext_num, class_num)).
            - class_weight: Matrix of reaction weights for each class (ndarray with shape (class_num, react_num)).
            - dark_colors: List of dark colours (list of strings).
            - light_colors: List of corresponding light colours (list of strings).
            - dark_light_colors: List of interleaved dark and light colours (list of strings).
        """
        # 1 - Save the parameters that do not allow for the None option
        self.met_noise = met_noise
        self.mich_ment = mich_ment
        self.react_rate = react_rate
        self.extreme_path = extreme_path
        self.stoich_int = stoich_int
        self.stoich_biomass = stoich_biomass
        self.enz_total = enz_total
        # 2 - Numbers of reactions, intracellular metabolites and extreme pathways
        self.react_num = self.stoich_int.shape[1]
        self.int_num = self.stoich_int.shape[0]
        self.path_num = self.extreme_path.shape[0]
        # 3 - Save/create the parameters with the None option
        # 3a - Stochiometric matrix for extracellular metabolites
        if stoich_ext is None:
            # Initialize an external stoichiometric matrix as a list
            ext_matrix = []
            # Iterate through the rows of stoich_int
            for row_index in range(self.int_num):
                # Get the current row
                row = self.stoich_int[row_index, :]
                # Identify columns (reactions) with a unique nonzero element at this row (import/export) that are not biomass production
                unique_columns = [
                    col_index for col_index in range(self.react_num)
                    if row[col_index] != 0 and np.count_nonzero(self.stoich_int[:, col_index]) == 1 and self.stoich_biomass[col_index] <= 0
                ]
                # If there are no such columns, continue to the next row
                if not unique_columns:
                    continue
                # Create a new row for the external stoichiometric matrix
                new_row = np.zeros_like(row)
                # Populate the new row based on the unique columns
                for col_index in unique_columns:
                    new_row[col_index] = -self.stoich_int[row_index, col_index]
                # Append the new row to the external stoichiometric matrix
                ext_matrix.append(new_row)
            # Convert the list to a numpy array for consistency
            self.stoich_ext = np.array(ext_matrix)
        else:
            self.stoich_ext = stoich_ext
        # Save the number of extracellular metabolites
        self.ext_num = self.stoich_ext.shape[0]
        # 3b - Stoichiometry of maintenance costs
        if stoich_cost is None:
            stoich_cost = np.zeros(self.react_num)
        self.stoich_cost = stoich_cost
        # 3c - Expression potential
        if exp_pot is None:
            # Calculate expression potential for each pathway, as a growth rate carried by a given pathway
            exp_pot = (self.extreme_path @ (self.stoich_biomass-stoich_cost))/self.enz_total
        # Save the calculated or provided exp_pot
        self.exp_pot = exp_pot
        # 3d - Intracellular metabolites
        if met_int_steady is None:
            met_int_steady = np.ones(self.int_num)
        self.met_int_steady = met_int_steady
        # 4 - Separate the stoichiometric matrices into matrices for reactants and for products
        self.stoich_int_in = np.where(self.stoich_int < 0, -self.stoich_int, 0)
        self.stoich_int_out = np.where(self.stoich_int > 0, self.stoich_int, 0)
        self.stoich_ext_in = np.where(self.stoich_ext < 0, -self.stoich_ext, 0)
        self.stoich_ext_out = np.where(self.stoich_ext > 0, self.stoich_ext, 0)
        # 5 - Partition the set of extreme pathways
        # Two pathways are in the same class if they have the same import stoichiometry
        # 5a- Create the matrix of maximal stoichiometries for each path
        # Initialize the stoich_ext_path matrix with the same shape as stoich_ext_in
        stoich_ext_path = np.zeros((self.path_num, self.ext_num))
        # Compute stoich_ext_path matrix
        for path in range(self.path_num):
            # Find reactions i where extreme_path[path][i] is positive
            positive_indices = np.where(self.extreme_path[path] > 0)[0]
            # For each extracellular metabolite "a" take the maximum over stoich_ext_in[a][i] for valid i
            for a in range(self.ext_num):
                stoich_ext_path[path][a] = np.max(self.stoich_ext_in[a, positive_indices]) if len(positive_indices) > 0 else 0
        # 5b - Find the equivalence classes
        # Create a dictionary to store equivalence classes
        equivalence_classes = {}
        for path, vector in enumerate(stoich_ext_path):
            # Convert the vector to a tuple since lists/arrays aren't hashable
            vector_tuple = tuple(vector)
            if vector_tuple not in equivalence_classes:
                equivalence_classes[vector_tuple] = []  # Initialize new equivalence class
            equivalence_classes[vector_tuple].append(path)
        # 5c - Save the dictionary values to a list of equivalence classes, save the number of classes
        self.class_to_paths = list(equivalence_classes.values())
        self.class_num = len(self.class_to_paths)
        # 5d - maps each path to its equivalence class
        self.path_to_class = np.zeros(self.path_num, dtype=int)
        # Update path_to_class based on class_to_paths
        for class_index, paths_in_class in enumerate(self.class_to_paths):
            for path in paths_in_class:
                self.path_to_class[path] = class_index
        # 5e - Determine the import stochiometries of a given class
        self.stoich_class = np.zeros((self.class_num, self.ext_num))
        # Assign stoichiometry of a representative pathway for each class
        for class_index, paths_in_class in enumerate(self.class_to_paths):
            # Use the first pathway as the representative for the class
            representative_path = paths_in_class[0]
            self.stoich_class[class_index] = stoich_ext_path[representative_path]
        self.stoich_class = self.stoich_class.T # Transpose to have metabolites as rows, classes as columns
        # 5f - Determine the weighting coefficients Y_ci of each reaction i for each class c
        # Initialize the class_path matrix
        self.class_weight = np.zeros((self.class_num, self.react_num))
        # Compute the expression potential weights of each path
        q = np.exp(self.exp_pot / self.met_noise)
        q = q / np.sum(q)  # Normalize q
        # Compute class_weight for each class
        for class_index, paths_in_class in enumerate(self.class_to_paths):
            for path in paths_in_class:
                # Add the contribution of this path to the class_path sum
                self.class_weight[class_index] += self.extreme_path[path] / self.react_rate * q[path]
        # Normalize the enz_total and biomass total in the FBA-like approach
        self.fba_approach = fba_approach
        if fba_approach:
            # Calculate the normalization factor in equations for mean enzyme expression and saturation levels when extracellular metabolites are absent
            met_ext = np.zeros(self.ext_num) # extracellular metabolites in the units of a molar fraction
            enz_sat = np.prod(met_ext[:,np.newaxis] ** self.stoich_class, axis=0) @ self.class_weight # extracellular metabolites along pathways
            stoich_diff = self.stoich_class[:,:,np.newaxis]-self.stoich_ext_in[:,np.newaxis,:] # differences in stoichiometry [a][c][i]
            stoich_diff = np.where(self.class_weight[np.newaxis, :, :] == 0, 0, stoich_diff) # regularize entries that do not contribute in the final calculation, i.e. where Y[c][i] is zero
            enz_exp = enz_sat+self.mich_ment*np.sum(np.prod(met_ext[:,np.newaxis,np.newaxis] ** stoich_diff, axis=0)*self.class_weight, axis=0) # add unsaturated enzymes
            norm0 = np.sum(enz_exp) # sum all enzyme expressions
            # Calculate the normalization factor in equations for mean enzyme expression and saturation levels when extracellular metabolites are present
            met_ext = np.ones(self.ext_num) # extracellular metabolites in the units of a molar fraction
            enz_sat = np.prod(met_ext[:,np.newaxis] ** self.stoich_class, axis=0) @ self.class_weight # extracellular metabolites along pathways
            stoich_diff = self.stoich_class[:,:,np.newaxis]-self.stoich_ext_in[:,np.newaxis,:] # differences in stoichiometry [a][c][i]
            stoich_diff = np.where(self.class_weight[np.newaxis, :, :] == 0, 0, stoich_diff) # regularize entries that do not contribute in the final calculation, i.e. where Y[c][i] is zero
            enz_exp = enz_sat+self.mich_ment*np.sum(np.prod(met_ext[:,np.newaxis,np.newaxis] ** stoich_diff, axis=0)*self.class_weight, axis=0) # add unsaturated enzymes
            norm1 = np.sum(enz_exp) # sum all enzyme expressions
            # set the enz_total to the average increase in the norm
            if norm1 <= norm0:
                raise Exception("Calculation of the correct normalization factor in the FBA-like approach failed. No extracellular metabolites are imported or there are negative entries in extreme pathways.")
            else:
                self.enz_total = norm1-norm0
            # modify the stoich_biomass accordingly
            self.stoich_biomass *= self.enz_total
        # Save the units
        if enz_units is None:
            self.enz_units = np.ones(self.react_num)
        else:
            self.enz_units = enz_units
        if met_int_units is None:
            self.met_int_units = np.ones(self.ext_num)
        else:
            self.met_int_units = met_int_units
        # Add a colour
        self.dark_colors, self.light_colors = generate_colors(max(self.react_num,self.ext_num))
        self.dark_light_colors = [item for pair in zip(self.dark_colors, self.light_colors) for item in pair]


    # Modification of the metabolic noise after initialization
    def change_met_noise(self, met_noise):
        """
        Safe change of the metabolic noise parameter after initialization.
        Metabolic noise is changed to the new value and the class weights are properly modified.

        :param met_noise:
            Metabolic noise T (positive float). Units: num. cell divisions/time.
        :type met_noise: float
        """
        # Save the new metabolic noise
        self.met_noise = met_noise
        # Compute the expression potential weights of each path
        q = np.exp(self.exp_pot / self.met_noise)
        q = q / np.sum(q)  # Normalize q
        # Compute class_weight for each class
        for class_index, paths_in_class in enumerate(self.class_to_paths):
            for path in paths_in_class:
                # Compute reaction rate for the given path
                # react_path[path] = prod_i (react_rate[i] ** extreme_path[path][i]) ** (1 / sum_i(extreme_path[path][i]))
                extreme = self.extreme_path[path]
                product_term = np.prod(self.react_rate ** extreme)
                sum_term = np.sum(extreme)
                react_path = product_term ** (1 / sum_term) if sum_term > 0 else 1
                # Add the contribution of this path to the class_path sum
                self.class_weight[class_index] += self.extreme_path[path] * react_path / self.react_rate * q[path]
        # Normalize the enz_total and biomass total in the FBA-like approach
        if self.fba_approach:
            # Calculate the normalization factor in equations for mean enzyme expression and saturation levels when extracellular metabolites are absent
            met_ext = np.zeros(self.ext_num) # extracellular metabolites in the units of a molar fraction
            enz_sat = np.prod(met_ext[:,np.newaxis] ** self.stoich_class, axis=0) @ self.class_weight # extracellular metabolites along pathways
            stoich_diff = self.stoich_class[:,:,np.newaxis]-self.stoich_ext_in[:,np.newaxis,:] # differences in stoichiometry [a][c][i]
            stoich_diff = np.where(self.class_weight[np.newaxis, :, :] == 0, 0, stoich_diff) # regularize entries that do not contribute in the final calculation, i.e. where Y[c][i] is zero
            enz_exp = enz_sat+self.mich_ment*np.sum(np.prod(met_ext[:,np.newaxis,np.newaxis] ** stoich_diff, axis=0)*self.class_weight, axis=0) # add unsaturated enzymes
            norm0 = np.sum(enz_exp) # sum all enzyme expressions
            # Calculate the relative increase in the normalization factor for each extracellular metabolite
            imported_ext_met = 0 # only consider those that are imported and change the normalization factor
            norm_total = 0 # total increments in normalization factors
            for met in range(self.ext_num):
                # Calculate the normalization factor in equations for mean enzyme expression and saturation levels when extracellular metabolites are absent
                met_ext = np.zeros(self.ext_num) # extracellular metabolites in the units of a molar fraction
                met_ext[met] = 1
                enz_sat = np.prod(met_ext[:,np.newaxis] ** self.stoich_class, axis=0) @ self.class_weight # extracellular metabolites along pathways
                stoich_diff = self.stoich_class[:,:,np.newaxis]-self.stoich_ext_in[:,np.newaxis,:] # differences in stoichiometry [a][c][i]
                stoich_diff = np.where(self.class_weight[np.newaxis, :, :] == 0, 0, stoich_diff) # regularize entries that do not contribute in the final calculation, i.e. where Y[c][i] is zero
                enz_exp = enz_sat+self.mich_ment*np.sum(np.prod(met_ext[:,np.newaxis,np.newaxis] ** stoich_diff, axis=0)*self.class_weight, axis=0) # add unsaturated enzymes
                norm_diff = np.sum(enz_exp)-norm0 # sum all enzyme expressions
                if norm_diff > 0: # norm is increased by the extracellular metabolite
                    imported_ext_met += 1
                    norm_total += norm_diff
            # set the enz_total to the average increase in the norm
            if imported_ext_met == 0:
                raise Exception("Calculation of the correct normalization factor in the FBA-like approach failed. No extracellular metabolites are imported.")
            else:
                self.enz_total = norm_total/imported_ext_met
            # modify the stoich_biomass accordingly
            self.stoich_biomass *= self.enz_total

    ##
    # STATIONARY PROTEOME ALLOCATION
    ##

    # Mean values of the stationary proteome allocation
    def stat_prot_mean(self, met_ext, met_ext_total=None):
        """
        Calculate the mean values in the stationary proteome allocation model.

        :param met_ext:
            Levels of extracellular metabolites (ndarray with shape (self.ext_num,)). Units: extracellular metabolites.
        :type met_ext: numpy.ndarray
        :param met_ext_total:
            Maximal levels of extracellular metabolites in the medium (ndarray with shape (self.ext_num,) or None).
            When None, it is assumed that extracellular metabolites are measured in the units of molar fraction and
            the maximum levels of extracellular metabolites are correspondingly set to 1. Units: extracellular metabolites.
        :type met_ext_total: Union(numpy.ndarray,None)
        :return: A tuple containing two numpy arrays:
            - enz_exp: Enzyme expression levels (ndarray with shape (self.react_num,)). Units: enzymes.
            - enz_sat: Enzyme saturation levels (ndarray with shape (self.react_num,)). Units: enzymes.
        """
        if met_ext_total is None: # set the maximum levels of extracellular metabolites to 1 if not provided
            met_ext_total = np.ones(self.ext_num)
        met_ext_rel = met_ext / met_ext_total # extracellular metabolites in the units of a molar fraction
        enz_sat = np.prod(met_ext_rel[:,np.newaxis] ** self.stoich_class, axis=0) @ self.class_weight # extracellular metabolites along pathways
        stoich_diff = self.stoich_class[:,:,np.newaxis]-self.stoich_ext_in[:,np.newaxis,:] # differences in stoichiometry [a][c][i]
        stoich_diff = np.where(self.class_weight[np.newaxis, :, :] == 0, 0, stoich_diff) # regularize entries that do not contribute in the final calculation, i.e. where Y[c][i] is zero
        enz_exp = enz_sat+self.mich_ment*np.sum(np.prod(met_ext_rel[:,np.newaxis,np.newaxis] ** stoich_diff, axis=0)*self.class_weight, axis=0) # add unsaturated enzymes
        norm = np.sum(enz_exp) # sum all enzyme expressions
        if norm == 0: # treat the case where norm is zero
            enz_sat = np.zeros(self.react_num)
            enz_exp = np.sum(self.class_weight, axis=0)
            norm = np.sum(enz_exp)
        enz_exp = self.enz_total * enz_exp / norm # normalize enzyme expression
        enz_sat = self.enz_total * enz_sat / norm # normalize enzyme saturation
        return enz_exp, enz_sat

    # Stochastic sampling of the stationary proteome allocation
    def stat_prot_sample(self, met_ext, sample_num, met_ext_total=None):
        """
        Sample stationary proteome allocation of sample_num cells.
        Note that this function only performs the sampling if budget_cap is not normalized and is set to a value > 1.

        :param met_ext:
            Levels of extracellular metabolites (ndarray with shape (self.ext_num,)). Units: extracellular metabolites.
        :type met_ext: numpy.ndarray
        :param sample_num:
            Number of sampled cells (positive integer).
        :type sample_num: int
        :param met_ext_total:
            Maximal levels of extracellular metabolites in the medium in chosen units (ndarray with shape (self.ext_num,) or None).
            When None, it is assumed that extracellular metabolites are measured in the units of molar fraction and
            the maximum levels of extracellular metabolites are correspondingly set to 1. Units: extracellular metabolites.
        :type met_ext_total: Union(numpy.ndarray,None)
        :return: A tuple containing two numpy arrays:
            - enz_exp: Enzyme expression levels (ndarray with shape (sample_num,self.react_num,)). Units: enzymes.
            - enz_sat: Enzyme saturation levels (ndarray with shape (sample_num,self.react_num,)). Units: enzymes.
        """
        if self.enz_total == 1:
            raise ValueError("Total number of enzymes is normalized to 1, cannot sample. Please provide a realistic enz_total.")
        else:
            if met_ext_total is None: # set the maximum levels of extracellular metabolites to 1 if not provided
                met_ext_total = np.ones(self.ext_num)
            met_ext_rel = met_ext / met_ext_total # extracellular metabolites in the units of a molar fraction
            stoich_diff = self.stoich_class[:,:,np.newaxis]-self.stoich_ext_in[:,np.newaxis,:] # differences in stoichiometry [a][c][i]
            stoich_diff = np.where(self.class_weight[np.newaxis, :, :] == 0, 0, stoich_diff) # regularize entries that do not contribute in the final calculation, i.e. where Y[c][i] is zero
            exp_weight = self.mich_ment*np.sum(np.prod(met_ext_rel[:,np.newaxis] ** stoich_diff, axis=0)*self.class_weight, axis=0) # find the weights for enzyme saturation
            exp_weight = exp_weight/np.sum(exp_weight) # normalize the weights
            sat_weight = np.prod(met_ext_rel[:,np.newaxis] ** self.stoich_ext_in, axis=0)
            sat_weight = sat_weight/(self.mich_ment+sat_weight)
            enz_exp = np.random.multinomial(self.enz_total, exp_weight, size=sample_num)
            enz_sat = np.zeros((sample_num,self.react_num))
            for sample in range(sample_num):
                for react in range(self.react_num):
                    enz_sat[sample][react] = np.random.binomial(enz_exp[sample][react],sat_weight[react])
            return enz_exp, enz_sat
        
    ##
    # DYNAMIC PROTEOME ALLOCATION
    ##

    # Slice the prot_state into enz_exp, enz_sat and met_int (needed for ODE solver)
    def slice_prot_state(self, prot_state):
        """
        Slices the given proteome state (resp. its forcing) into three parts: enz_exp, enz_sat and met_int.
        Can only be used in the natural units.

        :param prot_state:
            The collection representing the current state of the proteome enz_exp, enz_sat, met_int (ndarray with shape (2*self.react_num+self.int_num,))
        :type prot_state: numpy.ndarray
        :returns: A tuple containing three numpy arrays:
            - enz_exp: Enzyme expression levels (ndarray with shape (self.react_num,)). Units: enzymes.
            - enz_sat: Enzyme saturation levels (ndarray with shape (self.react_num,)). Units: enzymes.
            - met_int: Levels of intracellular metabolites (ndarray with shape (self.int_num,)). Units: intracellular metabolites.
        """
        return prot_state[:self.react_num], prot_state[self.react_num:2 * self.react_num], prot_state[2*self.react_num:]

    # Augment enz_exp, enz_sat and met_int to prot_state (needed for ODE solver)
    def augment_prot_state(self, enz_exp, enz_sat, met_int):
        """
        Combines enzyme expression levels (enz_exp), enzyme saturation levels (enz_sat) and intracellular metabolite
        levels (met_int) into a single proteome state (prot_state). Works for the corresponding forcing as well.
        Can only be used in the natural units.

        :param enz_exp: Enzyme expression levels (ndarray with shape (self.react_num,)). Units: enzymes.
        :type enz_exp: numpy.ndarray
        :param enz_sat: Enzyme saturation levels (ndarray with shape (self.react_num,)). Units: enzymes.
        :type enz_sat: numpy.ndarray
        :param met_int: Levels of intracellular metabolites (ndarray with shape (self.int_num,)). Units: intracellular metabolites.
        :type met_int: numpy.ndarray
        :returns: prot_state: a numpy array containing enz_exp, enz_sat, met_int (ndarray with shape (2*self.react_num+self.int_num,)).
        """
        return np.concatenate((enz_exp, enz_sat, met_int))

    # Forcing function in the dynamic proteome allocation model
    def dyn_prot_forcing(self, enz_exp, enz_sat, met_int, met_ext, met_ext_total=None):
        """
        Calculate the forcing function in the dynamic proteome allocation model.
        Can only be used in the natural units.

        :param enz_exp:
            Enzyme saturation levels (ndarray with shape (self.react_num,)). Units: enzymes.
        :type enz_exp: numpy.ndarray
        :param enz_sat:
            Enzyme expression levels (ndarray with shape (self.react_num,)). Units: enzymes.
        :type enz_sat: numpy.ndarray
        :param met_int:
            Levels of intracellular metabolites (ndarray with shape (self.int_num,)). Units: intracellular metabolites.
        :type met_int: numpy.ndarray
        :param met_ext:
            Levels of extracellular metabolites (ndarray with shape (self.ext_num,)). Unist: extracellular metabolites.
        :type met_ext: numpy.ndarray
        :param met_ext_total:
            Maximal levels of extracellular metabolites in the medium (ndarray with shape (self.ext_num,) or None).
            When None, it is assumed that extracellular metabolites are measured in the units of molar fraction and
            the maximum levels of extracellular metabolites are correspondingly set to 1. Units: extracellular metabolites.
        :type met_ext_total: Union(numpy.ndarray,None)
        :return: A tuple containing three numpy arrays:
            - exp_force: The forcing on enzyme expression rates (ndarray with shape (self.react_num,)). Units: enzymes/time.
            - sat_force: The forcing on enzyme saturation rates (ndarray with shape (self.react_num,)). Units: enzymes/time.
            - int_force: The forcing on intracellular metabolite levels (ndarray with shape (self.int_num,)). Units: intracellular metabolites/time.
        """
        if met_ext_total is None: # set the maximum levels of extracellular metabolites to 1 if not provided
                met_ext_total = np.ones(self.ext_num)
        met_ext_rel = met_ext / met_ext_total # extracellular metabolites in the units of a molar fraction
        # Growth rate contributions
        growth_up, growth_down = self.growth_rate_contributions(enz_sat)  # positive/negative contribution to growth rate
        # Saturation rate forcing
        sat_up = enz_exp-enz_sat # unsaturated enzymes can get saturated
        sat_up *= np.prod(met_ext_rel[:, np.newaxis] ** self.stoich_ext_in, axis=0) # extracellular metabolites that bind to saturate
        sat_up *= np.prod((met_int / self.met_int_steady)[:, np.newaxis] ** self.stoich_int_in, axis=0) # extracellular metabolites that bind to saturate
        sat_up *= self.react_rate/self.mich_ment # Michaelis-Menten kinetics
        sat_down = self.react_rate*enz_sat # saturated enzymes dissociate
        sat_force = sat_up-sat_down-growth_up*enz_sat # forcing includes saturation increase/decrease and dilution
        # Expression rate forcing
        exp_stat, sat_stat = self.stat_prot_mean(met_ext, met_ext_total) # stationary enzyme expression
        exp_force = growth_up*(exp_stat-enz_exp) # forcing includes translation and dilution
        # Intracellular metabolites forcing
        int_force = self.met_int_units*(self.stoich_int_in @ (sat_down/self.enz_units) - self.stoich_int_out @ (sat_up/self.enz_units)) # forcing includes chemical reactions
        int_force -= (growth_up-growth_down)*met_int # forcing also includes dilution due to growth
        int_force += self.met_int_units*(self.stoich_int_out @ (enz_sat/self.enz_units))*growth_down # forcing also includes released metabolites during enzyme degradation
        return exp_force, sat_force, int_force

    # ODE solver for the dynamic proteome allocation model
    def dyn_prot_ode(self, t_span, enz_exp0, enz_sat0, met_int0, met_ext, met_ext_total=None, **kwargs):
        """
        Solves the ODE system for the proteome dynamics. Can only be used in the natural units.

        :param t_span: Tuple (start_time, end_time) specifying the time span for ODE integration. Units: time.
        :type t_span: tuple
        :param enz_exp0: Initial enzyme expression levels (ndarray with shape (self.react_num,)). Units: enzymes.
        :type enz_exp0: numpy.ndarray
        :param enz_sat0: Initial enzyme saturation levels (ndarray with shape (self.react_num,)). Units: enzymes.
        :type enz_sat0: numpy.ndarray
        :param met_int0: Initial intracellular metabolite levels (ndarray with shape (self.int_num,)). Units: intracellular metabolites.
        :type met_int0: numpy.ndarray
        :param met_ext: Constant extracellular metabolite levels (ndarray with shape (self.ext_num,)). Units: extracellular metabolites.
        :type met_ext: numpy.ndarray
        :param met_ext_total:
            Maximal levels of extracellular metabolites in the medium (ndarray with shape (self.ext_num,) or None).
            When None, it is assumed that extracellular metabolites are measured in the units of molar fraction and
            the maximum levels of extracellular metabolites are correspondingly set to 1. Units: extracellular metabolites.
        :type met_ext_total: Union(numpy.ndarray,None)
        :param kwargs: Additional keyword arguments for the ODE solver.
        :returns: The solution to the ODE system as an `OdeResult` object, with an additional attribute `growth`
        containing instantaneous growth rates (ndarray of the shape (t.size,).
        """
        # Augment initial values into a single prot_state
        prot_state0 = self.augment_prot_state(enz_exp0, enz_sat0, met_int0)

        # Define the ODE system as a function
        def ode_system(t, prot_state):
            enz_exp, enz_sat, met_int = self.slice_prot_state(prot_state)
            exp_force, sat_force, int_force = self.dyn_prot_forcing(enz_exp, enz_sat, met_int, met_ext, met_ext_total)
            return self.augment_prot_state(exp_force, sat_force, int_force)
        # Set up the solver
        kwargs["fun"]=ode_system
        kwargs["t_span"]=t_span
        kwargs["y0"]=prot_state0
        if "method" not in kwargs:
            kwargs["method"]="RK45"
        # Solve the ODE system using solve_ivp
        solution = sp.integrate.solve_ivp(**kwargs)
        # Add instantaneous growth rates to the solution
        enz_exp, enz_sat, met_int = self.slice_solution(solution)
        solution.growth = np.zeros(solution.t.shape[0])
        for t_idx in range(solution.t.shape[0]):
            solution.growth[t_idx] = self.growth_rate(enz_sat[:,t_idx])
        return solution

    # Slice the solution of the ODE solver into enz_exp, enz_sat and met_int for all time points
    def slice_solution(self, solution):
        """
        Slices the outuput solution of the dyn_prot_ode() function into three parts: enz_exp, enz_sat and met_int.
        Can only be used in the natural units.

        :param solution:
            The output solution of the dyn_prot_ode() function (OdeResult object).
        :type solution: scipy.integrate.OdeResult
        :returns: A tuple containing two numpy arrays:
            - enz_exp: Enzyme expression levels (ndarray with shape (self.react_num, t_point_num)). Units: enzymes.
            - enz_sat: Enzyme saturation levels (ndarray with shape (self.react_num, t_point_num)). Units: enzymes.
            - met_int: Levels of intracellular metabolites (ndarray with shape (self.int_num, t_point_num)). Units: intracellular metabolites.
        """
        prot_state = solution.y
        return prot_state[:self.react_num, :], prot_state[self.react_num:2 * self.react_num, :], prot_state[2 * self.react_num:, :]


    ##
    # FORCING FUNCTIONS OF CONSUMER-RESOURCE AND CONSUMER-METABOLISM-RESOURCE MODELS
    ##

    # Forcing functions in the consumer-resource model that excludes dilution of the culture and nutrient import
    def cr_forcing(self, met_ext, met_ext_total=None):
        """
        Calculate the growth rate and total production-consumption rates in the consumer-resource model.

        :param met_ext:
            Levels of extracellular metabolites (ndarray with shape (self.ext_num,)). Units: extracellular metabolites.
        :type met_ext: numpy.ndarray
        :param met_ext_total:
            Maximal levels of extracellular metabolites in the medium (ndarray with shape (self.ext_num,) or None).
            When None, it is assumed that extracellular metabolites are measured in the units of molar fraction and
            the maximum levels of extracellular metabolites are correspondingly set to 1. Units: extracellular metabolites.
        :type met_ext_total: Union(numpy.ndarray,None)
        :return: A tuple containing:
            - growth_rate: Growth rate of the microbial species (float). Units: num. cell divisions/time.
            - prod_cons_rate: Total production-consumption rates for all extracellular metabolites (ndarray with shape (self.ext_num,)). Units: num. consumed or produced molecules/cell/time (natural), mmol/g/hr (fba).
        """
        enz_exp, enz_sat = self.stat_prot_mean(met_ext, met_ext_total)
        growth_rate = self.growth_rate(enz_sat)
        prod_cons_rate = self.stat_production_consumption(enz_sat)
        return growth_rate, prod_cons_rate

    # Forcing functions in the consumer-metabolism-resource model that excludes dilution of the culture and nutrient import
    def cmr_forcing(self, prot_state, met_ext, met_ext_total=None):
        """
        Calculate the forcing function in the dynamic proteome allocation model. Can only be used in the natural units.

        :param prot_state:
            Proteome state that includes enzyme expression and saturation levels and intracellular metabolite levels
            (ndarray with shape (2*self.react_num+self.int_num,)).
        :type prot_state: numpy.ndarray
        :param met_ext:
            Levels of extracellular metabolites (ndarray with shape (self.ext_num,)). Units: extracellular metabolites.
        :type met_ext: numpy.ndarray
        :param met_ext_total:
            Maximal levels of extracellular metabolites in the medium (ndarray with shape (self.ext_num,) or None).
            When None, it is assumed that extracellular metabolites are measured in the units of molar fraction and
            the maximum levels of extracellular metabolites are correspondingly set to 1. Units: extracellular metabolites.
        :type met_ext_total: Union(numpy.ndarray,None)
        :return: A tuple containing:
            - growth_rate: Growth rate of the microbial species (float). Units: num. cell divisions/time.
            - prod_cons_rate: Total production-consumption rates for all extracellular metabolites (ndarray with shape (self.ext_num,)). Units: num. consumed or produced molecules/cell/time.
            - prot_force: The forcing on enzymes and intracellular metabolites (ndarray with shape (2*self.react_num+self.int_num,)). Units: enzymes/time, resp. intracellular metabolites/time.
        """
        # Set the maximum levels of extracellular metabolites to 1 if not provided
        if met_ext_total is None:
                met_ext_total = np.ones(self.ext_num)
        met_ext_rel = met_ext / met_ext_total # extracellular metabolites in the units of a molar fraction
        # Unpack prot_state
        enz_exp, enz_sat, met_int = self.slice_prot_state(prot_state)
        # Growth rate contributions
        growth_up, growth_down = self.growth_rate_contributions(enz_sat)  # positive/negative contribution to growth rate
        # Saturation rate forcing
        sat_up = enz_exp-enz_sat # unsaturated enzymes can get saturated
        sat_up *= np.prod(met_ext_rel[:, np.newaxis] ** self.stoich_ext_in, axis=0) # extracellular metabolites that bind to saturate
        sat_up *= np.prod((met_int / self.met_int_steady)[:, np.newaxis] ** self.stoich_int_in, axis=0) # extracellular metabolites that bind to saturate
        sat_up *= self.react_rate/self.mich_ment # Michaelis-Menten kinetics
        sat_down = self.react_rate*enz_sat # saturated enzymes dissociate
        sat_force = sat_up-sat_down-growth_up*enz_sat # forcing includes saturation increase/decrease and dilution
        # Expression rate forcing
        exp_stat, sat_stat = self.stat_prot_mean(met_ext, met_ext_total) # stationary enzyme expression
        exp_force = growth_up*(exp_stat-enz_exp) # forcing includes translation and dilution
        # Intracellular metabolites forcing
        int_force = self.met_int_units*(self.stoich_int_in @ (sat_down/self.enz_units) - self.stoich_int_out @ (sat_up/self.enz_units)) # forcing includes chemical reactions
        int_force -= (growth_up-growth_down)*met_int # forcing also includes dilution due to growth
        int_force += self.met_int_units*(self.stoich_int_out @ (enz_sat/self.enz_units))*growth_down # forcing also includes released metabolites during enzyme degradation
        # Production minus consumption rate
        prod_cons_rate = self.stoich_ext_out @ (sat_down/self.enz_units) - self.stoich_ext_in @ (sat_up/self.enz_units)
        prot_force = self.augment_prot_state(exp_force, sat_force, int_force)
        return growth_up-growth_down, prod_cons_rate, prot_force

    ##
    # RATES
    ##
    
    # Microbial growth rate per individual microbial cell.
    def growth_rate(self, enz_sat):
        """
        Microbial growth rate per individual microbial cell.
        :param enz_sat: Enzyme saturation levels (ndarray with shape (self.react_num,)). Units: enzymes.
        :type enz_sat: numpy.ndarray
        :returns: growth rate (float). Units: num. cell divisions/time.
        """
        return np.sum((self.stoich_biomass-self.stoich_cost)*self.react_rate*enz_sat)/self.enz_total

    # Positive and negative contributions to the microbial growth rate per individual microbial cell.
    def growth_rate_contributions(self, enz_sat):
        """
        Positive and negative contributions to the microbial growth rate per individual microbial cell.
        Positive contribution comes from the translation of new enzymes, negative contribution comes from maintenance
        costs that lead to malfunctioning/degradation of the old enzymes.
        :param enz_sat: Enzyme saturation levels (ndarray with shape (self.react_num,)). Units: enzymes.
        :type enz_sat: numpy.ndarray
        :returns: Tuple containing two numbers:
            - positive contribution to growth rate (non-negative float),
            - negative contribution to growth rate (non-negative float).
            Units: num. cell divisions or destructions/time.
        """
        return np.sum(self.stoich_biomass*self.react_rate*enz_sat) / self.enz_total, np.sum(self.stoich_cost * self.react_rate * enz_sat) / self.enz_total

    # Production rate of extracellular metabolites
    def ext_production(self, enz_sat):
        """
        Production rate of extracellular metabolites per individual microbial cell.
        :param enz_sat: Enzyme saturation levels (ndarray with shape (self.react_num,)). Units: enzymes.
        :type enz_sat: numpy.ndarray
        :returns: production rate of extracellular metabolites (ndarray with shape (self.ext_num,)). Units: num. produced molecules/cell/time (natural), mmol/g/hr (fba).
        """
        return self.stoich_ext_out @ (self.react_rate*enz_sat/self.enz_units)

    # Consumption rate of extracellular metabolites
    def ext_consumption(self, enz_exp, enz_sat, met_ext, met_int=None, met_ext_total=None):
        """
        Consumption rate of extracellular metabolites per individual microbial cell.

        :param enz_exp: Enzyme expression levels (ndarray with shape (self.react_num,)). Units: enzymes.
        :type enz_exp: numpy.ndarray
        :param enz_sat: Enzyme saturation levels (ndarray with shape (self.react_num,)). Units: enzymes.
        :type enz_sat: numpy.ndarray
        :param met_ext: Extracellular metabolite levels (ndarray with shape (self.ext_num,)). Units: extracellular metabolites.
        :type met_ext: numpy.ndarray
        :param met_int: Intracellular metabolite levels (ndarray with shape (self.int_num,)).
            When met_int is not provided, the consumption rate is calculated assuming that the intracellular metabolites
            are at their steady state. Units: intracellular metabolites.
        :type met_int: Union[numpy.ndarray,None]
        :param met_ext_total:
            Maximal levels of extracellular metabolites in the medium (ndarray with shape (self.ext_num,) or None).
            When None, it is assumed that extracellular metabolites are measured in the units of molar fraction and
            the maximum levels of extracellular metabolites are correspondingly set to 1. Units: extracellular metabolites.
        :type met_ext_total: Union(numpy.ndarray,None)
        :returns: consumption rate of extracellular metabolites (ndarray with shape (self.ext_num,)). Units: num. consumed molecules/cell/time (natural), mmol/g/hr (fba).
        """
        # Set the maximum levels of extracellular metabolites to 1 if not provided
        if met_ext_total is None:
            met_ext_total = np.ones(self.ext_num)
        met_ext_rel = met_ext / met_ext_total # extracellular metabolites in the units of a molar fraction
        # Set intracellular metabolites to their stationary levels if values not provided
        if met_int is None:
            met_int = self.met_int_steady
        # Compute the rates at which enzymes gain saturation
        sat_up = enz_exp-enz_sat # unsaturated enzymes can get saturated
        sat_up *= np.prod(met_ext_rel[:, np.newaxis] ** self.stoich_ext_in, axis=0) # extracellular metabolites that bind to saturate
        sat_up *= np.prod((met_int / self.met_int_steady)[:, np.newaxis] ** self.stoich_ext_in, axis=0) # extracellular metabolites that bind to saturate
        sat_up *= self.react_rate/self.mich_ment # Michaelis-Menten kinetics
        # Calculate the consumption rate
        return self.stoich_ext_in @ (sat_up/self.enz_units)

    # Stationary production-consumption rates of extracellular metabolites
    def stat_production_consumption(self, enz_sat):
        """
        Compute the stationary production minus consumption rates for extracellular metabolites metabolites.

        :param enz_sat: The stationary saturation levels of enzymes in the system (ndarray of shape (self.react_num,)). Units: enzymes.
        :type enz_sat: numpy.ndarray
        :returns: production-consumption rates for extracellular metabolites (ndarray of shape (self.ext_num,)). Units: num. consumed or produced molecules/cell/time (natural), mmol/g/hr (fba).
        """
        return self.stoich_ext @ (self.react_rate*enz_sat/self.enz_units)

    # Infer the parameters of the Monod equation
    def infer_monod_parameters(self,met_ext,met_ext_index, met_ext_total=None):
        """
        Infer the parameters of the Monod equation: r, m, p, p0 and K, where the growth is then given by (r*x-m*K)/(K+x)
        and the production-consumption rate by (p*x+p0*K)/(K+x).
        The function works correctly only if unit input stoichiometry is used and the standard Monod equation applies.

        :param met_ext: Levels of extracellular metabolites.
        :type met_ext: numpy.ndarray
        :param met_ext_index: Index of the extracellular metabolite for which Monod parameters are inferred
        :type met_ext_index: int
        :param met_ext_total:
            Maximal levels of extracellular metabolites in the medium (ndarray with shape (self.ext_num,) or None).
            When None, it is assumed that extracellular metabolites are measured in the units of molar fraction and
            the maximum levels of extracellular metabolites are correspondingly set to 1. Units: extracellular metabolites.
        :type met_ext_total: Union(numpy.ndarray,None)
        :return A tuple containing 5 floats: r (Unit: num. cell divisions/time), m (Unit: num. cells degraded/time),
                p (Unit: num. consumed or produced molecules/cell/time (natural), mmol/g/hr (fba)), p0 (Unit: num. consumed or produced molecules/cell/time (natural), mmol/g/hr (fba)),
                K (Unit: molar fraction).
        """
        # Check that Monod kinetics is valid
        if not all(x in (0, 1) for x in np.ravel(self.stoich_ext_in)):
            raise ValueError("All negative entries in stoich_ext must be -1.")
        # Extract three points of growth and consumption
        growth_vals = []
        prod_cons_vals = []
        for x in range(3):
            met_ext[met_ext_index] = x
            enz_exp, enz_sat = self.stat_prot_mean(met_ext, met_ext_total)
            growth_vals.append(self.growth_rate(enz_sat))
            prod_cons_vals.append(self.stat_production_consumption(enz_sat))
        # Extract the values
        # Compute K
        K_growth = 2*(growth_vals[1] - growth_vals[2]) / (growth_vals[2] - 2*growth_vals[1] + growth_vals[0])
        K_prod_cons = 2*(prod_cons_vals[1] - prod_cons_vals[2]) / (prod_cons_vals[2] - 2 * prod_cons_vals[1] + prod_cons_vals[0])
        # Compute m, p0
        p0 = prod_cons_vals[0]
        m = - growth_vals[0]
        # Compute r, p
        r = (K_growth+1)*growth_vals[1]-K_growth*growth_vals[0]
        p = (K_prod_cons + 1) * prod_cons_vals[1] - K_growth * prod_cons_vals[0]
        return r, m, p, p0, K_growth

    ##
    # PLOTTING
    ##

    # Plot Monod equation
    def plot_monod(self, met_ext, met_ext_index, met_ext_max, ax, met_ext_total=None, **kwargs):
        """
        Plot the Monod equation when extracellular metabolite with met_ext_index is varied between 0 and met_ext_max,
        while the remaining metabolites are fixed at prescribed values.
        :param met_ext: Prescribed extracellular metabolite levels (ndarray with shape (self.ext_num,)). Units: extracellular metabolites.
        :type met_ext: numpy.ndarray
        :param met_ext_index: Index of the extracellular metabolite to be varied (int).
        :type met_ext_index: int
        :param met_ext_max: Maximum value of the extracellular metabolite level to be varied (positive float). Units: extracellular metabolites.
        :type met_ext_max: float
        :param ax: axes used for plotting (matplotlib.axes.Axes object).
        :type ax: matplotlib.axes.Axes
        :param met_ext_total:
            Maximal levels of extracellular metabolites in the medium (ndarray with shape (self.ext_num,) or None).
            When None, it is assumed that extracellular metabolites are measured in the units of molar fraction and
            the maximum levels of extracellular metabolites are correspondingly set to 1. Units: extracellular metabolites.
        :type met_ext_total: Union(numpy.ndarray,None)
        :param kwargs: optional keyword arguments for the plot function.
        :returns: curve: plotted Monod curve (list of matplotlib.lines.Line2D objects).
        """
        # prepare axes
        num = 100
        ext_met_axis = np.linspace(0,met_ext_max,num=num)
        growth_axis = np.zeros(num)
        # obtain growth rate for each nutrient concentration
        for num_i in range(num):
            # set nutrient level to a given nut_axis concentration
            met_ext[met_ext_index] = ext_met_axis[num_i]
            # find appropriate expression and saturation levels
            enz_exp, enz_sat = self.stat_prot_mean(met_ext, met_ext_total)
            # find the growth rate
            growth_axis[num_i] = self.growth_rate(enz_sat)
        # If colour not specified, use the default colour scheme
        if "color" not in kwargs:
            kwargs["color"]=self.dark_colors[met_ext_index]
        # Make the plot
        curve = ax.plot(ext_met_axis, growth_axis, **kwargs)
        # Set limits limits
        ax.set_xlim((0,met_ext_max))
        ax.set_ylim((np.min(growth_axis),np.max(growth_axis)*1.1))
        return curve

    # Plot consumption rates
    def plot_production_consumption(self, met_ext, met_ext_plot_index, met_ext_vary_index, met_ext_max, ax, met_ext_total=None, **kwargs):
        """
        Plot the production/consumption rate of the extracellular metabolite with met_ext_plot_index, when
        the metabolite with met_ext_vary_index is varied between 0 and met_ext_max, while the remaining metabolites are fixed
        at prescribed values.
        :param met_ext: Prescribed extracellular metabolite levels (ndarray with shape (self.ext_num,)). Units: extracellular metabolites.
        :type met_ext: numpy.ndarray
        :param met_ext_plot_index: Index of the extracellular metabolite to be plotted (int).
        :type met_ext_plot_index: int
        :param met_ext_vary_index: Index of the extracellular metabolite to be varied (int).
        :type met_ext_vary_index: int
        :param met_ext_max: Maximum value of the extracellular metabolite level to be varied (positive float). Units: extracellular metabolites.
        :type met_ext_max: float
        :param ax: axes used for plotting (matplotlib.axes.Axes object).
        :type ax: matplotlib.axes.Axes
        :param met_ext_total:
            Maximal levels of extracellular metabolites in the medium (ndarray with shape (self.ext_num,) or None).
            When None, it is assumed that extracellular metabolites are measured in the units of molar fraction and
            the maximum levels of extracellular metabolites are correspondingly set to 1. Units: extracellular metabolites.
        :type met_ext_total: Union(numpy.ndarray,None)
        :param kwargs: optional keyword arguments for the plot function.
        :returns: curve: plotted consumption-production curve (list of matplotlib.lines.Line2D objects).
        """
        # prepare axes
        num = 100
        ext_met_axis = np.linspace(0,met_ext_max,num=num)
        cons_axis = np.zeros(num)
        # obtain growth rate for each nutrient concentration
        for num_i in range(num):
            # set nutrient level to a given nut_axis concentration
            met_ext[met_ext_vary_index] = ext_met_axis[num_i]
            # find appropriate expression and saturation levels
            enz_exp, enz_sat = self.stat_prot_mean(met_ext, met_ext_total)
            # find the growth rate
            cons_axis[num_i] = np.abs(self.stat_production_consumption(enz_sat)[met_ext_plot_index])
        # If colour not specified, use the default colour scheme
        if "color" not in kwargs:
            kwargs["color"]=self.dark_colors[met_ext_plot_index]
        if "linestyle" not in kwargs:
            kwargs["linestyle"]="--"
        # Make the plot
        curve = ax.plot(ext_met_axis, cons_axis, **kwargs)
        # Set limits limits
        ax.set_xlim(0,met_ext_max)
        ax.set_ylim(np.min(cons_axis),np.max(cons_axis)*1.1)
        return curve

    # Plot Herbert-Pirt equation
    def plot_herbert_pirt(self, met_ext, met_ext_index, met_ext_max, ax, met_ext_total=None, **kwargs):
        """
        Plot the Herbert-Pirt equation when extracellular metabolite with met_ext_index is varied between 0 and met_ext_max,
        while the remaining metabolites are fixed at prescribed values.
        :param met_ext: Prescribed extracellular metabolite levels (ndarray with shape (self.ext_num,)). Units: extracellular metabolites.
        :type met_ext: numpy.ndarray
        :param met_ext_index: Index of the extracellular metabolite to be varied (int).
        :type met_ext_index: int
        :param met_ext_max: Maximum value of the extracellular metabolite level to be varied (positive float). Units: extracellular metabolites.
        :type met_ext_max: float
        :param ax: axes used for plotting (matplotlib.axes.Axes object).
        :type ax: matplotlib.axes.Axes
        :param met_ext_total:
            Maximal levels of extracellular metabolites in the medium (ndarray with shape (self.ext_num,) or None).
            When None, it is assumed that extracellular metabolites are measured in the units of molar fraction and
            the maximum levels of extracellular metabolites are correspondingly set to 1. Units: extracellular metabolites.
        :type met_ext_total: Union(numpy.ndarray,None)
        :param kwargs: optional keyword arguments for the plot function.
        :returns: curve: plotted Herbert-Pirt curve (list of matplotlib.lines.Line2D objects).
        """
        # prepare axes
        num = 100
        ext_met_axis = np.linspace(0, met_ext_max, num=num)
        growth_axis = np.zeros(num)
        consumption_axis = np.zeros(num)
        # obtain growth rate for each nutrient concentration
        for num_i in range(num):
            # set nutrient level to a given nut_axis concentration
            met_ext[met_ext_index] = ext_met_axis[num_i]
            # find appropriate expression and saturation levels
            enz_exp, enz_sat = self.stat_prot_mean(met_ext, met_ext_total)
            # find the growth rate
            growth_axis[num_i] = self.growth_rate(enz_sat)
            # find the absolute value of the production-consumption rate (i.e. the effective production/consumption rate)
            consumption_axis[num_i] = np.abs(self.stat_production_consumption(enz_sat)[met_ext_index])
        # If colour not specified, use the default colour scheme
        if "color" not in kwargs:
            kwargs["color"] = self.dark_colors[met_ext_index]
        # Make the plot
        curve = ax.plot(growth_axis, consumption_axis, **kwargs)
        # Set limits limits
        ax.set_xlim(0, np.max(growth_axis) * 1.1)
        ax.set_ylim(0, np.max(consumption_axis) * 1.1)
        return curve

    # Plot the plane of proteome allocation
    def plot_prot_plane(self, prot_selection, met_ext, met_ext_index, met_ext_max, ax, met_ext_total=None, plot_exp=True, **kwargs):
        """
        Plot the relative proportion of selected proteome in the surface where extracellular metabolites with
        met_ext_index are varied between 0 and met_ext_max, while the remaining metabolites are fixed at prescribed values.

        :param prot_selection: List/tuple of two boolean arrays specifying which enzyme budgets are related to first and second
         extracellular metabolite (list/tuple of two ndarray with shape (self.react_num)).
        :type prot_selection: list[numpy.ndarray]
        :param met_ext: Prescribed extracellular metabolite levels (ndarray with shape (self.ext_num,)). Units: extracellular metabolites.
        :type met_ext: numpy.ndarray
        :param met_ext_index: List/tuple of the two indices of the extracellular metabolite to be varied (list/tuple of two integers).
        :type met_ext_index: list[int]
        :param met_ext_max: List/tuple of the two maximum values of the extracellular metabolite levels to be varied (list/tuple of two positive floats). Units: extracellular metabolites.
        :type met_ext_max: list[float]
        :param ax: axes used for plotting (matplotlib.axes.Axes object).
        :type ax: matplotlib.axes.Axes
        :param met_ext_total:
            Maximal levels of extracellular metabolites in the medium (ndarray with shape (self.ext_num,) or None).
            When None, it is assumed that extracellular metabolites are measured in the units of molar fraction and
            the maximum levels of extracellular metabolites are correspondingly set to 1. Units: extracellular metabolites.
        :type met_ext_total: Union(numpy.ndarray,None)
        :param plot_exp: Boolean indicating whether to plot enzyme expression levels (True) or enzyme saturation levels (False) (bool, default is True).
        :type plot_exp: bool
        :param kwargs: optional keyword arguments for the plot function.
        :returns: heatmap: proteome heatmap (matplotlib.collections.QuadMesh object).
        """
        # prepare axes
        num = 100
        met_ext_axis0 = np.linspace(0, met_ext_max[0], num=num)
        met_ext_axis1 = np.linspace(0, met_ext_max[1], num=num)
        ext0, ext1 = np.meshgrid(met_ext_axis0, met_ext_axis1)
        prot = np.zeros((num,num))
        # obtain growth rate for each nutrient concentration
        for num0_i in range(num):
            for num1_i in range(num):
                # set nutrients
                met_ext[met_ext_index[0]] = met_ext_axis0[num0_i]
                met_ext[met_ext_index[1]] = met_ext_axis1[num1_i]
                # find appropriate expression and saturation levels
                enz_exp, enz_sat = self.stat_prot_mean(met_ext, met_ext_total)
                # set proteome value if expression levels are plotted
                if plot_exp:
                    prot0 = np.sum(enz_exp[prot_selection[0]])
                    prot1 = np.sum(enz_exp[prot_selection[1]])
                # set proteome value if saturation levels are plotted
                else:
                    prot0 = np.sum(enz_sat[prot_selection[0]])
                    prot1 = np.sum(enz_sat[prot_selection[1]])
                # save the relative proteome investment
                prot[num1_i][num0_i] = prot0 / (prot0 + prot1)
        # make colourmap between the colours of the nutrients
        cmap = mcolors.LinearSegmentedColormap.from_list("CustomCMap", [self.dark_colors[met_ext_index[1]], "white", self.dark_colors[met_ext_index[0]]])
        if "cmap" not in kwargs:
            kwargs["cmap"] = cmap
        if "rasterized" not in kwargs:
            kwargs["rasterized"] = True
        kwargs["vmin"] = 0
        kwargs["vmax"] = 1
        heatmap = ax.pcolormesh(ext0, ext1, prot, **kwargs)
        # limits
        ax.set_xlim(0, met_ext_max[0])
        ax.set_ylim(0, met_ext_max[1])
        return heatmap

    # Plot the zero net growth isocline
    def plot_zngi(self, dilution, met_ext, met_ext_index, met_ext_max, ax, met_ext_total=None, cons_prod_vectors=True, **kwargs):
        """
        Plot the section of the zero net growth isocline in the surface where extracellular metabolites with
        met_ext_index is varied between 0 and met_ext_max, while the remaining metabolites are fixed at prescribed values.

        :param dilution: Dilution rate of a continuous culture (positive float). Units: 1/time.
        :type dilution: float
        :param met_ext: Prescribed extracellular metabolite levels (ndarray with shape (self.ext_num,)). Units: extracellular metabolites.
        :type met_ext: numpy.ndarray
        :param met_ext_index: List/tuple of the two indices of the extracellular metabolite to be varied (list/tuple of two integers).
        :type met_ext_index: list[int]
        :param met_ext_max: List/tuple of the two maximum values of the extracellular metabolite levels to be varied (list/tuple of two positive floats). Units: extracellular metabolites.
        :type met_ext_max: list[float]
        :param ax: axes used for plotting (matplotlib.axes.Axes object).
        :type ax: matplotlib.axes.Axes
        :param met_ext_total:
            Maximal levels of extracellular metabolites in the medium (ndarray with shape (self.ext_num,) or None).
            When None, it is assumed that extracellular metabolites are measured in the units of molar fraction and
            the maximum levels of extracellular metabolites are correspondingly set to 1. Units: extracellular metabolites.
        :type met_ext_total: Union(numpy.ndarray,None)
        :param cons_prod_vectors: Boolean indicating whether to plot consumption-production vectors (True) (bool, default is True).
        :type cons_prod_vectors: bool
        :param kwargs: optional keyword arguments for the plot function.
        :returns: zngi: zero net growth isocline (matplotlib.contour.QuadContourSet object), vector_field = consumption-production vectors (optional, matplotlib.quiver.Quiver object).
        """
        # prepare axes
        num = 100
        met_ext_axis0 = np.linspace(0, met_ext_max[0], num=num)
        met_ext_axis1 = np.linspace(0, met_ext_max[1], num=num)
        ext0, ext1 = np.meshgrid(met_ext_axis0, met_ext_axis1)
        growth = np.zeros((num, num))
        # obtain growth rate for each nutrient concentration
        for num0_i in range(num):
            for num1_i in range(num):
                # set nutrients
                met_ext[met_ext_index[0]] = met_ext_axis0[num0_i]
                met_ext[met_ext_index[1]] = met_ext_axis1[num1_i]
                # find appropriate expression and saturation levels
                enz_exp, enz_sat = self.stat_prot_mean(met_ext, met_ext_total)
                # find growth rate
                growth[num1_i][num0_i] = self.growth_rate(enz_sat)
        # plot zngi
        kwargs["levels"]=[dilution]
        if "colors" not in kwargs:
            kwargs["colors"] = "black"
        zngi = ax.contour(ext0, ext1, growth, **kwargs)
        # limits
        ax.set_xlim(0, met_ext_max[0])
        ax.set_ylim(0, met_ext_max[1])
        # add 3 consumption vectors
        if cons_prod_vectors:
            # Sample points along the contour
            num_vectors = 10
            contour_paths = zngi.get_paths()
            vector_positions = []
            for path in contour_paths:
                verts = path.vertices
                indices = np.linspace(0, len(verts)-1, num_vectors, dtype=int)
                vector_positions.extend(verts[indices])
            vector_positions = np.array(vector_positions)
            ext0_v, ext1_v = vector_positions[:, 0], vector_positions[:, 1]
            vec0,vec1 = np.zeros(num_vectors), np.zeros(num_vectors)
            for num in range(num_vectors):
                # set nutrients
                met_ext[met_ext_index[0]] = ext0_v[num]
                met_ext[met_ext_index[1]] = ext1_v[num]
                # find appropriate expression and saturation levels
                enz_exp, enz_sat = self.stat_prot_mean(met_ext, met_ext_total)
                prod_cons = self.stat_production_consumption(enz_sat)
                vec0[num] = prod_cons[met_ext_index[0]]
                vec1[num] = prod_cons[met_ext_index[1]]
            # Normalize vector size for visibility
            vec0 *= 0.2 / np.max(np.sqrt(vec0**2 + vec1**2))
            vec1 *= 0.2 / np.max(np.sqrt(vec0**2 + vec1**2))
            vector_field = ax.quiver(ext0_v, ext1_v, vec0, vec1, angles='xy', scale_units='xy', scale=1, color='black')
            return zngi,vector_field
        else:
            return zngi

    # Plot Scott-Hwa law
    def plot_hwa(self, prot_selection, met_ext, met_ext_index, met_ext_max, ax, met_ext_total=None, **kwargs):
        """
        Plot the Scott-Hwa equation when extracellular metabolite with met_ext_index is varied between 0 and met_ext_max,
        while the remaining metabolites are fixed at prescribed values.

        :param prot_selection: A boolean array specifying the combination of plotted enzyme budgets (ndarray with shape (self.react_num,)).
        :type prot_selection: numpy.ndarray
        :param met_ext: Prescribed extracellular metabolite levels (ndarray with shape (self.ext_num,)). Units: extracellular metabolites.
        :type met_ext: numpy.ndarray
        :param met_ext_index: Index of the extracellular metabolite to be varied (int).
        :type met_ext_index: int
        :param met_ext_max: Maximum value of the extracellular metabolite level to be varied (positive float). Units: extracellular metabolites.
        :type met_ext_max: float
        :param met_ext_total:
            Maximal levels of extracellular metabolites in the medium (ndarray with shape (self.ext_num,) or None).
            When None, it is assumed that extracellular metabolites are measured in the units of molar fraction and
            the maximum levels of extracellular metabolites are correspondingly set to 1. Units: extracellular metabolites.
        :type met_ext_total: Union(numpy.ndarray,None)
        :param ax: axes used for plotting (matplotlib.axes.Axes object).
        :type ax: matplotlib.axes.Axes
        :param kwargs: optional keyword arguments for the plot function.
        :returns: curve: plotted Scott-Hwa curve (list of matplotlib.lines.Line2D objects).
        """
        # prepare axes
        num = 100
        ext_met_axis = np.linspace(0, met_ext_max, num=num)
        growth_axis = np.zeros(num)
        exp_axis = np.zeros(num)
        # obtain growth rate for each nutrient concentration
        for num_i in range(num):
            # set nutrient level to a given nut_axis concentration
            met_ext[met_ext_index] = ext_met_axis[num_i]
            # find appropriate expression and saturation levels
            enz_exp, enz_sat = self.stat_prot_mean(met_ext, met_ext_total)
            # find the growth rate
            growth_axis[num_i] = self.growth_rate(enz_sat)
            exp_axis[num_i] = np.sum(enz_exp[prot_selection])
        # If colour not specified, use the default colour scheme
        if "color" not in kwargs:
            kwargs["color"] = self.dark_colors[met_ext_index]
        # Make the plot
        curve = ax.plot(growth_axis, exp_axis, **kwargs)
        # Set limits
        ax.set_xlim(0, np.max(growth_axis))
        ax.set_ylim(0, self.enz_total)
        return curve

    # Plot time lag law
    def plot_lag_law(self, met_ext_pre, met_ext_pre_index, met_ext_pre_max, met_ext_post, t_max, ax, met_ext_total=None, **kwargs):
        """
        Plot the law on lag phase when extracellular metabolite with met_ext_index is varied between 0 and met_ext_max,
        while the remaining metabolites are fixed at prescribed values.

        :param met_ext_pre: Preshift prescribed extracellular metabolite levels (ndarray with shape (self.ext_num,)). Units: extracellular metabolites.
        :type met_ext_pre: numpy.ndarray
        :param met_ext_pre_index: Index of the extracellular metabolite to be varied in the preshift medium (int).
        :type met_ext_pre_index: int
        :param met_ext_pre_max: Maximum value of the extracellular metabolite level to be varied (positive float). Units: extracellular metabolites.
        :type met_ext_pre_max: float
        :param met_ext_post: Postshift prescribed extracellular metabolite levels (ndarray with shape (self.ext_num,)). Units: extracellular metabolites.
        :type met_ext_post: numpy.ndarray
        :param t_max: Maximum time of the simulation (positive float). Units: time.
        :type t_max: float
        :param ax: axes used for plotting (matplotlib.axes.Axes object).
        :type ax: matplotlib.axes.Axes
        :param met_ext_total:
            Maximal levels of extracellular metabolites in the medium (ndarray with shape (self.ext_num,) or None).
            When None, it is assumed that extracellular metabolites are measured in the units of molar fraction and
            the maximum levels of extracellular metabolites are correspondingly set to 1. Units: extracellular metabolites.
        :type met_ext_total: Union(numpy.ndarray,None)
        :param kwargs: optional keyword arguments for the plot function.
        :returns: curve: plotted lag law curve (list of matplotlib.lines.Line2D objects).
        """
        # prepare axes
        num = 25
        met_ext_axis = np.linspace(0, met_ext_pre_max, num=num)
        growth_axis = np.zeros(num)
        lag_axis = np.zeros(num)
        # obtain proteome allocation and growth after shift
        enz_exp1, enz_sat1 = self.stat_prot_mean(met_ext_post, met_ext_total)
        growth_post = self.growth_rate(enz_sat1)
        # obtain lag time for each nutrient concentration
        for num_i in range(num):
            # set the preshift extracellular metabolites to appropriate levels
            met_ext_pre[met_ext_pre_index] = met_ext_axis[num_i]
            # obtain proteome allocation before shift
            enz_exp0, enz_sat0 = self.stat_prot_mean(met_ext_pre, met_ext_total)
            # obtain the growth rate before the shift
            growth_axis[num_i] = self.growth_rate(enz_sat0)
            # solve the system up to t_max
            solution = self.dyn_prot_ode((0,t_max),enz_exp0,enz_sat0,self.met_int_steady,met_ext_post, met_ext_total)
            growth_rates = solution.growth
            lag = t_max - sp.integrate.trapezoid(growth_rates, solution.t)/growth_post
            if lag > 0:
                lag_axis[num_i] = 1/lag
            else:
                lag_axis[num_i] = 'NaN'
        # If colour not specified, use the default colour scheme
        if "color" not in kwargs:
            kwargs["color"] = self.dark_colors[met_ext_pre_index]
        # Make the plot
        curve = ax.plot(growth_axis, lag_axis, **kwargs)
        # limits
        ax.set_xlim(np.min(growth_axis)*0.9, np.max(growth_axis) * 1.1)
        ax.set_ylim(0, np.max(lag_axis) * 1.1)
        return curve

    # Plot the plane of proteome allocation
    def plot_production_consumption_vectors(self, met_ext, met_ext_index, met_ext_max, ax, met_ext_total=None, stream_lines=False, vec_num=8, **kwargs):
        """
        Plot the consumption and production vector fields on the plane with extracellular metabolites with met_ext_index
        being varied between 0 and met_ext_max.

        :param met_ext: Prescribed extracellular metabolite levels (ndarray with shape (self.ext_num,)). Units: extracellular metabolites.
        :type met_ext: numpy.ndarray
        :param met_ext_index: List/tuple of the two indices of the extracellular metabolite to be varied (list/tuple of two integers).
        :type met_ext_index: list[int]
        :param met_ext_max: List/tuple of the two maximum values of the extracellular metabolite levels to be varied (list/tuple of two positive floats). Units: extracellular metabolites.
        :type met_ext_max: list[float]
        :param ax: axes used for plotting (matplotlib.axes.Axes object).
        :type ax: matplotlib.axes.Axes
        :param met_ext_total:
            Maximal levels of extracellular metabolites in the medium (ndarray with shape (self.ext_num,) or None).
            When None, it is assumed that extracellular metabolites are measured in the units of molar fraction and
            the maximum levels of extracellular metabolites are correspondingly set to 1. Units: extracellular metabolites.
        :type met_ext_total: Union(numpy.ndarray,None)
        :param stream_lines: Boolean indicating whether to plot streamlines (True) or a vector field (False) (bool, default is False).
        :type stream_lines: bool
        :param vec_num: Number of vectors plotted per one axis.
        :type vec_num: int
        :param kwargs: optional keyword arguments for the plot function.
        :returns: vector_field: vector field of consumption-production vectors (Quiver or StreamplotSet object).
        """
        # Define a grid for extracellular metabolite levels
        num = vec_num
        met_ext_axis0 = np.linspace(0, met_ext_max[0], num=num)
        met_ext_axis1 = np.linspace(0, met_ext_max[1], num=num)

        # Initialize vector fields
        vector_field0 = np.zeros((num,num))  # x-component (production/consumption rate for metabolite 0)
        vector_field1 = np.zeros((num,num))  # y-component (for metabolite 1)

        # Loop through each grid point
        for i in range(num):
            for j in range(num):
                # Set metabolite levels at grid points
                met_ext[met_ext_index[0]] = met_ext_axis0[i]
                met_ext[met_ext_index[1]] = met_ext_axis1[j]

                # Compute the production/consumption rates
                enz_exp, enz_sat = self.stat_prot_mean(met_ext, met_ext_total)
                prod_cons = self.stat_production_consumption(enz_sat)

                # Save the rates into the vector field
                vector_field0[j,i] = prod_cons[met_ext_index[0]]
                vector_field1[j,i] = prod_cons[met_ext_index[1]]

        # Normalize vectors for consistent arrow size
        norm = np.sqrt(vector_field0**2 + vector_field1**2)
        vector_field0[norm != 0] /= norm[norm != 0]
        vector_field1[norm != 0] /= norm[norm != 0]

        # By default, make the lines black
        if "color" not in kwargs:
            kwargs["color"] = "black"

        # Plotting the results
        if stream_lines:
            # Streamline plot
            vector_field = ax.streamplot(met_ext_axis0, met_ext_axis1, vector_field0, vector_field1, **kwargs)
        else:
            if "angles" not in kwargs:
                kwargs["angles"] = "xy"
            if "scale_units" not in kwargs:
                kwargs["scale_units"] = "xy"
            # Vector field (quiver) plot
            vector_field = ax.quiver(met_ext_axis0, met_ext_axis1, vector_field0, vector_field1, **kwargs)

        # Set axis limits and labels
        ax.set_xlim(0, met_ext_max[0])
        ax.set_ylim(0, met_ext_max[1])

        return vector_field

    # Muller plot for enzyme expression and saturation
    def plot_muller_proteome(self, times, enz_exp, enz_sat, ax, **kwargs):
        """
        Make a Muller plot for enzyme expression and saturation.

        :param times: Time points (ndarray with shape (time_num,)). Units: time.
        :type times: numpy.ndarray
        :param enz_exp: Enzyme expression (ndarray with shape (self.react_num,time_num)).). Units: enzymes.
        :type enz_exp: numpy.ndarray
        :param enz_sat: Enzyme saturation (ndarray with shape (self.react_num,time_num)).). Units: enzymes.
        :type enz_sat: numpy.ndarray
        :param ax: axes used for plotting (matplotlib.axes.Axes object).
        :type ax: matplotlib.axes.Axes
        :param kwargs: keyword arguments for the plot function.
        :return: plotted polygons in the Muller plot (list of matplotlib.patches.Polygon objects).
        """

        # prepare arrays
        enz_exp = enz_exp/self.enz_total
        enz_sat = enz_sat/self.enz_total
        exp_minus_sat = enz_exp - enz_sat
        proteome = np.zeros((2*self.react_num,times.size))
        proteome[0::2] = enz_sat
        proteome[1::2] = exp_minus_sat
        # make the plot
        if "colors" not in kwargs:
            kwargs["colors"] = self.dark_light_colors
        muller = ax.stackplot(times, proteome, **kwargs)
        # limits
        ax.set_ylim(0, 1)
        ax.set_xlim(times[0], times[-1])
        # hide ticks
        ax.set_xticks([])
        ax.set_yticks([])
        return muller

    # Plot the growth rate over time
    def plot_growth_rate(self, times, growth_rate, ax, **kwargs):
        """
        Make the growth rate over time plot.

        :param times: Time points (ndarray with shape (time_num,)). Units: time.
        :type times: numpy.ndarray
        :param growth_rate: Growth rates (ndarray with shape (time_num,)).). Units: num. cell divisions/time.
        :type growth_rate: numpy.ndarray
        :param ax: axes used for plotting (matplotlib.axes.Axes object).
        :type ax: matplotlib.axes.Axes
        :param kwargs: keyword arguments for the plot function.
        :returns: plotted growth rate curve (matplotlib.lines.Line2 object).
        """

        # make the plot
        if "color" not in kwargs:
            kwargs["color"] = "black"
        line = ax.plot(times, growth_rate, **kwargs)
        # limits
        ax.set_ylim(min(0,growth_rate.min()), 1.1*growth_rate.max())
        ax.set_xlim(times[0], times[-1])
        return line

    # Plot the growth plane
    def plot_growth_plane(self, met_ext, met_ext_index, met_ext_max, ax, met_ext_total=None, contours=False, prod_cons=False, **kwargs):
        """
        Plots the growth rate on the plane of extracellular metabolites, contours of growth and consumption vectors can
        be optionally overlaid on this heatmap plot.

        :param met_ext: Prescribed enzyme expression (ndarray with shape (self.ext_num,)). Units: extracellular metabolites.
        :type met_ext: numpy.ndarray
        :param met_ext_index: List/tuple of the two indices of the extracellular metabolite to be varied (list/tuple of two integers).
        :type met_ext_index: list[int]
        :param met_ext_max: List/tuple of the two maximum values of the extracellular metabolite levels to be varied (list/tuple of two positive floats). Units: extracellular metabolites.
        :type met_ext_max: list[float]
        :param ax: Axes used for plotting (matplotlib.axes.Axes object).
        :type ax: matplotlib.axes.Axes
        :param met_ext_total:
            Maximal levels of extracellular metabolites in the medium (ndarray with shape (self.ext_num,) or None).
            When None, it is assumed that extracellular metabolites are measured in the units of molar fraction and
            the maximum levels of extracellular metabolites are correspondingly set to 1. Units: extracellular metabolites.
        :type met_ext_total: Union(numpy.ndarray,None)
        :param contours: Plot contour lines of the growth rate (default False).
        :type contours: bool
        :param prod_cons: Plot production-consumption vectors on top of the growth data (default False).
        :param kwargs: Optional keyword arguments for the plot function.
        :return: A tuple with up to three objects:
            - heatmap: proteome heatmap (matplotlib.collections.QuadMesh object).
            - contour_lines: contour lines of the growth rate (matplotlib.contour.QuadContourSet object).
            - vector_field: vector field of the production-consumption vectors (matplotlib.quiver.Quiver object).
        """
        # prepare axes
        num = 100
        met_ext_axis0 = np.linspace(0, met_ext_max[0], num=num)
        met_ext_axis1 = np.linspace(0, met_ext_max[1], num=num)
        ext0, ext1 = np.meshgrid(met_ext_axis0, met_ext_axis1)
        growth = np.zeros((num,num))
        # obtain growth rate for each nutrient concentration
        for num0_i in range(num):
            for num1_i in range(num):
                # set nutrients
                met_ext[met_ext_index[0]] = met_ext_axis0[num0_i]
                met_ext[met_ext_index[1]] = met_ext_axis1[num1_i]
                # find appropriate expression and saturation levels
                enz_exp, enz_sat = self.stat_prot_mean(met_ext, met_ext_total)
                # set proteome value if expression levels are plotted
                growth[num1_i][num0_i] = self.growth_rate(enz_sat)
        # make colourmap between the colours of the nutrients
        if "cmap" not in kwargs:
            kwargs["cmap"] = "cividis"
        if "rasterized" not in kwargs:
            kwargs["rasterized"] = True
        min_growth = np.min(growth)
        max_growth = np.max(growth)
        kwargs["vmin"] = min_growth
        kwargs["vmax"] = max_growth
        heatmap = ax.pcolormesh(ext0, ext1, growth, **kwargs)
        # limits
        ax.set_xlim(0, met_ext_max[0])
        ax.set_ylim(0, met_ext_max[1])
        # obtain the contours
        if contours:
            # Exclude extreme values by slightly narrowing the range (e.g., 10% inward)
            level_min = min_growth +  0.15 * (max_growth - min_growth)
            level_max = max_growth -  0.1 * (max_growth - min_growth)

            # Generate approximately 5 contour levels
            levels = np.linspace(level_min, level_max, 4)

            # Do the plot
            contours = ax.contour(ext0, ext1, growth, colors="white", levels=levels)
            # ax.clabel(contours, inline=True, fontsize=8)

            # Plot production consumption vectors
            if prod_cons:
                # Sample points along the contour
                contour_paths = contours.get_paths()
                num_vectors = 10
                vector_positions = []
                for path in contour_paths:
                    verts = path.vertices
                    indices = np.linspace(0, len(verts)-1, num_vectors, dtype=int)
                    vector_positions.extend(verts[indices])
                vector_positions = np.array(vector_positions)
                ext0_v, ext1_v = vector_positions[:, 0], vector_positions[:, 1]
                vec0,vec1 = np.zeros(num_vectors*len(contour_paths)), np.zeros(num_vectors*len(contour_paths))
                for num in range(num_vectors*len(contour_paths)):
                    # set nutrients
                    met_ext[met_ext_index[0]] = ext0_v[num]
                    met_ext[met_ext_index[1]] = ext1_v[num]
                    # find appropriate expression and saturation levels
                    enz_exp, enz_sat = self.stat_prot_mean(met_ext, met_ext_total)
                    prod_cons = self.stat_production_consumption(enz_sat)
                    vec0[num] = prod_cons[met_ext_index[0]]
                    vec1[num] = prod_cons[met_ext_index[1]]
                # Normalize vector size for visibility
                vec0 *= 0.2 / np.max(np.sqrt(vec0**2 + vec1**2))
                vec1 *= 0.2 / np.max(np.sqrt(vec0**2 + vec1**2))
                vectors = ax.quiver(ext0_v, ext1_v, vec0, vec1, angles='xy', scale_units='xy', scale=1, color='white')
                return heatmap, contours, vectors
            else:
                return heatmap, contours
        elif prod_cons:
            vectors = self.plot_production_consumption_vectors(met_ext, met_ext_index, met_ext_max, ax, met_ext_total)
            return heatmap, contours, vectors
        else:
            return heatmap

####
# CULTURE CLASS
####
# Keeps record of all nutrients and ideal microbes present in the culture

class Culture:
    ##
    # INITIALIZATION FUNCTIONS
    ##

    # Initialization function
    def __init__(self, microbes, dilution, met_ext_input, met_ext_total=None, mic_units=None, met_ext_units=None):
        """
        Initialize a culture with a list of microbes, dilution rate and imported extracellular metabolite levels.

        Defines class variables of the same name and type as input, as well as additional class variables:
            - mic_num: number of microbes present in the culture (int).
            - ext_num: number of extracellular metabolites present in the culture (int).
            - cmr_slicing_num: indices used in the slicing of states in the consumer-metabolism-resource model (list of 3+self.mic_num integers).
            - microbe_colors: list of colors for plotting microbes (list of rgb tuples).
            - dark_colors: list of darker colors for plotting extracellular metabolites (list of rgb tuples).
            - light_colors: list of lighter colors for plotting extracellular metabolites (list of rgb tuples).
        :param microbes:
            List of Microbes that interact in the culture (list of Microbe objects).
        :type microbes: list[Microbe]
        :param dilution:
            Flow rate of the culture (positive float). For batch culture this is 0. Units: 1/time.
        :type dilution: float
        :param met_ext_input:
            Levels of extracellular metabolites in the inflow medium of a continuous culture (ndarray with shape (ext_num,)).
            In a batch culture, these parameters can be arbitrary as there is no flow from this medium.
            Units: extracellular metabolites (natural), mM (fba).
        :type met_ext_input: numpy.ndarray
        :param met_ext_total:
            Total number of extracellular metabolite molecules in the medium mixture, specified in the appropriate units
            for each extracellular metabolite (ndarray with shape (ext_num,) or None).
            If met_ext_total is None, it is assumed that extracellular metabolites are measured in the units of molar fraction
            and the total of extracellular metabolites is correspondingly set to 1 for every extracellular metabolite.
            FBA-like approach: Set each element of the vector to the total number of H20 molecules measured in mmol per 1 liter of water.
            Units: extracellular metabolites.
        :param mic_units:
            Specified conversion coefficients for the units of different microbes (1D ndarray, positive floats, default: None).
            If mic_units=None, the default values are set to 1.
            Each entry specifies how many units correspond to one cell (e.g. grams of cell dry weight for a single cell).
            When the units are relative to a volume, the same volume should be used for microbial units and units of extracellular metabolites.
            For example, g/l for microbes and mM=mmol/l for extracellular metabolites are consistent unit systems.
            FBA-like approach: Use the default value.
            Units: e.g. g (natural), dimensionless parameter (fba).
        :type mic_units: Union[numpy.ndarray, None]
        :param met_ext_units:
            Specified conversion coefficients for the units of different nutrients i (1D ndarray, positive floats, default: None).
            If met_ext_units=None, the default values are set to 1.
            Each entry specifies how many units correspond to one molecule (e.g. how many mmol is a single molecule).
            When the units are relative to a volume, the same volume should be used for microbial units and units of extracellular metabolites.
            For example, g/l for microbes and mM=mmol/l for extracellular metabolites are consistent unit systems.
            FBA-like approach: Use the default value.
            Units: e.g. mmol (natural), dimensionless parameter (fba).
        :type met_ext_units: Union[numpy.ndarray, None]
        """
        # Save input parameters
        self.microbes = microbes
        self.dilution = dilution
        self.met_ext_input = met_ext_input
        # Save number of microbes and extracellular metabolites
        self.mic_num = len(microbes)
        self.ext_num = met_ext_input.size
        # Save the units
        if mic_units is None:
            mic_units = np.ones(self.mic_num)
        self.mic_units = mic_units
        if met_ext_units is None:
            met_ext_units = np.ones(self.ext_num)
        self.met_ext_units = met_ext_units
        # Save the total number of extracellular metabolites
        if met_ext_total is None:
            met_ext_total = np.ones(self.ext_num)
        self.met_ext_total = met_ext_total
        # Save indices used in the slicing of states in the consumer-metabolism-resource model
        self.cmr_slicing_num = [0, self.mic_num, self.mic_num+self.ext_num]
        for mic in range(self.mic_num):
            prot_num = 2*self.microbes[mic].react_num+self.microbes[mic].int_num
            self.cmr_slicing_num.append(self.cmr_slicing_num[-1]+prot_num)
        # Create colour schemes
        self.microbe_colors = [(int(240/self.mic_num*mic), int(240/self.mic_num*mic), int(240/self.mic_num*mic)) for mic in range(self.mic_num)]
        self.dark_colors, self.light_colors = generate_colors(self.ext_num)

    ##
    # CONSUMER-RESOURCE MODEL
    ##

    # Slice the state of the consumer-resource model into mic_level, met_ext (needed for ODE solver)
    def slice_cr_state(self, cr_state):
        """
        Slices the given state of the consumer-resource model (resp. its forcing) into two parts: mic_level and met_ext.

        :param cr_state: State of the consumer-resource model (resp. its forcing) (ndarray with shape (self.mic_num+self.ext_num,)).
        :type cr_state: numpy.ndarray
        :returns: A tuple containing two numpy arrays:
            - mic_level: Levels of microbes (ndarray with shape (self.mic_num,)). Units: microbes (natural), g/l (fba).
            - met_ext: Levels of extracellular metabolites (ndarray with shape (self.ext_num,)). Units: extracellular metabolites (natural), mM (fba).
        """
        return cr_state[:self.mic_num], cr_state[self.mic_num:]

    # Augment mic_level, met_ext into a single cr_state (needed for ODE solver)
    def augment_cr_state(self, mic_level, met_ext):
        """
        Combines levels of microbes and extracellular metabolites into a single state of the consumer-resource model.
        Works for the corresponding forcing as well.

        :param mic_level: Levels of microbes (ndarray with shape (self.mic_num,)). Units: microbes (natural), g/l (fba).
        :type mic_level: numpy.ndarray
        :param met_ext: Levels of extracellular metabolites (ndarray with shape (self.ext_num,)). Units: extracellular metabolites (natural), mM (fba).
        :type met_ext: numpy.ndarray
        """
        return np.concatenate((mic_level, met_ext))

    # Forcing function of the consumer-resource model
    def cr_forcing(self, mic_level, met_ext):
        """
        Calculate the forcing function in the consumer-resource model.

        :param mic_level: Levels of microbes (ndarray with shape (self.mic_num,)). Units: microbes (natural), g/l (fba).
        :type mic_level: numpy.ndarray
        :param met_ext: Levels of extracellular metabolites (ndarray with shape (self.ext_num,)). Units: extracellular metabolites (natural), mM (fba).
        :type met_ext: numpy.ndarray
        :return: A tuple containing two numpy arrays:
            - mic_force: The forcing on microbes (ndarray with shape (self.mic_num,)). Units: microbes/time (natural), g/l/hr (fba).
            - ext_force: The forcing on extracellular metabolites (ndarray with shape (self.ext_num,)). Units: extracellular metabolites/time (natural), mM/hr (fba).
        """
        mic_forcing = mic_level.copy()
        ext_forcing = self.dilution*(self.met_ext_input-met_ext)
        for mic in range(self.mic_num):
            growth, prod_cons = self.microbes[mic].cr_forcing(met_ext, self.met_ext_total)
            mic_forcing[mic] *= growth-self.dilution
            ext_forcing += mic_level[mic]/self.mic_units[mic]*prod_cons*self.met_ext_units
        return mic_forcing, ext_forcing

    # ODE solver for the dynamic proteome allocation model
    def cr_model_ode(self, t_span, mic_level0, met_ext0, **kwargs):
        """
        Solves the ODE system for the consumer-resource model.

        :param t_span: Tuple (start_time, end_time) specifying the time span for ODE integration. Units: time.
        :type t_span: tuple
        :param mic_level0: Initial levels of microbes (ndarray with shape (self.mic_num,)). Units: microbes (natural), g/l (fba).
        :type mic_level0: numpy.ndarray
        :param met_ext0: Initial levels of extracellular metabolites (ndarray with shape (self.ext_num,)). Units: extracellular metabolites (natural), mM (fba).
        :type met_ext0: numpy.ndarray
        :returns: The solution to the ODE system as an `OdeResult` object.
        """
        # Augment initial values into a single prot_state
        cr_state0 = self.augment_cr_state(mic_level0, met_ext0)

        # Define the ODE system as a function
        def ode_system(t, cr_state):
            mic_level, met_ext = self.slice_cr_state(cr_state)
            mic_force, ext_force = self.cr_forcing(mic_level, met_ext)
            return self.augment_cr_state(mic_force, ext_force)

        # Set up the solver
        kwargs["fun"]=ode_system
        kwargs["t_span"]=t_span
        kwargs["y0"]=cr_state0
        if "method" not in kwargs:
            kwargs["method"]="RK45"

        # Solve the ODE system using solve_ivp
        solution = sp.integrate.solve_ivp(**kwargs)

        return solution

    # Slice the solution of the ODE solver into mic_level, met_ext for all time points
    def slice_cr_solution(self, solution):
        """
        Slices the outuput solution of the cr_model_ode() function into two parts: mic_level and met_ext.

        :param solution:
            The output solution of the cr_model_ode() function (OdeResult object)
        :type solution: scipy.integrate.OdeResult
        :returns: A tuple containing two numpy arrays:
            - mic_level: Levels of microbes (ndarray with shape (self.mic_num, t_point_num)). Units: microbes (natural), g/l (fba).
            - met_ext: Levels of extracellular metabolites (ndarray with shape (self.ext_num, t_point_num)). Units: extracellular metabolites (natural), mM (fba).
        """
        cr_state = solution.y
        return cr_state[:self.mic_num,:], cr_state[self.mic_num:,:]

    # Find the quasi-stationary proteome allocation along of a given microbe along the solution
    def proteome_cr_solution(self, mic_index, solution):
        """
        Calculate the quasi-stationary proteome allocation of a given microbe along the solution trajectory.

        :param mic_index: Index of the microbe whose quasi-stationary proteome allocation is to be calculated (int).
        :type mic_index: int
        :param solution: Solution of the consumer-resource model (OdeResult object).
        :type solution: scipy.integrate.OdeResult
        :return: a tuple with two vectors:
            - enz_exp: Enzyme expression (ndarray with shape (self.react_num,t_point_num)). Units: enzymes.
            - enz_sat: Enzyme saturation (ndarray with shape (self.react_num,t_point_num)). Units: enzymes.
        """
        # Time points and number of reactions
        t_point_num = len(solution.t)  # Number of time points in the solution
        enz_exp = np.zeros((t_point_num, self.microbes[mic_index].react_num))  # Initialize enzyme expression
        enz_sat = np.zeros((t_point_num, self.microbes[mic_index].react_num))  # Initialize enzyme saturation

        # Matrix with metabolite abundances, transpose to have entries [time t][metabolite a]
        mic_level, met_ext = self.slice_cr_solution(solution)
        met_ext = met_ext.T
        # Iterate over all time points in the solution
        for t in range(t_point_num):
            # Get the enzyme expression and saturation levels
            enz_exp[t], enz_sat[t] = self.microbes[mic_index].stat_prot_mean(met_ext[t], self.met_ext_total)

        # Transpose the outputs
        enz_exp = enz_exp.T
        enz_sat = enz_sat.T

        # Return the results as a tuple
        return enz_exp, enz_sat


    ##
    # CONSUMER-METABOLISM-RESOURCE MODEL
    ##

    # Forcing function of the consumer-metabolism-resource model
    def cmr_forcing(self, cmr_state):
        """
        Calculate the forcing function in the consumer-metabolism-resource model. Can only be used in the natural units.

        :param cmr_state: State of the consumer-metabolism-resource model (ndarray with shape (self.cmr_slicing_num[-1],)).
        :type cmr_state: numpy.ndarray
        :return: A forcing of the cmr_state (ndarray with shape (self.cmr_slicing_num[-1],)).
        """
        # Slice the cmr_state into mic_level, met_ext and prot_states for each microbe
        sliced_state = tuple(cmr_state[self.cmr_slicing_num[i]:self.cmr_slicing_num[i+1]] for i in range(2+self.mic_num))
        # Prepare the forcing vector
        cmr_force = np.zeros(self.cmr_slicing_num[-1])
        # Dilution and import of nutrients
        cmr_force[self.cmr_slicing_num[1]:self.cmr_slicing_num[2]] = self.dilution*(self.met_ext_input-sliced_state[1])
        # Go over all microbes
        for mic in range(self.mic_num):
            # Identify the prot_state for the current microbe
            prot_state = sliced_state[2+mic]
            # Growth rate, production/consumption rates and proteome forcing
            growth, prod_cons, prot_force = self.microbes[mic].cmr_forcing(prot_state,sliced_state[1],self.met_ext_total)
            # Forcing of microbes
            cmr_force[self.cmr_slicing_num[0]:self.cmr_slicing_num[1]] = (growth-self.dilution)*sliced_state[0][mic]
            # Consumption of nutrients by the current microbe
            cmr_force[self.cmr_slicing_num[1]:self.cmr_slicing_num[2]] += sliced_state[0][mic]*(prod_cons*self.met_ext_units)/self.mic_units[mic]
            # Forcing of proteome state in the current microbe
            cmr_force[self.cmr_slicing_num[mic+2]:self.cmr_slicing_num[mic+3]] = prot_force
        return cmr_force

    # ODE solver for the dynamic proteome allocation model
    def cmr_model_ode(self, t_span, mic_level0, met_ext0, enz_exp0, enz_sat0, met_int0, **kwargs):
        """
        Solves the ODE system for the consumer-metabolism-resource model. Can only be used in the natural units.

        :param t_span: Tuple (start_time, end_time) specifying the time span for ODE integration. Units: time.
        :type t_span: tuple
        :param mic_level0: Initial levels of microbes (ndarray with shape (self.mic_num,)). Units: microbes.
        :type mic_level0: numpy.ndarray
        :param met_ext0: Initial levels of extracellular metabolites (ndarray with shape (self.ext_num,)). Units: extracellular metabolites.
        :type met_ext0: numpy.ndarray
        :param enz_exp0: Initial levels of enzyme expression for each microbe (list of ndarray with shape (react_num,)). Units: enzymes.
        :type enz_exp0: list[numpy.ndarray]
        :param enz_sat0: Initial levels of enzyme saturation for each microbe (list of ndarray with shape (react_num,)). Units: enzymes.
        :type enz_sat0: list[numpy.ndarray]
        :param met_int0: Initial levels of internal metabolites for each microbe (list of ndarray with shape (int_num,)). Units: intracellular metabolites.
        :type met_int0: list[numpy.ndarray]
        :returns: The solution to the ODE system as an `OdeResult` object.
        """
        # Augment initial values into a single prot_state
        prot_states = [np.concatenate([enz_exp0[mic],enz_sat0[mic],met_int0[mic]]) for mic in range(self.mic_num)]
        cmr_state0 = np.concatenate((mic_level0,met_ext0,*prot_states))

        # Define the ODE system as a function
        def ode_system(t, cr_state):
            return self.cmr_forcing(cr_state)

        # Set up the solver
        kwargs["fun"]=ode_system
        kwargs["t_span"]=t_span
        kwargs["y0"]=cmr_state0
        if "method" not in kwargs:
            kwargs["method"]="RK45"

        # Solve the ODE system using solve_ivp
        solution = sp.integrate.solve_ivp(**kwargs)

        return solution

    # Slice the solution of the ODE solver into mic_level, met_ext, enz_exp, enz_sat, met_int for all time points
    def slice_cmr_solution(self, solution):
        """
        Slices the outuput solution of the cmr_model_ode() function into: mic_level, met_ext, enz_exp, enz_sat, met_int.
        Can only be used in the natural units.

        :param solution:
            The output solution of the cmr_model_ode() function (OdeResult object)
        :type solution: scipy.integrate.OdeResult
        :returns: A tuple containing two numpy arrays:
            - mic_level: Levels of microbes (ndarray with shape (self.mic_num, t_point_num)). Units: microbes.
            - met_ext: Levels of extracellular metabolites (ndarray with shape (self.ext_num, t_point_num)). Units: extracellular metabolites.
            - enz_exp: Enzyme expression for each microbe (list of ndarray with shape (self.react_num, t_point_num)). Units: enzymes.
            - enz_sat: Enzyme saturation for each microbe (list of ndarray with shape (self.react_num, t_point_num)). Units: enzymes.
            - met_int: Internal metabolites levels for each microbe (list of ndarray with shape (self.int_num, t_point_num)). Units: intracellular metabolites.
        """
        cmr_state = solution.y
        mic_level = cmr_state[self.cmr_slicing_num[0]:self.cmr_slicing_num[1],:]
        met_ext = cmr_state[self.cmr_slicing_num[1]:self.cmr_slicing_num[2],:]
        enz_exp = []
        enz_sat = []
        met_int = []
        for mic in range(self.mic_num):
            prot_state = cmr_state[self.cmr_slicing_num[2+mic]:self.cmr_slicing_num[3+mic],:]
            enz_exp.append(prot_state[:self.microbes[mic].react_num, :])
            enz_sat.append(prot_state[self.microbes[mic].react_num:2*self.microbes[mic].react_num, :])
            met_int.append(prot_state[2*self.microbes[mic].react_num:, :])
        return mic_level, met_ext, enz_exp, enz_sat, met_int

    ##
    # PLOTTING
    ##

    # Plot the time-evolution of microbial abundances
    def plot_microbes(self, times, mic_level, ax, **kwargs):
        """
        Plot the time-evolution of microbial abundances.

        :param times: A vector of time points (ndarray with shape (time_num,)). Units: time (natural), hours (fba).
        :type times: numpy.ndarray
        :param mic_level: A vector of selected microbial abundances (ndarray with shape (<=self.mic_num, time_num)). Units: microbes (natural), g/l (fba).
        :type mic_level: numpy.ndarray
        :param ax: axes used for plotting (matplotlib.axes.Axes object).
        :type ax: matplotlib.axes.Axes
        :param kwargs: Optional keyword arguments for the plot function.
        :return: Plotted lines (list of matplotlib.lines.Line2 object).
        """
        # plot each microbe in the list of mic_indices
        lines = []
        for mic_i in range(mic_level.shape[0]):
            if 'color' not in kwargs:
                lines.append(ax.plot(times, mic_level[mic_i,:], color=self.microbe_colors[mic_i], **kwargs))
            else:
                lines.append(ax.plot(times, mic_level[mic_i,:], **kwargs))
        # Set axis limits
        ax.set_ylim(0, np.max(mic_level) * 1.1)
        ax.set_xlim(times[0], times[-1])
        return lines

    # Plot the time-evolution of extracellular metabolites
    def plot_extracellular_metabolites(self, times, met_ext, ax, **kwargs):
        """
        Plot the time-evolution of extracellular metabolites.

        :param times: A vector of time points (ndarray with shape (time_num,)). Units: time (natural), hours (fba).
        :type times: numpy.ndarray
        :param met_ext: A vector of selected extracellular metabolites (ndarray with shape (<=self.met_ext, time_num)). Units: extracellular metabolites (natural), mM (fba).
        :type met_ext: numpy.ndarray
        :param ax: axes used for plotting (matplotlib.axes.Axes object).
        :type ax: matplotlib.axes.Axes
        :param kwargs: Optional keyword arguments for the plot function.
        :return: Plotted lines (list of matplotlib.lines.Line2 object).
        """
        # plot each microbe in the list of mic_indices
        lines = []
        for met_i in range(met_ext.shape[0]):
            if 'color' not in kwargs:
                lines.append(ax.plot(times, met_ext[met_i, :], color=self.dark_colors[met_i], **kwargs))
            else:
                lines.append(ax.plot(times, met_ext[met_i,:], **kwargs))
        # Set axis limits
        ax.set_ylim(0, np.max(met_ext) * 1.1)
        ax.set_xlim(times[0], times[-1])
        return lines

    # Plot the trajectory of two selected extracellular metabolites
    def plot_nut_traj(self, met_ext, met_ext_indices, ax, **kwargs):
        """
        Plot the trajectory of two selected extracellular metabolites.

        :param met_ext: A vector of extracellular metabolites (ndarray with shape (self.met_ext, time_num)). Units: extracellular metabolites (natural), mM (fba).
        :type met_ext: numpy.ndarray
        :param met_ext_indices: Tuple/list of two indices of the two selected extracellular metabolites (tuple of two integers).
        :type met_ext_indices: Union(tuple[int],list[int])
        :param ax: axes used for plotting (matplotlib.axes.Axes object).
        :type ax: matplotlib.axes.Axes
        :param kwargs: Optional keyword arguments for the plot function.
        :return: Plotted line (matplotlib.lines.Line2 object).
        """
        # plot trajectory
        if "color" not in kwargs:
            kwargs["color"] = "black"
        line = ax.plot(met_ext[met_ext_indices[0],:], met_ext[met_ext_indices[1],:], **kwargs)
        return line

