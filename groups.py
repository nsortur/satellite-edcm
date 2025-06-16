from escnn import gspaces
import escnn

class GSpaceInfo():
    def __init__(self, group_act: str):
        if group_act == 'c2c2c2':
            gd2 = escnn.group.cyclic_group(2)
            self.group = gspaces.no_base_space(escnn.group.direct_product(escnn.group.direct_product(gd2, gd2), gd2))
            self.input_reps = [escnn.group.directsum([self.group.irrep(((0,), (1,), 0),(0,)),
                                                                    self.group.irrep(((0,), (0,), 0),(1,)),
                                                                    self.group.irrep(((1,), (0,), 0),(0,))])]
        elif group_act == 'c2c2':
            gd2 = escnn.group.cyclic_group(2)
            self.group = gspaces.no_base_space(escnn.group.direct_product(gd2, gd2))
            self.input_reps = 1*[(self.group.irrep((0,), (1,), 0) + self.group.irrep((1,),(0,), 0))] + 1*[self.group.trivial_repr]
        elif group_act == 'c2':
            self.group = gspaces.no_base_space(escnn.group.cyclic_group(2))
            self.input_reps = 1*[self.group.trivial_repr] + 1*[self.group.irrep(1)] + 1*[self.group.trivial_repr]
        elif group_act == 'trivial':
            self.group = gspaces.no_base_space(escnn.group.cyclic_group(1))
            self.input_reps = 3*[self.group.trivial_repr]
        elif group_act == 'c4':
            self.group = escnn.group.cyclic_group(4)
            raise NotImplementedError()
        elif group_act == 'oh':
            self.group = gspaces.no_base_space(escnn.group.octa_group())
            self.input_reps = 1*[self.group.fibergroup.standard_representation]
        else:
            raise NotImplementedError(f"Group {group_act} not found")
