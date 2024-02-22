"""
Base or Mother class containing common attributes and functions
"""

class CLASSBase:
    def __init__(self):
        pass 

   
    def get_children_strings(self):
        list_of_strings = []
        out = dict()
        for attr_name in dir(self):
            if attr_name not in dir(CLASSBase):
                attr = getattr(self, attr_name)
                if hasattr(attr, 'get_children_strings'):
                    list_of_strings.extend(["." + attr_name + child_string for child_string in attr.get_children_strings()])
                else:
                    list_of_strings.append("" + attr_name + " = " + str(attr))
                    out["./" + attr_name] = attr
        return  list_of_strings 

    def get_attr(self):
        out = dict()
        for attr_name in dir(self):
            if attr_name not in dir(CLASSBase()):
                attr = getattr(self, attr_name)
                out[attr_name] = attr
        return  out 

    def print_att(self):
        out = self.get_attr()
        print("Attributes of ", self.__class__.__name__ , "\n   >>   ", out)