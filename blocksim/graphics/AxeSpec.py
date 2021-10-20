class AxeSpec(object):
    """Class that provides a description of an axe, without data.
    It the lines to be drawn
    (with the name of the variables instead of a concrete set of data)

    Args:
      props
        A dictionary. Supported keys :

        * nrow for the number of rows subdivisions
        * ncol for the number of columns subdivisions
        * ind for the number of axe (1 is the first one in the layout)
        * sharex is the numer of an axe whose X axe will be shared with the instance of :class:`AxeSpec`
        * title for the title of the axe
      lines
        List of dict to specify the lines' spec. Supported keys :

        * the matplotlib keyword arguments of the funcion *plot*
        * varx for the name of the X variable
        * vary for the name of the y variable

    """

    def __init__(self, props, lines):
        self.props = props
        self.lines = lines

    def __repr__(self, ntabs=0):
        st = " " * ntabs
        s = ""
        s += st + 10 * "=" + " Axe '%s' " % self.props["title"] + 10 * "=" + "\n"
        kys = list(self.props.keys())
        kys.sort()
        for k in kys:
            if k == "title":
                continue
            s += st + "%s:\t'%s'\n" % (k, self.props[k])

        for k, l in enumerate(self.lines):
            s += st + 10 * "-" + " Line #%i " % (k + 1) + 10 * "-" + "\n"
            kys = list(l.keys())
            kys.sort()
            for k in kys:
                s += 2 * st + "%s:\t'%s'\n" % (k, l[k])

        return s
