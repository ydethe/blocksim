import numpy as np
import scipy.linalg as lin


class WrongDataType(Exception):
    def __init__(self, txt):
        self.txt = txt

    def __str__(self):
        return self.txt


class IncompatibleShapes(Exception):
    def __init__(self, src_name: str, src_shape, dst_name: str, dst_shape):
        self.src_name = src_name
        self.src_shape = src_shape
        self.dst_name = dst_name
        self.dst_shape = dst_shape

    def __str__(self):
        return "%s[%s] not incompatible of %s[%s]" % (
            self.src_name,
            self.src_shape,
            self.dst_name,
            self.dst_shape,
        )


class InvalidLogFile(Exception):
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return "Log file '%s' invalid" % self.name


class NameIsPythonKeyword(Exception):
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return "When logging the variable '%s' : this is a python keyword" % self.name


class UnsetCommLink(Exception):
    pass


class IncorrectDataframe(Exception):
    def __init__(self, buf):
        self.buf = buf

    def __str__(self):
        return "Incorrect dataframe : '%s'" % self.buf.decode()


class FailedAcknowledgeCheck(Exception):
    def __init__(self, ack, tst_ack):
        self.ack = ack
        self.tst_ack = tst_ack

    def __str__(self):
        return "Ack received: %s. Expected: %s." % (str(self.tst_ack), str(self.ack))


class TooWeakAcceleration(Exception):
    def __init__(self, elem_name, acc):
        self.elem_name = elem_name
        self.acc = acc

    def __str__(self):
        return (
            "For the element '%s', the measured acceleration is too weak : %s / norm = %g"
            % (self.elem_name, self.acc, lin.norm(self.acc))
        )


class TooWeakMagneticField(Exception):
    def __init__(self, elem_name, acc):
        self.elem_name = elem_name
        self.acc = acc

    def __str__(self):
        return (
            "For the element '%s', the measured magnetic field is too weak : %s / norm = %g"
            % (self.elem_name, self.acc, lin.norm(self.acc))
        )


class SimulationGraphError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class DuplicateElement(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class DuplicateStateOutputName(Exception):
    def __init__(self, elem_name, name):
        self.elem_name = elem_name
        self.name = name

    def __str__(self):
        return (
            "For the element '%s', the name '%s' appears both in state and output name"
            % (self.elem_name, self.name)
        )


class DuplicateStateName(Exception):
    def __init__(self, state_name, ex_el, elem):
        self.state_name = state_name
        self.ex_el = ex_el
        self.elem = elem

    def __str__(self):
        return "When adding '%s' : state name '%s' already used in '%s'" % (
            self.elem.getName(),
            self.state_name,
            self.ex_el.getName(),
        )


class InvalidAssignedVector(Exception):
    def __init__(self, txt):
        self.txt = txt

    def __str__(self):
        return self.txt


class DuplicateOutputName(Exception):
    def __init__(self, output_name, ex_el, elem):
        self.output_name = output_name
        self.ex_el = ex_el
        self.elem = elem

    def __str__(self):
        return "When adding '%s' : output name '%s' already used in '%s'" % (
            self.elem.getName(),
            self.output_name,
            self.ex_el.getName(),
        )


class FileError(Exception):
    def __init__(self, fic):
        self.fic = fic

    def __str__(self):
        return "The file '%s' does not exist" % (self.fic,)


class CyclicGraph(Exception):
    def __init__(self, cycles):
        self.cycles = cycles

    def __str__(self):
        return (
            "The simulation graph has a cycle even when controllers are removed : %s"
            % self.cycles
        )


class DuplicateInput(Exception):
    def __init__(self, elem_name, input_name):
        self.input_name = input_name
        self.elem_name = elem_name

    def __str__(self):
        return "The new input '%s' already exists in element '%s'" % (
            self.input_name,
            self.elem_name,
        )


class DuplicateOutput(Exception):
    def __init__(self, elem_name, output_name):
        self.output_name = output_name
        self.elem_name = elem_name

    def __str__(self):
        return "The new output '%s' already exists in element '%s'" % (
            self.output_name,
            self.elem_name,
        )


class InvalidSrcDataName(Exception):
    def __init__(self, src_data_name, valid_values):
        self.src_data_name = src_data_name
        self.valid_values = valid_values

    def __str__(self):
        return "The data type does not exist : '%s'. Must be one of %s" % (
            self.src_data_name,
            self.valid_values,
        )


class InvalidInputName(Exception):
    def __init__(self, input_name):
        self.input_name = input_name

    def __str__(self):
        return "The name '%s' is invalid for an input" % self.input_name


class VectorLengthIncoherence(Exception):
    def __init__(self, elem_name, nvec, nname):
        self.elem_name = elem_name
        self.nvec = nvec
        self.nname = nname

    def __str__(self):
        return "Length incoherence for '%s' vector length %i vs names number %i" % (
            self.elem_name,
            self.nvec,
            self.nname,
        )


class UnknownInput(Exception):
    def __init__(self, elem_name, input_name):
        self.elem_name = elem_name
        self.input_name = input_name

    def __str__(self):
        return "For element '%s' : the input '%s' does not exist" % (
            self.elem_name,
            self.input_name,
        )


class UnknownState(Exception):
    def __init__(self, elem_name, state_name):
        self.elem_name = elem_name
        self.state_name = state_name

    def __str__(self):
        return "For element '%s' : the state '%s' does not exist" % (
            self.elem_name,
            self.state_name,
        )


class UnknownOutput(Exception):
    def __init__(self, elem_name, output_name):
        self.elem_name = elem_name
        self.output_name = output_name

    def __str__(self):
        return "For element '%s' : the output '%s' does not exist" % (
            self.elem_name,
            self.output_name,
        )


class DenormalizedQuaternion(Exception):
    def __init__(self, elem_name, q):
        self.elem_name = elem_name
        self.q = q

    def __str__(self):
        return (
            "For element '%s' : the attitude quaternion '%s' has a norm different from 1 : %f"
            % (self.elem_name, self.q, np.sum(self.q ** 2))
        )


class UnorderedDict(Exception):
    def __init__(self, cls):
        self.cls = cls

    def __str__(self):
        return "In %s, data passed is not a OrderedDict instance" % self.cls.__name__
