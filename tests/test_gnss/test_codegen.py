import pickle
import bz2

from numpy.testing import assert_allclose

from blocksim.gnss.GNSScodegen import GNSScodegen


def test_codegen_gps():
    # Loading reference data from original matlab
    with bz2.BZ2File("tests/test_gnss/data.pbz2", "rb") as f:
        data = pickle.load(f)

    # Verifying results
    for mod in [
        "L1CA",
        "L5I",
        "L5Q",
        "L2CM",
        "L2CL",
    ]:
        for sv in range(1, 31, 5):
            seq_ref = data[sv, mod]
            seq = GNSScodegen(sv, mod)
            assert_allclose(seq, seq_ref)


def test_codegen_galileo():
    # Loading reference data from original matlab
    with bz2.BZ2File("tests/test_gnss/data.pbz2", "rb") as f:
        data = pickle.load(f)

    # Verifying results
    for mod in [
        "E5aI",
        "E5aQ",
        "E5bI",
        "E5bQ",
        "E6-B",
        "E6-C",
        "E1B",
        "E1C",
    ]:
        for sv in range(1, 31, 5):
            seq_ref = data[sv, mod]
            seq = GNSScodegen(sv, mod)
            assert_allclose(seq, seq_ref)


def test_codegen_beidou():
    # Loading reference data from original matlab
    with bz2.BZ2File("tests/test_gnss/data.pbz2", "rb") as f:
        data = pickle.load(f)

    # Verifying results
    for mod in [
        "B1I",
    ]:
        for sv in range(1, 31, 5):
            seq_ref = data[sv, mod]
            seq = GNSScodegen(sv, mod)
            assert_allclose(seq, seq_ref)


if __name__ == "__main__":
    test_codegen_gps()
