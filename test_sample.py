import utils as uti


def test_fname_id():
    fname = '/home/sdf/78.tif'
    # assert 4 == 5
    assert 78 == uti.fname_to_id(fname)
