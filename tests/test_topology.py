"""T1: CGTopology。"""
import pytest

from socg.topology import (N_RES_CLASSES, N_SEP_BUCKETS, RES_CLASS_GLY,
                           RES_CLASS_OTHER, RES_CLASS_PRO, CGTopology)


def test_ala10_frozen():
    topo = CGTopology.from_sequence("AAAAAAAAAA")
    assert topo.n == 10
    expected = [True] + [False] * 8 + [True]
    assert list(topo.frozen_mask) == expected


def test_cln025_shapes():
    topo = CGTopology.from_sequence("YYDPETGTWY")
    assert topo.bonds.shape == (9, 2)
    assert topo.angles.shape == (8, 3)
    assert topo.dihedrals.shape == (7, 4)
    assert topo.pairs.shape == (28, 2)
    bucket = topo.pair_bucket
    assert int((bucket == 0).sum()) == 7
    assert int((bucket == 1).sum()) == 6
    assert int((bucket == 2).sum()) == 15
    assert N_SEP_BUCKETS == 3


def test_res_class():
    topo = CGTopology.from_sequence("GPAG")
    assert list(topo.res_class) == [RES_CLASS_GLY, RES_CLASS_PRO, RES_CLASS_OTHER, RES_CLASS_GLY]
    assert N_RES_CLASSES == 3


def test_unknown_letter_raises():
    with pytest.raises(ValueError):
        CGTopology.from_sequence("AAXA")


def test_too_short_raises():
    with pytest.raises(ValueError):
        CGTopology.from_sequence("AAA")


def test_frozen_length_mismatch_raises():
    with pytest.raises(ValueError):
        CGTopology(sequence="AAAA", frozen=(True, False))


def test_capped_no_frozen():
    topo = CGTopology.from_sequence("AAAAAAAAAA", capped=True)
    assert not topo.frozen_mask.any()


def test_json_roundtrip():
    topo = CGTopology.from_sequence("YYDPETGTWY")
    topo2 = CGTopology.from_json(topo.to_json())
    assert topo2 == topo
    assert (topo2.pairs == topo.pairs).all()
    assert (topo2.pair_bucket == topo.pair_bucket).all()
