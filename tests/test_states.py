"""T2: 态指派（核心集 + TBA）。"""
import numpy as np
import pytest

from socg.states import (FROZEN_STATE, N_STATES, STATE_NAMES,
                         assign_states, core_state, nearest_state)


def _frame(*pairs):
    phi = np.asarray([[p[0] for p in pairs]], dtype=np.float64)
    psi = np.asarray([[p[1] for p in pairs]], dtype=np.float64)
    return phi, psi


def test_core_centers():
    phi, psi = _frame((-63, -43), (-120, 130), (-70, 145), (60, 45), (180, 0))
    out = core_state(phi, psi)[0]
    assert list(out) == [0, 1, 1, 2, -1]


def test_periodicity():
    # (-70, -178)：Δψ 相对 (-70,145) 折回后为 37，应在 B 核心内
    phi, psi = _frame((-70, -178))
    assert core_state(phi, psi)[0, 0] == 1


def test_hysteresis_tba():
    seq = [(-63, -43), (-63, -43), (-100, -100), (-120, 130), (-100, -100)]
    phi = np.asarray([[p[0]] for p in seq], dtype=np.float64)
    psi = np.asarray([[p[1]] for p in seq], dtype=np.float64)
    frozen = np.array([False])
    out = assign_states(phi, psi, frozen)
    assert list(out[:, 0]) == [0, 0, 0, 1, 1]


def test_first_frame_outside_core_uses_nearest():
    phi, psi = _frame((-100, -100))
    frozen = np.array([False])
    out = assign_states(phi, psi, frozen)
    assert out[0, 0] == nearest_state(phi, psi)[0, 0] == 0


def test_frozen_column_all_nan():
    T = 20
    rng = np.random.default_rng(0)
    phi = rng.uniform(-180, 180, size=(T, 2))
    psi = rng.uniform(-180, 180, size=(T, 2))
    phi[:, 1] = np.nan
    psi[:, 1] = np.nan
    frozen = np.array([False, True])
    out = assign_states(phi, psi, frozen)
    assert (out[:, 1] == FROZEN_STATE).all()
    assert not (out[:, 0] == FROZEN_STATE).all()


def test_split_trajectories_differ_from_concat():
    # Review Focus 1：拼接后再指派会在轨迹边界把状态带过去
    frozen = np.array([False])
    t1_phi = np.asarray([[-63.0], [-120.0]])   # A -> B（结束于 B）
    t1_psi = np.asarray([[-43.0], [130.0]])
    t2_phi = np.asarray([[-100.0], [-63.0]])   # 核心外(最近 A) -> A
    t2_psi = np.asarray([[-100.0], [-43.0]])
    sep = [assign_states(t1_phi, t1_psi, frozen),
           assign_states(t2_phi, t2_psi, frozen)]
    concat = assign_states(np.vstack([t1_phi, t2_phi]),
                           np.vstack([t1_psi, t2_psi]), frozen)
    joined = np.vstack(sep)
    assert not (joined == concat).all()
    # 具体差异在第 2 帧（第二条轨迹的第 0 帧）
    assert concat[2, 0] == 1        # 拼接版从上一轨迹继承 B
    assert joined[2, 0] == 0        # 单独指派取 nearest_state -> A


def test_input_validation():
    frozen = np.array([False])
    with pytest.raises(ValueError):
        assign_states(np.zeros((5, 2, 1)), np.zeros((5, 2)), frozen)
    with pytest.raises(ValueError):
        assign_states(np.zeros((5, 2)), np.zeros((4, 2)), frozen)
    with pytest.raises(ValueError):
        assign_states(np.zeros((5, 2)), np.zeros((5, 2)), np.array([False, False, False]))


def test_vocab_constants():
    assert STATE_NAMES == ("A", "B", "L")
    assert N_STATES == 3
    assert FROZEN_STATE == 1


def test_leading_segment_holds_frame0_nearest():
    # 审阅 P1-4：开头若干帧都在核心外且各帧 nearest 不同 -> 全部保持第 0 帧的 nearest
    seq = [(-100, -100),   # nearest A
           (-100, 80),     # nearest B（若逐帧取 nearest 会在这里伪跳变）
           (-100, -100),
           (60, 45),       # 进入 L 核心
           (-100, 80)]     # 核心外：保持 L
    phi = np.asarray([[p[0]] for p in seq], dtype=np.float64)
    psi = np.asarray([[p[1]] for p in seq], dtype=np.float64)
    assert nearest_state(phi[1:2], psi[1:2])[0, 0] == 1
    out = assign_states(phi, psi, np.array([False]))
    assert list(out[:, 0]) == [0, 0, 0, 2, 2]


def test_never_in_core_is_constant():
    phi = np.asarray([[-100.0], [-100.0], [-100.0]])
    psi = np.asarray([[-100.0], [80.0], [-100.0]])
    out = assign_states(phi, psi, np.array([False]))
    assert (out[:, 0] == out[0, 0]).all()
