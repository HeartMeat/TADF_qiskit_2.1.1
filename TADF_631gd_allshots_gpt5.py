# Python 3.11.13 / Qiskit 2.1.1 / qiskit-aer 0.17.1 / NumPy 2.3.x / SciPy 1.16.x / Matplotlib 3.10.x
# Env: macOS (Apple M3 Pro), conda env: TADF_211
#
# ──────────────────────────────────────────────────────────────────────────────────
# 본 스크립트는 다음 논문 워크플로우를 최신 Qiskit으로 "단일 파일"에 재현한다.
#   Motta et al., “Applications of quantum computing for investigations of electronic
#   transitions in phenylsulfonyl-carbazole TADF emitters” (PSPCz 계열, HOMO–LUMO
#   2-qubit 액티브 스페이스, 6-31G(d)), qEOM-VQE & VQD로 S0/T1/S1와 ΔE_ST 분석.
#   (논문/요약: arXiv:2007.15795 등)
#
# 알고리즘:
#   1) VQE (ground state S0): ansatz = Ry(depth=1)+CX, HF(비트스트링|10>) 시동, SLSQP
#   2) qEOM-VQE (excited states T1,S1): Ollitrault et al., PRResearch 2, 043140 (2020)
#      - “대칭화된 이중 교환자(double commutator)” 기반의 행렬 [M,Q,V,W] 구성
#      - 일반화 고유값 문제 A v = ω B v를 풀어 양의 ω만 취함 → E_abs = E0 + ω
#      - 여기 연산자(비에르미트)로 σ⁺ 및 Z-드레싱을 사용(아래 설명)
#      - EstimatorV2에는 항상 Hermitian 파울리 항만 전달(항별 기대값 측정 후 고전합)
#   3) VQD (excited states T1,S1): Higgott et al., Quantum 3, 156 (2019)
#      - 코스트 = ⟨H⟩ + ∑_k β_k |⟨ψ(θ)|ψ_k⟩|^2   (오버랩 패널티)
#      - 여기서는 겹침 |⟨ψ(θ)|ψ_k⟩|^2을 n-qubit 완전 projective Z-자기상관(=Hadamard test 無)
#        로 계산 가능한 projector 분해로 측정(2-qubit이므로 간단)
#
# 화학/매핑:
#   - HOMO–LUMO만의 액티브 스페이스(전자 2개/오비탈 2개)를 파리티 감축하여 2-qubit 모델로.
#   - 논문 보충자료 Table S2의 파울리 계수(6-31G(d))를 그대로 사용.
#   - 해밀토니안 형식: H = h_II·II + h_ZI·(ZI − IZ) + h_ZZ·ZZ + h_XX·XX
#                      + h_XI·(XI + IX + XZ − ZX)
#     (TADF 논문 보충자료 식과 동일한 형태로 구현. 각 분자별 계수만 다름)
#
# qEOM-VQE 수학 개요 (Ollitrault 2020, PRR 2, 043140):
#   - 기준상태 |Ψ0(θ*)> (VQE 최적화 결과)를 참조로 하고, “여기 연산자” 집합 {E_a}를 정의.
#     E_a는 보통 비에르미트(예: σ^+, σ^−, UCC excitation 등). 여기서는 다음 4개 사용:
#       E1 = σ^+_0 = (IX − i·IY)/2              (오른쪽 qubit의 상승)
#       E2 = σ^+_1 = (XI − i·YI)/2              (왼쪽  qubit의 상승)
#       E3 = Z_L·σ^+_0 = (ZX − i·ZY)/2          (좌측 Z 드레싱 후 오른쪽 상승)
#       E4 = Z_R·σ^+_1 = (XZ − i·YZ)/2          (우측 Z 드레싱 후 왼쪽  상승)
#     ※ 이 4개는 저자 깃허브의 QSE 풀 {XI, IX, XZ, ZX}가 span하는 동일 서브스페이스를 생성.
#        (QSE는 에르미트 풀을 쓰지만, qEOM은 비에르미트 여기자 사용이 자연스러움.)
#   - 기대값 행렬 블록 (대칭화된 이중 교환자 사용):
#       M_ij = ⟨ [E_i^†, H, E_j] ⟩,   Q_ij = −⟨ [E_i^†, H, E_j^†] ⟩
#       V_ij = ⟨ [E_i^†,     E_j] ⟩,  W_ij = −⟨ [E_i^†,     E_j^†] ⟩
#     여기서 [A,B] = AB − BA,  [A,B,C] = ½( [ [A,B], C ] + [ A, [B,C] ] )
#     (이 구성은 ω가 실수가 되도록 보장하며, 수치적으로는 Hermitian 블록을 형성)
#   - 일반화 고유값 문제:
#       A v = ω B v,   A = [[M, Q],[Q*, M*]],   B = [[V, W], [−W*, −V*]]
#     양의 ω만 물리적 여기 에너지로 채택. 절대 에너지는 E_abs = E0 + ω.
#
# 구현 노트:
#   - EstimatorV2에는 “관측가능이 Hermitian”이어야 하므로(문서 가이드),
#     비헤르미트 파울리 결합을 통째로 넘기지 않고 “파울리 항별 기대값”을 측정해
#     고전적으로 계수(복소포함)를 곱해 합산한다. 이 방식은 Qiskit 이슈/가이드와 부합.
#   - try/except, assert를 쓰지 않아 에러 발생 시 즉시 중단(연구용 코드 스타일).
#   - 모든 import는 파일 최상단.
#   - 하드코딩 최소화: 분자/계수 사전, ansatz 설정, β, 반복 횟수 등만 상단에서 통제.
#   - 출력: 주요 수치(mHa & eV 병기), FCI 대비 비교 그래프, 레벨 다이어그램, ΔE_ST 상관.
#
# 참고 웹소스(요지):
#   • qEOM-VQE 수학 (이중교환자, 일반화 고유값): Ollitrault et al., PRR 2, 043140 (2020).  # [1]
#   • VQD(오버랩 페널티): Higgott et al., Quantum 3, 156 (2019).                              # [2]
#   • TADF 논문: PSPCz 계열, HOMO–LUMO 2-qubit, qEOM-VQE & VQD 사용.                         # [3]
#   • EstimatorV2 관측가능 가이드/관행: 항별 Hermitian 관측가능 측정 후 고전합.               # [4]
#   (링크는 본 스크립트 상단 설명에 인용)
# ──────────────────────────────────────────────────────────────────────────────────

import numpy as np

import matplotlib
matplotlib.use("TkAgg")  # macOS/IntelliJ 환경에서 상호작용 플롯을 위해 TkAgg 사용
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Dict, List, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import EstimatorV2 as AerEstimator

from scipy.optimize import minimize
from scipy.linalg import eig as scipy_eig
from numpy.linalg import eigh


# ============================== 전역/단위/기본 설정 ===============================

# 샘플링 정밀도(EstimatorV2의 target precision). 시뮬레이터 사용이므로 상대적으로 빡셈.
TARGET_PRECISION = 1e-4
# 최적화/샘플링 난수 고정(재현성)
RNG_SEED = 1337

# 단위계
HARTREE_TO_EV = 27.211386245981
MHA = 1.0e3  # milli-Hartree
def ha_to_ev(x_ha: float) -> float: return x_ha * HARTREE_TO_EV
def ha_to_mha(x_ha: float) -> float: return x_ha * MHA


# ============================== 1,2-qubit Pauli 행렬 ==============================

# 2x2 표준 Pauli 및 항등
I2 = np.eye(2, dtype=complex)
X  = np.array([[0, 1], [1, 0]], dtype=complex)
Y  = np.array([[0,-1j],[1j, 0]], dtype=complex)
Z  = np.array([[1, 0], [0,-1]], dtype=complex)

def kron(a, b):  # 2-qubit tensor product
    return np.kron(a, b)


# ============================== 2-qubit 해밀토니안 ================================

def build_hamiltonian_matrix(h: Dict[str, float]) -> np.ndarray:
    """
    H(2-qubit) = h_II·II + h_ZI·(ZI − IZ) + h_ZZ·ZZ + h_XX·XX
               + h_XI·(XI + IX + XZ − ZX)
    (논문 보충자료 Table S2 계수 사용, HOMO–LUMO 파리티 매핑 결과)
    """
    return (
            h["h_II"] * kron(I2, I2) +
            h["h_ZI"] * (kron(Z, I2) - kron(I2, Z)) +
            h["h_ZZ"] * kron(Z, Z) +
            h["h_XX"] * kron(X, X) +
            h["h_XI"] * (kron(X, I2) + kron(I2, X) + kron(X, Z) - kron(Z, X))
    )

def build_hamiltonian_op(h: Dict[str, float]) -> SparsePauliOp:
    """
    같은 해밀토니안을 파울리 리스트(SparsePauliOp)로 표현.
    EstimatorV2에 ⟨H⟩ 평가용으로 사용(ground state VQE/VQD cost 등).
    """
    return SparsePauliOp.from_list([
        ("II", h["h_II"]),
        ("ZI", h["h_ZI"]), ("IZ", -h["h_ZI"]),
        ("ZZ", h["h_ZZ"]),
        ("XX", h["h_XX"]),
        ("XI", h["h_XI"]), ("IX", h["h_XI"]),
        ("XZ", h["h_XI"]), ("ZX", -h["h_XI"]),
    ])

def format_hamiltonian_equation(h: Dict[str, float]) -> str:
    """
    사람이 읽기 쉬운 해밀토니안 전개(검증용 로그).
    """
    return (
        "H = h_II·II + h_ZI·(ZI - IZ) + h_ZZ·ZZ + h_XX·XX + h_XI·(XI + IX + XZ - ZX)\n"
        f"H = {h['h_II']:+.6f}·II  {h['h_ZI']:+.6f}·ZI  {-h['h_ZI']:+.6f}·IZ  "
        f"{h['h_ZZ']:+.6f}·ZZ  {h['h_XX']:+.6f}·XX  {h['h_XI']:+.6f}·XI  "
        f"{h['h_XI']:+.6f}·IX  {h['h_XI']:+.6f}·XZ  {-h['h_XI']:+.6f}·ZX  (Ha)"
    )


# ============================== Table S2: 6-31G(d) 계수 ===========================

COEFFS_631GD: Dict[str, Dict[str, float]] = {
    # 논문 보충자료 Table S2 (6-31G(d))에서 발췌된 2-qubit Pauli 계수
    "PSPCz":   {"h_II": -0.518418, "h_ZI": -0.136555, "h_ZZ": -0.025866, "h_XI": -0.000296, "h_XX": 0.015725},
    "2F-PSPCz":{"h_II": -0.553646, "h_ZI": -0.126735, "h_ZZ": -0.034240, "h_XI":  0.012806, "h_XX": 0.011196},
    "4F-PSPCz":{"h_II": -0.540009, "h_ZI": -0.181261, "h_ZZ": -0.083888, "h_XI":  0.000951, "h_XX": 0.000107},
}
MOLECULES = COEFFS_631GD  # 일반화를 위해 별칭


# ============================== Ansatz: Ry(depth=1)+CX, HF |10> ====================

@dataclass
class RyAnsatzConfig:
    n_qubits: int = 2
    reps: int = 1
    include_final_layer: bool = True
    entangle_cx: bool = True
    hf_bitstring: str = "10"  # 왼쪽(상위비트)→오른쪽(하위비트) 순서로 표기

def build_ry_ansatz(cfg: RyAnsatzConfig) -> Tuple[QuantumCircuit, ParameterVector]:
    """
    논문에서 시뮬레이터/디바이스에 쓴 간단한 ansatz: Ry(각 큐빗) → CX → (옵션)Ry
    - HF 초기화: |10> (액티브스페이스에서 전자 수/스핀 제약 반영)
    - reps=1이므로 파라미터 수 = 2(qubits) × 2(layers) = 4
    """
    qc = QuantumCircuit(cfg.n_qubits)
    # HF 점화: 문자열 우측이 qubit0이므로 역순으로 읽음
    for q, b in enumerate(cfg.hf_bitstring[::-1]):
        if b == "1": qc.x(q)

    n_layers = 2 if cfg.include_final_layer else 1
    thetas = ParameterVector("theta", length=cfg.n_qubits * n_layers * cfg.reps)
    p = 0
    for _ in range(cfg.reps):
        for q in range(cfg.n_qubits):
            qc.ry(thetas[p], q); p += 1
        if cfg.entangle_cx:
            qc.cx(0, 1)
        if cfg.include_final_layer:
            for q in range(cfg.n_qubits):
                qc.ry(thetas[p], q); p += 1
    return qc, thetas


# ============================== EstimatorV2 & 기대값 유틸 ==========================

def make_estimator() -> AerEstimator:
    """
    Qiskit Aer EstimatorV2 시뮬레이터 (샘플링 시드 고정).
    참고: EstimatorV2는 (회로, 관측가능) 쌍의 기대값을 추정.
    관측가능은 Hermitian이어야 하므로, 비헤르미트 파울리 합은 항별 측정 후 고전합으로 처리. [4]
    """
    return AerEstimator(options={"run_options": {"seed_simulator": RNG_SEED}})

def expval_pauli_sum(est: AerEstimator, qc: QuantumCircuit, theta: np.ndarray,
                     pauli_terms: List[Tuple[str, complex]]) -> complex:
    """
    ⟨∑_k c_k P_k⟩를 평가. Estimator에는 항상 "단일 파울리 P_k (계수 1.0)"만 전달하여
    Hermitian 제약을 지키고, 결과를 고전적으로 ∑ c_k ⟨P_k⟩로 합산.
    (계수 c_k는 일반적으로 복소수: qEOM의 비에르미트 여기자 확장에서 필수)
    """
    total = 0.0 + 0.0j
    for lbl, coeff in pauli_terms:
        op = SparsePauliOp.from_list([(lbl, 1.0)])
        res = est.run([(qc, [[op]], np.atleast_2d(theta))], precision=TARGET_PRECISION).result()
        ev = complex(res[0].data.evs[0, 0])  # 수치오차 대비 복소 취급
        total += complex(coeff) * ev
    return total


# ============================== 파울리 문자열 대수 유틸 ============================

def pauli_mult_1q(a: str, b: str) -> Tuple[complex, str]:
    """
    1-qubit 파울리 곱: X·Y = iZ, Y·X = −iZ, 등.
    (위상因子는 qEOM 이중교환자 전개에서 중요)
    """
    if a == "I": return (1+0j, b)
    if b == "I": return (1+0j, a)
    if a == b:   return (1+0j, "I")
    table = {
        ("X","Y"):(1j,"Z"), ("Y","Z"):(1j,"X"), ("Z","X"):(1j,"Y"),
        ("Y","X"):(-1j,"Z"),("Z","Y"):(-1j,"X"),("X","Z"):(-1j,"Y"),
    }
    return table[(a,b)]

def pauli_str_mult(s1: str, s2: str) -> Tuple[complex, str]:
    """
    2-qubit 파울리 문자열 곱. 예: "XZ"·"ZX" = (−i)^2 · "YY" = (−1)·"YY"
    """
    phase = 1+0j
    out = []
    for a, b in zip(s1, s2):
        ph, c = pauli_mult_1q(a, b)
        phase *= ph
        out.append(c)
    return phase, "".join(out)

def sum_add(dst: Dict[str, complex], lbl: str, coeff: complex):
    """
    파울리 합의 계수 누적(0 계수 항 제거는 별도).
    """
    dst[lbl] = dst.get(lbl, 0.0+0.0j) + coeff

def sum_mul(A: Dict[str, complex], B: Dict[str, complex]) -> Dict[str, complex]:
    """
    (∑ a_i P_i)·(∑ b_j Q_j) = ∑_ij a_i b_j (phase(P_i,Q_j), P_iQ_j)
    """
    out: Dict[str, complex] = {}
    for l1, c1 in A.items():
        for l2, c2 in B.items():
            ph, l12 = pauli_str_mult(l1, l2)
            sum_add(out, l12, c1 * c2 * ph)
    return out

def sum_sub(A: Dict[str, complex], B: Dict[str, complex]) -> Dict[str, complex]:
    """
    A − B (0이 된 항은 제거)
    """
    out = dict(A)
    for l2, c2 in B.items():
        out[l2] = out.get(l2, 0.0+0.0j) - c2
        if abs(out[l2]) == 0.0:
            out.pop(l2, None)
    return out

def comm(A: Dict[str, complex], B: Dict[str, complex]) -> Dict[str, complex]:
    """
    [A, B] = A·B − B·A
    """
    return sum_sub(sum_mul(A, B), sum_mul(B, A))

def double_comm(A: Dict[str, complex], B: Dict[str, complex], C: Dict[str, complex]) -> Dict[str, complex]:
    """
    [A, B, C] = ½( [[A,B], C] + [A, [B,C]] )  — qEOM의 대칭화된 이중 교환자
    (Ollitrault 2020 Eq.(2) 주변 공식)
    """
    AB   = comm(A, B)
    ABC1 = comm(AB, C)
    BC   = comm(B, C)
    ABC2 = comm(A, BC)
    out: Dict[str, complex] = {}
    for d in (ABC1, ABC2):
        for l, c in d.items():
            out[l] = out.get(l, 0.0+0.0j) + 0.5 * c
    return out

def dict_to_list(d: Dict[str, complex]) -> List[Tuple[str, complex]]:
    """
    SparsePauliOp.from_list로 넘길 (label, coeff) 리스트 (0 계수 제거)
    """
    return [(l, c) for l, c in d.items() if abs(c) > 1e-12]

def dagger(D: Dict[str, complex]) -> Dict[str, complex]:
    """
    (∑ c_l P_l)^† = ∑ c_l^* P_l    (파울리는 에르미트)
    """
    return {l: np.conjugate(c) for l, c in D.items()}


# ============================== qEOM: 여기 연산자 집합 =============================

# σ^+_q = (X_q − i Y_q)/2,  σ^-_q = (X_q + i Y_q)/2
# 문자열 표기: 왼→오 순서가 [qubit1, qubit0].
def sigma_plus(q: int) -> Dict[str, complex]:
    """
    상승(creation) 연산자 σ^+의 파울리 분해를 2-qubit 문자열로 표현.
    q=0 (오른쪽/하위비트): IX, IY 조합
    q=1 (왼쪽 /상위비트): XI, YI 조합
    """
    if q == 0:
        return {"IX": 0.5, "IY": -0.5j}
    else:
        return {"XI": 0.5, "YI": -0.5j}

def left_Z_mul(A: Dict[str, complex]) -> Dict[str, complex]:
    """
    왼쪽(qubit1)에 Z를 곱해 Z-드레싱: 라벨 앞에 'Z'를 텐서곱으로 부여
    """
    out: Dict[str, complex] = {}
    for lbl, c in A.items():
        ph, lbl2 = pauli_str_mult("ZI", lbl)  # 왼쪽 Z, 오른쪽 I
        out[lbl2] = out.get(lbl2, 0.0) + c * ph
    return out

def right_Z_mul(A: Dict[str, complex]) -> Dict[str, complex]:
    """
    오른쪽(qubit0)에 Z를 곱해 Z-드레싱
    """
    out: Dict[str, complex] = {}
    for lbl, c in A.items():
        ph, lbl2 = pauli_str_mult("IZ", lbl)
        out[lbl2] = out.get(lbl2, 0.0) + c * ph
    return out

def build_qeom_excitations() -> List[Dict[str, complex]]:
    """
    qEOM 여기 연산자 집합 {E_a} 구성.
    - 저자 깃허브의 QSE 풀 {XI, IX, XZ, ZX}가 생성하는 서브스페이스를
      qEOM에 자연스러운 비에르미트 여기자(σ⁺)와 Z-드레싱으로 재표현.
    - 작은(2-qubit) 공간에서 T1/S1을 안정적으로 얻는 데 충분한 최소 집합.
    """
    E1 = sigma_plus(0)             # (IX − i·IY)/2
    E2 = sigma_plus(1)             # (XI − i·YI)/2
    E3 = left_Z_mul(sigma_plus(0)) # (ZX − i·ZY)/2
    E4 = right_Z_mul(sigma_plus(1))# (XZ − i·YZ)/2
    return [E1, E2, E3, E4]


# ============================== qEOM 행렬 구축 & 고유값 ============================

def qeom_matrices(H: Dict[str, complex],
                  excitations: List[Dict[str, complex]],
                  est: AerEstimator,
                  ansatz: QuantumCircuit, pvec: ParameterVector, theta0: np.ndarray
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    qEOM의 핵심 블록 행렬 M,Q,V,W를 대칭화된 이중교환자/단일교환자로 구성.
    각 엔트리는 참조상태 |Ψ0(θ0)>에 대한 기대값.
      M_ij = ⟨ [E_i^†, H, E_j] ⟩
      Q_ij = −⟨ [E_i^†, H, E_j^†] ⟩
      V_ij = ⟨ [E_i^†,      E_j] ⟩
      W_ij = −⟨ [E_i^†,      E_j^†] ⟩
    (Ollitrault 2020 정의. M,V는 Hermitian으로 대칭화)
    """
    K = len(excitations)
    M = np.zeros((K, K), dtype=complex)
    Q = np.zeros((K, K), dtype=complex)
    V = np.zeros((K, K), dtype=complex)
    W = np.zeros((K, K), dtype=complex)

    for i, Ei in enumerate(excitations):
        Ei_d = dagger(Ei)
        for j, Ej in enumerate(excitations):
            Ej_d = dagger(Ej)
            DC_M = double_comm(Ei_d, H, Ej)      # [E_i^†, H, E_j]
            DC_Q = double_comm(Ei_d, H, Ej_d)    # [E_i^†, H, E_j^†]
            C_V  = comm(Ei_d, Ej)                # [E_i^†,      E_j]
            C_W  = comm(Ei_d, Ej_d)              # [E_i^†,      E_j^†]

            m_ij =  expval_pauli_sum(est, ansatz, theta0, dict_to_list(DC_M))
            q_ij = -expval_pauli_sum(est, ansatz, theta0, dict_to_list(DC_Q))
            v_ij =  expval_pauli_sum(est, ansatz, theta0, dict_to_list(C_V))
            w_ij = -expval_pauli_sum(est, ansatz, theta0, dict_to_list(C_W))

            M[i, j] = m_ij
            Q[i, j] = q_ij
            V[i, j] = v_ij
            W[i, j] = w_ij

    # 수치안정: Hermitian 블록은 대칭화
    M = 0.5 * (M + M.conj().T)
    V = 0.5 * (V + V.conj().T)
    return M, Q, V, W

def qeom_excitation_energies(H_op: SparsePauliOp,
                             cfg: RyAnsatzConfig, theta0: np.ndarray,
                             excitations: List[Dict[str, complex]] | None = None
                             ) -> np.ndarray:
    """
    1) H_op → dict(label→coeff)로 변환
    2) qEOM 행렬 M,Q,V,W 구성
    3) 일반화 고유값 A v = ω B v 풀기 (복소→실수화), 양의 ω만 선택
    """
    if excitations is None:
        excitations = build_qeom_excitations()

    # H를 파울리 딕셔너리로 변환 (복소계수 포함 가능)
    H_dict: Dict[str, complex] = {lbl: complex(c) for lbl, c in zip(H_op.paulis.to_labels(), H_op.coeffs)}

    # 참조 회로/Estimator 준비 (VQE 최적 θ0에 대해 기대값 측정)
    ansatz, pvec = build_ry_ansatz(cfg)
    est = make_estimator()

    M, Q, V, W = qeom_matrices(H_dict, excitations, est, ansatz, pvec, theta0)

    # 일반화 고유값 문제: A v = ω B v
    A = np.block([[M, Q], [Q.conj(), M.conj()]])
    B = np.block([[V, W], [-W.conj(), -V.conj()]])
    w, _ = scipy_eig(A, B)

    # 수치오차로 복소 꼬리가 생길 수 있어 실수부만 취함
    w = np.real_if_close(w, tol=1e-8).real
    w_pos = np.sort(w[w > 1e-8])  # 물리적 양의 여기 에너지
    return w_pos


# ============================== FCI 기준(정확 해) ================================

def exact_diagonalization(H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    4x4(2-qubit) 해밀토니안의 정확한 고유분해. FCI 기준값 역할.
    """
    w, v = eigh(H)
    idx = np.argsort(w)
    return w[idx], v[:, idx]


# ============================== VQE (S0) ===========================================

@dataclass
class VQEResult:
    energy: float
    params: np.ndarray

def vqe_ground_state(H_op: SparsePauliOp, cfg: RyAnsatzConfig,
                     x0: np.ndarray | None = None, maxiter: int = 500) -> VQEResult:
    """
    S0 바닥상태 VQE (SLSQP). 코스트 = ⟨H⟩.
    ansatz = Ry(depth=1)+CX, 파라미터 4개. 초기값 x0 미지정 시 0 벡터.
    """
    ansatz, pvec = build_ry_ansatz(cfg)
    npar = len(pvec)
    if x0 is None:
        x0 = np.zeros(npar)

    est = make_estimator()
    def cost(theta):
        ev = float(est.run([(ansatz, [[H_op]], np.atleast_2d(np.asarray(theta, float)))],
                           precision=TARGET_PRECISION).result()[0].data.evs[0, 0])
        return ev

    res = minimize(cost, x0, method="SLSQP",
                   options={"maxiter": maxiter, "ftol": 1e-10, "disp": False})
    theta_opt = res.x
    e_opt = float(est.run([(ansatz, [[H_op]], np.atleast_2d(theta_opt))],
                          precision=TARGET_PRECISION).result()[0].data.evs[0, 0])

    print(f"VQE [SLSQP] final: E0 = {e_opt:.9f} Ha  |  n_params={npar}")
    return VQEResult(energy=e_opt, params=theta_opt)


# ============================== VQD (Overlap-penalized VQE) ========================

@dataclass
class VQDResult:
    energy: float
    params: np.ndarray

def fidelity_via_projector(est: AerEstimator, cfg: RyAnsatzConfig,
                           ansatz: QuantumCircuit, pvec: ParameterVector,
                           theta_a: np.ndarray, theta_b: np.ndarray) -> float:
    """
    |⟨ψ(θ_a)|ψ(θ_b)⟩|^2을 완전 projective Z-기준 분해로 계산.
    - 회로: |ψ(θ_b)> 준비 후 |ψ(θ_a)>의 역연산을 붙여, |0…0> 투영 확률을 평가.
    - n-qubit에서 projector = (1/2^n) ∑_z ⊗_q Z_q^{z_q}  (모든 Z-텐서의 평균)
      → 모든 Z/I 조합의 기대값 평균으로 중첩도 산출.
    """
    n = cfg.n_qubits

    # |ψ(θ_b)> 준비 후 |ψ(θ_a)>† 적용
    circ = QuantumCircuit(n)
    bindB = {par: float(val) for par, val in zip(pvec, theta_b)}
    circ.compose(ansatz.assign_parameters(bindB), inplace=True)
    circ.compose(
        ansatz.assign_parameters({par: float(val) for par, val in zip(pvec, theta_a)}).inverse(),
        inplace=True
    )

    # projector = (1/2^n) * sum over all Z-strings
    terms = []
    for mask in range(1 << n):
        label = "".join("Z" if ((mask >> q) & 1) else "I" for q in range(n))
        terms.append((label, 1.0))
    projector = (1.0 / (2 ** n)) * SparsePauliOp.from_list(terms)

    res = est.run([(circ, [[projector]])], precision=TARGET_PRECISION).result()
    F = float(res[0].data.evs[0, 0])
    # 수치 안정화(0~1 클램프)
    return max(0.0, min(1.0, F))

def vqd_state(H_op: SparsePauliOp,
              refs: List[np.ndarray], betas: List[float],
              cfg: RyAnsatzConfig, x0: np.ndarray | None = None, maxiter: int = 500,
              tag: str = "VQD") -> VQDResult:
    """
    VQD 단일 상태 탐색(레퍼런스 상태들과의 오버랩 페널티 포함).
    refs: 이미 찾은 상태들의 파라미터 리스트(예: [θ(S0)], [θ(S0), θ(T1)], …)
    betas: 각 레퍼런스에 대응하는 β 가중치(양수).
    """
    ansatz, pvec = build_ry_ansatz(cfg)
    npar = len(pvec)
    if x0 is None:
        x0 = np.zeros(npar)

    est = make_estimator()
    def cost(theta):
        theta = np.asarray(theta, dtype=float)
        E = float(est.run([(ansatz, [[H_op]], np.atleast_2d(theta))],
                          precision=TARGET_PRECISION).result()[0].data.evs[0, 0])
        pen = 0.0
        for beta, ref_theta in zip(betas, refs):
            F = fidelity_via_projector(est, cfg, ansatz, pvec, theta, ref_theta)
            pen += beta * F
        return E + pen

    res = minimize(cost, x0, method="SLSQP",
                   options={"maxiter": maxiter, "ftol": 1e-10, "disp": False})
    theta_opt = res.x
    E_final = float(est.run([(ansatz, [[H_op]], np.atleast_2d(theta_opt))],
                            precision=TARGET_PRECISION).result()[0].data.evs[0, 0])
    Fs_final = [fidelity_via_projector(est, cfg, ansatz, pvec, theta_opt, r) for r in refs]
    pen_final = float(np.dot(betas, Fs_final))
    print(f"{tag} [β={','.join(f'{b:.2f}' for b in betas)}] final: E = {E_final:.9f} Ha, "
          f"penalty = {pen_final:.9f}, maxF = {max(Fs_final):.6f}")
    return VQDResult(energy=E_final, params=theta_opt)

def vqd_state_multistart(H_op: SparsePauliOp,
                         refs: List[np.ndarray], betas: List[float],
                         cfg: RyAnsatzConfig, num_starts: int = 2,
                         maxiter: int = 500, seed: int = RNG_SEED,
                         tag: str = "VQD") -> VQDResult:
    """
    VQD 멀티스타트: 서로 다른 초기 파라미터에서 시작해 가장 낮은 값을 채택.
    2-qubit/간단 ansatz에서는 2~4개면 충분.
    """
    ansatz, pvec = build_ry_ansatz(cfg)
    npar = len(pvec)
    rng = np.random.default_rng(seed)

    best_res: VQDResult | None = None
    best_idx = -1
    for s in range(num_starts):
        x0 = rng.uniform(-np.pi, np.pi, size=npar)
        res = vqd_state(H_op, refs, betas, cfg, x0=x0, maxiter=maxiter, tag=f"{tag}-start{s+1}")
        if (best_res is None) or (res.energy < best_res.energy):
            best_res = res
            best_idx = s
    print(f"{tag} select: best start = {best_idx+1}  |  E = {best_res.energy:.9f} Ha")
    return best_res  # type: ignore[return-value]


# ============================== 출력/요약 유틸 =====================================

@dataclass
class Spectrum:
    E0: float
    T1: float
    S1: float
    all_evals: np.ndarray | None = None  # 필요시 절대 에너지 배열 저장

def line_summary(tag: str, E0: float, T1: float, S1: float, ref: Spectrum | None = None) -> str:
    """
    한 줄 요약(단위: mHa와 eV를 병기). ref가 주어지면 FCI 대비 편차도 함께 표기.
    """
    s = []
    s.append(
        f"{tag:10s} | S0 {ha_to_mha(E0):9.3f} mHa ({ha_to_ev(E0):7.3f} eV) | "
        f"T1 {ha_to_mha(T1):9.3f} mHa ({ha_to_ev(T1):7.3f} eV) | "
        f"S1 {ha_to_mha(S1):9.3f} mHa ({ha_to_ev(S1):7.3f} eV) | "
        f"ΔEST {ha_to_mha(S1-T1):8.3f} mHa ({ha_to_ev(S1-T1):6.3f} eV)"
    )
    if ref is not None:
        s.append(
            f"   deviation vs FCI: S0 {ha_to_mha(E0-ref.E0):+7.3f} mHa, "
            f"T1 {ha_to_mha(T1-ref.T1):+7.3f} mHa, "
            f"S1 {ha_to_mha(S1-ref.S1):+7.3f} mHa, "
            f"ΔEST {ha_to_mha((S1-T1)-(ref.S1-ref.T1)):+7.3f} mHa"
        )
    return "\n".join(s)


# ============================== 메인 워크플로우 ====================================

def run_workflow():
    """
    전체 파이프라인:
      For each mol in {PSPCz, 2F-PSPCz, 4F-PSPCz}:
        (1) Table S2 계수로 2-qubit H 구성 → FCI(정확해)
        (2) VQE(SLSQP)로 E0, θ0
        (3) qEOM-VQE: 여기 에너지 ω>0 → 절대에너지 E_abs = E0 + ω → {T1,S1}
        (4) VQD: T1(β=5, ref=[θ0]) → S1(β=5, ref=[θ0, θ(T1)])
      마지막에 FCI vs qEOM-VQE vs VQD 비교 요약/시각화.
    """
    cfg = RyAnsatzConfig()
    results: Dict[str, Dict[str, object]] = {}

    # (선택) ansatz 회로 그려 확인
    ansatz_for_plot, _ = build_ry_ansatz(cfg)
    fig_circ = ansatz_for_plot.draw(output="mpl")
    fig_circ.suptitle("Ry (depth=1) + CX with HF |10> preparation")
    plt.show()

    for mol, coeffs in MOLECULES.items():
        print(f"\n[{mol}] Hamiltonian (6-31G(d), HOMO-LUMO 2-qubit)")
        print(format_hamiltonian_equation(coeffs))

        # (1) FCI 기준
        H_mat = build_hamiltonian_matrix(coeffs)
        H_op  = build_hamiltonian_op(coeffs)
        evals, _ = exact_diagonalization(H_mat)
        spec_fci = Spectrum(E0=float(evals[0]), T1=float(evals[1]), S1=float(evals[2]), all_evals=evals)
        print(f"FCI final: S0 = {spec_fci.E0:.9f} Ha,  T1 ≈ {spec_fci.T1:.9f} Ha,  S1 ≈ {spec_fci.S1:.9f} Ha\n")

        # (2) VQE (ground state)
        vqe_res = vqe_ground_state(H_op, cfg, x0=None, maxiter=500)
        print()

        # (3) qEOM-VQE (여기 연산자: {σ⁺_0, σ⁺_1, Z_L σ⁺_0, Z_R σ⁺_1})
        omegas = qeom_excitation_energies(H_op, cfg, vqe_res.params, excitations=None)
        E_abs  = np.sort(vqe_res.energy + omegas)  # 절대 에너지
        # T1, S1은 가장 낮은 두 개의 여기 상태(2-qubit에서 안정적으로 2개 확보되도록 설계)
        T1_qeom, S1_qeom = float(E_abs[0]), float(E_abs[1])
        print(f"qEOM-VQE final: T1 ≈ {T1_qeom:.9f} Ha,  S1 ≈ {S1_qeom:.9f} Ha  |  basis = {{σ+_0, σ+_1, Z_L σ+_0, Z_R σ+_1}}\n")

        # (4) VQD (오버랩 패널티, β=5.0, 멀티스타트 2회)
        beta = 5.0
        vqd_T1 = vqd_state_multistart(
            H_op, refs=[vqe_res.params], betas=[beta], cfg=cfg,
            num_starts=2, maxiter=500, seed=RNG_SEED, tag="VQD-T1"
        )
        vqd_S1 = vqd_state_multistart(
            H_op, refs=[vqe_res.params, vqd_T1.params], betas=[beta, beta], cfg=cfg,
            num_starts=2, maxiter=500, seed=RNG_SEED + 1, tag="VQD-S1"
        )

        # 수집
        results[mol] = {
            "H_mat": H_mat, "H_op": H_op,
            "fci": spec_fci,
            "vqe": vqe_res,
            "qeom": Spectrum(E0=vqe_res.energy, T1=T1_qeom, S1=S1_qeom, all_evals=E_abs),
            "vqd":  Spectrum(E0=vqe_res.energy, T1=vqd_T1.energy, S1=vqd_S1.energy),
        }

    # ===================== 텍스트 요약 =====================
    print("=== PSPCz family (6-31G(d), HOMO-LUMO active space, 2 qubits) ===")
    for mol, R in results.items():
        fci = R["fci"]; qe = R["qeom"]; vq = R["vqd"]
        print(f"\n[{mol}]  energies in mHa (and eV)")
        print(line_summary("FCI", fci.E0, fci.T1, fci.S1))
        print(line_summary("qEOM-VQE", qe.E0, qe.T1, qe.S1, ref=fci))
        print(line_summary("VQD", vq.E0, vq.T1, vq.S1, ref=fci))

    # ===================== 시각화 1: 절대에너지 바 =====================
    mols = list(results.keys())
    ncols = 3
    fig, axes = plt.subplots(1, ncols, figsize=(15, 4), constrained_layout=True)
    for k, mol in enumerate(mols):
        ax = axes[k]
        fci = results[mol]["fci"]; qe = results[mol]["qeom"]; vq = results[mol]["vqd"]
        labels = ["S0", "T1", "S1"]; x = np.arange(len(labels)); w = 0.25
        bars1 = [fci.E0, fci.T1, fci.S1]
        bars2 = [qe.E0, qe.T1, qe.S1]
        bars3 = [vq.E0, vq.T1, vq.S1]
        ax.bar(x - w, [ha_to_mha(b) for b in bars1], w, label="FCI")
        ax.bar(x,      [ha_to_mha(b) for b in bars2], w, label="qEOM-VQE")
        ax.bar(x + w,  [ha_to_mha(b) for b in bars3], w, label="VQD")
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_title(mol); ax.set_ylabel("Energy [mHa]")
        ax.grid(True, alpha=0.3)
        if k == ncols - 1:
            ax.legend()
    plt.show()

    # ===================== 시각화 2: ΔE_ST 상관도 =====================
    fig2, ax2 = plt.subplots(1, 1, figsize=(6,5), constrained_layout=True)
    fci_gaps = []; qe_gaps = []; vqd_gaps = []
    for mol in mols:
        fci = results[mol]["fci"]; qe = results[mol]["qeom"]; vq = results[mol]["vqd"]
        fci_gaps.append(ha_to_ev(fci.S1 - fci.T1))
        qe_gaps.append(ha_to_ev(qe.S1 - qe.T1))
        vqd_gaps.append(ha_to_ev(vq.S1 - vq.T1))
    mn = min(fci_gaps + qe_gaps + vqd_gaps) - 0.1
    mx = max(fci_gaps + qe_gaps + vqd_gaps) + 0.1
    ax2.scatter(fci_gaps, qe_gaps, marker="o", label="qEOM-VQE")
    ax2.scatter(fci_gaps, vqd_gaps, marker="s", label="VQD")
    ax2.plot([mn, mx], [mn, mx], "k--", lw=1)
    ax2.set_xlabel("FCI ΔE_ST [eV]"); ax2.set_ylabel("Algorithm ΔE_ST [eV]")
    ax2.set_title("ΔE_ST correlation vs FCI")
    ax2.grid(True, alpha=0.3); ax2.legend()
    plt.show()

    # ===================== 시각화 3: 에너지 레벨 다이어그램 =====================
    fig3, axes3 = plt.subplots(1, ncols, figsize=(15, 4), constrained_layout=True)
    for k, mol in enumerate(mols):
        ax = axes3[k]
        fci = results[mol]["fci"]; qe = results[mol]["qeom"]; vq = results[mol]["vqd"]
        def draw_levels(x0, levels, color, label):
            for E in levels:
                y = ha_to_ev(E - fci.E0)  # FCI S0 기준 상대 에너지[eV]
                ax.hlines(y, x0-0.3, x0+0.3, lw=2, color=color)
            ax.text(x0, ha_to_ev(levels[-1] - fci.E0) + 0.05, label, ha="center")
        x0, x1, x2 = 0.8, 1.6, 2.4
        draw_levels(x0, [fci.E0, fci.T1, fci.S1], "C0", "FCI")
        draw_levels(x1, [qe.E0, qe.T1, qe.S1], "C1", "qEOM")
        draw_levels(x2, [vq.E0, vq.T1, vq.S1], "C2", "VQD")
        ax.set_title(mol); ax.set_ylabel("Energy rel. to FCI S0 [eV]")
        ax.set_xticks([x0, x1, x2]); ax.set_xticklabels(["FCI", "qEOM", "VQD"])
        ax.grid(True, axis="y", alpha=0.3)
    plt.show()

    # ===================== 시각화 4: FCI 대비 오차 바 =====================
    fig4, axes4 = plt.subplots(1, ncols, figsize=(15, 4), constrained_layout=True)
    for k, mol in enumerate(mols):
        ax = axes4[k]
        fci = results[mol]["fci"]; qe = results[mol]["qeom"]; vq = results[mol]["vqd"]
        labels = ["S0", "T1", "S1"]; x = np.arange(len(labels)); w = 0.35
        err_qe = [ha_to_mha(qe.E0 - fci.E0), ha_to_mha(qe.T1 - fci.T1), ha_to_mha(qe.S1 - fci.S1)]
        err_vq = [ha_to_mha(vq.E0 - fci.E0), ha_to_mha(vq.T1 - fci.T1), ha_to_mha(vq.S1 - fci.S1)]
        ax.bar(x - w/2, err_qe, w, label="qEOM-VQE vs FCI")
        ax.bar(x + w/2, err_vq, w, label="VQD vs FCI")
        ax.axhline(0.0, lw=1, color="k")
        ax.set_xticks(x); ax.set_xticklabels(labels)
        gap_err_qe = ha_to_mha((qe.S1 - qe.T1) - (fci.S1 - fci.T1))
        gap_err_vq = ha_to_mha((vq.S1 - vq.T1) - (fci.S1 - fci.T1))
        ax.set_title(f"{mol}\nΔE_ST err: qEOM {gap_err_qe:+.3f} mHa | VQD {gap_err_vq:+.3f} mHa")
        ax.set_ylabel("Error vs FCI [mHa]")
        ax.grid(True, axis="y", alpha=0.3)
        if k == ncols - 1:
            ax.legend(loc="upper left")
    plt.show()


if __name__ == "__main__":
    run_workflow()
