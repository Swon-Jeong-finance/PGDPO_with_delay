# PGDPO with Delay — 현재 코드 진행상황 및 P1-C DP Refinement Ladder 구현 권고안

- **검토일:** 2026-08-07
- **검토 대상:**
  - `PGDPO_delay_reference_layer_handoff_20260807(1).md`
  - `PGDPO_P1C_DP_ladder_design_20260807.md`
  - `pgdpo-delay.zip`
- **판정 기준:** 실제 압축파일 안의 코드가 handoff 문서보다 우선한다.
- **이번 문서의 목적:** 현재 reference/oracle 코드가 어디까지 닫혔는지 확인하고, P1-C small DP의 `dp_ladder.py`를 어떤 순서와 기준으로 구현할지 결정한다.

---

# 1. 최종 요약

## 1.1 한 문장 판정

> **P1-U와 P2의 exact reference layer는 실제 코드와 검증 명령까지 닫혀 있다. P1-C small DP는 analytic-terminal-expectation 및 sub-grid action patch가 반영되어 있지만, refinement ladder를 실행하기 전에 active-set 판정, exact final-step 처리, canonical policy readout, storage/streaming, domain audit를 먼저 보강해야 한다.**

현재 상태를 구분하면 다음과 같다.

| 층 | 현재 상태 | 판정 |
|---|---|---|
| Theorems 5.1–5.3 및 이론 원고 | 완료 | 실험 설계로 넘어갈 수 있음 |
| P1-U exact oracle / recovery targets | 완료 | `verify` 통과 |
| P1-U estimator contract | 완료 | full verify에서 handoff 수치 재현 |
| P2 structured oracle | 완료 | dense/mode 검산 통과 |
| P1-C main, `H=16` | 평가 설계 완료 | global DP oracle 없이 paired objective/KKT/switching 사용 |
| P1-C small, `H=3` | DP prototype + 일부 patch 완료 | 아직 certified reference 아님 |
| P1-C `dp_ladder.py` | **미구현** | 설계 MD만 존재 |
| 공통 Stage I / Stage II solver | skeleton | `run`은 아직 연결되지 않음 |
| P3-R / P3-D / P4 | 미착수 | package 디렉터리만 존재 |

따라서 다음 작업은 곧바로 r1–r3를 돌리는 것이 아니라, **P1-C DP correctness patch → ladder engine → final certification run** 순서가 맞다.

---

# 2. 실제 저장소 구조와 실행 확인

## 2.1 실제 코드 구조

압축파일에는 다음 구조가 들어 있다.

```text
pgdpo-delay/
├── main.py
├── pyproject.toml
├── configs/
├── src/pgdpo_delay/
│   ├── cli.py
│   ├── registry.py
│   ├── configs/
│   ├── core/
│   │   ├── artifacts.py
│   │   ├── estimators.py
│   │   ├── projection.py
│   │   ├── runner.py
│   │   ├── stage1.py
│   │   └── stage2.py
│   └── problems/
│       ├── p1/
│       │   ├── config.py
│       │   ├── oracle.py
│       │   ├── contract.py
│       │   ├── h_refine.py
│       │   ├── calibrate.py
│       │   ├── dp_small.py
│       │   ├── evaluate.py
│       │   └── no_anticipation.py
│       ├── p2/
│       ├── p3/
│       └── p4/
└── tests/
```

`dp_ladder.py`는 아직 존재하지 않는다.

## 2.2 직접 실행한 검증

### Unit tests

```bash
PYTHONPATH=src pytest -q
```

결과:

```text
5 passed
```

### P1 fast verify

```bash
PYTHONPATH=src python main.py verify --problem p1
```

통과했다. P1-U structured/dense recursion, Bellman residual, q-form FOC, anchored identity, recovery-input decoder와 P1-C tiny last-step gate가 모두 통과했다.

### P1 full verify

```bash
PYTHONPATH=src python main.py verify --problem p1 --full
```

통과했으며, handoff의 핵심 수치를 재현했다.

| 항목 | 재현값 |
|---|---:|
| `nRMSE(p_hat)` | 0.5044% |
| raw covariance `q` nRMSE | 8.7768% |
| nested-antithetic OLS `q_anc` nRMSE | 1.7670% |
| `Pi_hat` nRMSE | 0.4067% |
| nested `zeta_hat` nRMSE | 1.0616% |
| Path-A estimator layer RMSE | 5.061e-3 |
| Path-A finite-h layer RMSE | 4.825e-3 |

### P2 fast verify

```bash
PYTHONPATH=src python main.py verify --problem p2
```

통과했다. Dense/mode value, detached curvature, affine term, control, q-form FOC 및 anchored recovery 차이는 대략 `1e-15` 수준이었다.

## 2.3 P1-C r1 로컬 프로파일

현재 `dp_small.py`로 r1을 직접 실행한 결과:

```text
(n_x, n_GH, n_u, L) = (25, 5, 21, 3.0)
runtime              ≈ 8.84 s
peak RSS              ≈ 328 MB
grid-quadrature OOB   ≈ 5.283%
```

이는 설계 MD의 약 12초 추정과 대체로 맞는다. 다만 아래에서 설명하듯 현재 `oob_frac`은 rollout 또는 고정 audit bank의 경계 이탈률이 아니라, **전체 Bellman tensor grid에서 모든 action/GH 노드를 균일 가중한 continuation clipping 비율**이다. 따라서 domain certification 지표로 그대로 사용하면 안 된다.

---

# 3. 현재 코드에서 잘 반영된 부분

## 3.1 P1/P2 reference layer와 solver layer가 분리되어 있다

현재 저장소는 reference/oracle 검산을 먼저 닫고, 공통 torch solver는 별도 skeleton으로 남기는 구조다. 이것은 현재 연구 단계와 맞다.

- `problems/p1/oracle.py`, `contract.py`, `h_refine.py`: exact target 및 estimator contract
- `problems/p2/oracle.py`, `eigencheck.py`, `scaling.py`: structured exact reference
- `core/stage1.py`, `core/stage2.py`: production solver가 들어갈 공통 위치

특히 `core/stage2.py`에는 다음 순서가 문서가 아니라 코드 contract로 남아 있다.

```text
raw (p, zeta, Pi)
→ q_anc = zeta + Pi sigma_ref
→ blockwise projection of (p, q_anc, sym Pi)
→ zeta^N = q_anc^N - Pi^N sigma_ref
→ local solve
→ r_num
```

이 순서는 Theorems 5.1–5.3과 맞다.

## 3.2 P1-C last-step analytic formula가 별도 함수로 있다

`problems/p1/dp_small.py:30-38`의

```python
exact_last_step_action(cfg, x0, xH)
```

은 마지막 Bellman step의 정확한 clipped minimizer를 반환한다. `registry.py:39-70`에는 SciPy bounded minimization과의 비교 및 tiny-DP gate도 들어 있다.

## 3.3 Terminal expectation에서 interpolation을 제거했다

`dp_small.py:80-83`은 `k=N-1`에서 terminal quadratic expectation을 analytic하게 계산한다. 과거의 가장 큰 last-step linear-interpolation bias는 이 patch로 제거됐다.

## 3.4 Action grid 뒤에 parabolic sub-grid refinement가 있다

`dp_small.py:89-109`은 coarse action argmin 주변 세 점으로 parabola를 맞추고 sub-grid vertex를 선택한다. Bound 후보도 포함한다. 이 구조 자체는 ladder 전에 필요한 올바른 개선이다.

## 3.5 Action readout 역할을 나누려는 방향이 반영되어 있다

- `dp_action_label_at`: nearest-node action
- `dp_action_interpolated_at`: multilinear action

이라는 분리가 이미 들어 있다. 다만 아래에서 설명하듯 **action과 regime label 자체를 더 명확히 분리해야 하며**, off-grid canonical policy로 쓸 `dp_action_reoptimized_at`은 아직 없다.

## 3.6 YAML validation과 package resource가 구현되어 있다

`problems/p1/config.py`는 paper parameter fallback을 두지 않고, 명시적인 validation을 수행한다. `pyproject.toml`에도 package data가 등록되어 있다.

---

# 4. Ladder 실행 전에 반드시 고쳐야 할 사항

아래 1–6은 ladder correctness blocker다. 7–9는 강한 engineering recommendation이다.

## 4.1 Blocker 1 — `dp_ladder.py`와 CLI 분기가 아직 없다

설계 문서에는

```bash
python main.py verify --problem p1 --ladder
```

가 제안되어 있지만, 현재 `cli.py`에는 `--ladder`가 없고 registry에도 ladder callable이 없다.

현재 구현 상태:

- `cli.py`: `--full`까지만 지원
- `registry.py`: P1 fast/full verify만 지원
- `problems/p1/dp_ladder.py`: 없음

따라서 ladder는 아직 design-only다.

## 4.2 Blocker 2 — “analytic final step”이 아직 완전한 analytic solve는 아니다

현재 `dp_reference`는 마지막 step에서도 모든 action grid를 순회한다. 달라진 것은 terminal expectation을 interpolation 없이 계산한다는 점이고, 실제 action은 여전히

```text
action grid → argmin → 3-point parabola
```

로 정한다.

즉, `exact_last_step_action`이 존재하지만 `dp_reference`의 policy table을 직접 채우는 데 사용하지 않는다.

로컬 r1 audit 결과:

```text
last-step action RMSE ≈ 4.36e-7
last-step max error   ≈ 6.57e-6
```

이는 매우 작지만 “machine precision exact action”은 아니다. 설계 규율을 정확히 구현하려면 `k=N-1`을 완전히 special-case해야 한다.

### 권장 patch

`k=N-1`에서는 action loop를 생략하고 다음을 직접 계산한다.

```python
u_star = exact_last_step_action(cfg, X0, XH)
V_last = running_state_cost \
       + 0.5*h*R*u_star**2 \
       + 0.5*QT*((mean0 + h*b*u_star - xtar)**2
                  + h*(sigma_state + gu*u_star)**2)
```

그 뒤 `u_star`와 `V_last`를 middle lag axes에 broadcast한다.

효과:

1. last-step action이 실제 analytic formula와 일치
2. G5 gate가 action-grid density와 무관
3. DP 한 step의 action/GH loop 제거
4. 문서의 “analytic final step”과 코드가 정확히 일치

## 4.3 Blocker 3 — float32 policy와 `1e-9` active-set tolerance가 충돌한다

현재 policy table은 `float32`로 저장된다. 그러나 `evaluate.py:108-110`, `regime_disagreement:124-130`은 lower/upper bound를 `1e-9` tolerance로 판정한다.

특히 upper bound는

```text
true bound       = 0.650000000...
float32 stored   = 0.649999976...
```

이므로, `dp_action_label_at`이 Python float로 반환된 뒤

```python
u >= hi - 1e-9
```

를 검사하면 upper-bound action을 interior로 잘못 판정한다.

### 실제 영향 확인

r1, `Np=2000`, seed 1에서 현재 코드와 robust tolerance를 비교했다. 이는 paper result가 아니라 **bug diagnostic**이다.

| 통계 | 현재 `1e-9` 판정 | `1e-6` 판정 |
|---|---:|---:|
| lower occupancy | 24.15% | 24.15% |
| interior occupancy | 75.85% | 46.47% |
| upper occupancy | **0.00%** | **29.38%** |
| mean transitions | 0.1985 | 0.8475 |
| switched fraction | 11.1% | 38.2% |

따라서 현재 상태로 ladder의 occupancy, switching, regime disagreement를 실행하면 결과가 잘못될 수 있다.

### 가장 안전한 해결

action으로 label을 재추정하지 말고, solve 시점에 별도의 `int8` regime table을 저장한다.

```text
-1 = lower active
 0 = interior
+1 = upper active
```

추가 API:

```python
dp_regime_label_at(dp, k, z) -> int
```

보조 tolerance가 필요하다면 dtype에 맞춰

```python
active_tol = max(1e-7, 8*np.finfo(np.float32).eps*max(1.0, abs(lo), abs(hi)))
```

처럼 정한다. 그러나 explicit label table이 우선이다.

## 4.4 Blocker 4 — rollout action과 regime readout이 서로 다른 역할로 분리되지 않았다

현재 설계 MD는 active-set 통계에

```python
active_set_stats(cfg, dp_policy(dp, label=True), ...)
```

를 쓰도록 제안한다. 그런데 이 방식은 label만 nearest-node로 읽는 것이 아니라, **nearest-node action 자체를 rollout dynamics에 사용**한다.

그러면 다음 두 정책이 섞인다.

- action accuracy에 쓰는 multilinear action policy
- switching에 쓰는 nearest-node action policy

두 정책은 state trajectory부터 달라진다. 따라서 “같은 DP reference의 action accuracy와 switching”이라고 해석하기 어렵다.

### 권장 contract

`active_set_stats`를 다음처럼 분리한다.

```python
active_set_stats(
    cfg,
    action_policy=canonical_policy,
    regime_fn=dp_regime_label_at,
    ...,
)
```

- dynamics에는 하나의 canonical action policy만 사용
- 동일한 방문 state에서 regime label만 별도 readout

## 4.5 Blocker 5 — `dp_action_reoptimized_at`이 없다

handoff에서 요구했던 세 readout 중 현재 둘만 있다.

```text
dp_action_label_at
dp_action_interpolated_at
dp_action_reoptimized_at   # missing
```

High-accuracy numerical reference의 canonical off-grid action으로는 local reoptimization이 가장 정직하다.

### 권장 역할

| API | 역할 |
|---|---|
| `dp_action_interpolated_at` | 빠른 smooth readout, rung action-Cauchy 비교 |
| `dp_regime_label_at` | nearest-node active-set label |
| `dp_action_reoptimized_at` | off-grid Bellman objective를 다시 최소화한 canonical audit action |

`dp_action_reoptimized_at`은

1. 현재 off-grid buffer `z`
2. GH quadrature
3. 다음 value table interpolation
4. bounded scalar minimization
5. lower/upper 후보 포함

으로 구현한다. Final paper의 control/reference error와 switching audit에는 가능한 한 reoptimized action을 사용하고, interpolated action은 cheap diagnostic으로 남긴다.

## 4.6 Blocker 6 — 현재 domain rung은 domain과 grid-spacing을 동시에 바꾼다

설계 MD의

```text
r3  = (41, 9, 41), L=3.0
r3' = (41, 9, 41), L=3.5
```

비교는 동일한 `n_x`에서 domain만 넓히므로 공간격자가

```text
dx: 0.1500 → 0.1750
```

로 16.7% 거칠어진다. 따라서 차이가 생기면 그것이

- domain truncation 감소 때문인지
- interpolation grid가 거칠어졌기 때문인지

분리할 수 없다.

### 권장 domain rungs

공간격자를 대략 유지한다.

| 목적 | 제안 rung | dx |
|---|---|---:|
| r2 domain screening | `(n_x,n_GH,n_u,L)=(39,7,31,3.5)` | 0.1842 |
| r2 base | `(33,7,31,3.0)` | 0.1875 |
| r3 final domain | `(47,9,41,3.5)` | 0.1522 |
| r3 base | `(41,9,41,3.0)` | 0.1500 |

따라서 기존 `r3'=(41,...,L=3.5)`는 **순수 domain gate**로 쓰지 않는 것을 권한다. 사용한다면 “combined domain/coarser-grid sensitivity”라고만 부른다.

## 4.7 Strong recommendation — 현재 `oob_frac`의 의미를 분리해야 한다

`dp_small.py:40-53, 72, 87, 115`의 `oob_frac`은 전체 tensor grid의 모든 candidate action과 GH node에서 발생한 first-coordinate clipping 비율의 평균이다.

이는 다음과 다르다.

1. fixed audit-state bank에서의 one-step quadrature OOB
2. canonical policy rollout에서의 state boundary hit
3. off-grid policy/value readout OOB

또한 서로 다른 `L`에서는 tensor grid 자체의 state set이 달라지므로 현재 `dp["oob_frac"]`를 직접 비교하는 것도 완전히 공정하지 않다.

### 권장 명칭과 지표

```text
grid_quadrature_oob_frac       # 현재 값, diagnostic only
audit_bank_quadrature_oob_frac # fixed common bank, G4용
rollout_boundary_hit_frac      # common CRN rollout
readout_oob_frac               # value/action readout
min_boundary_margin            # rollout state의 L-|z_i|
```

G4는 fixed-bank 및 rollout 지표를 사용하고, `grid_quadrature_oob_frac`은 supplementary diagnostic으로만 둔다.

## 4.8 Strong recommendation — parabolic policy와 stored value의 일치 guard를 명시한다

현재 r1에서는 `v_par > f_grid_min`인 inconsistency가 관측되지 않았지만, 코드 구조상 다음 invariant를 명시하는 것이 안전하다.

```python
use_parabolic = (
    positive_curvature
    & vertex_in_box
    & (v_parabolic <= f_grid_min + value_tol)
)

u_store = where(use_parabolic, vertex, u_grid_min)
V_store = where(use_parabolic, v_parabolic, f_grid_min)
```

즉 action과 value가 반드시 같은 candidate를 가리키게 한다.

## 4.9 Strong recommendation — canonical YAML의 shadowing을 없앤다

현재 동일한 canonical YAML이 두 위치에 중복되어 있다.

```text
configs/p1/main.yaml
src/pgdpo_delay/configs/p1/main.yaml
```

파일들은 현재 동일하지만, `load_config("main")`은 repo root에서 실행할 때 `./configs/p1/main.yaml`을 package resource보다 먼저 읽는다. 따라서 두 복사본이 나중에 달라지면 “package YAML이 single source”라는 규율이 깨진다.

### 권장

- canonical `main.yaml`, `dp_small.yaml`, `dp_ladder.yaml`은 package resource에만 둔다.
- user-derived config는 `configs/derived/p1/` 또는 명시적 path로 둔다.
- 최소한 test에서 root/package hash equality를 강제한다.

---

# 5. Ladder 결정사항에 대한 최종 추천

## 5.1 Gate threshold

기존 제안의 기본 scale은 합리적이므로 유지하되, gate 정의를 아래처럼 보강한다.

### 권장 최종 gate

| Gate | 최종 권고 |
|---|---|
| G0 solver integrity | terminal formula, bounds, finite value, policy-value consistency, regime-label integrity 통과 |
| G1 action | pooled/all-seed `RMSE_u(r2,r3) < 5e-3`; `p95(|Δu|)`도 함께 보고 |
| G2 value | `RMSE(V2-V3)/RMS(V3) < 1e-3` |
| G2b Cauchy trend | action/value의 `Δ23`가 `Δ12`보다 커지지 않아야 함; rate 주장은 하지 않음 |
| G3 regime | 각 occupancy 차이 < 1 pp, label disagreement < 2% |
| G3b switching | switched-fraction 차이 < 1 pp, jointly-switching paths의 paired first-switch MAE ≤ `h` |
| G4 domain | matched-`dx` domain rung과 action RMSE < 5e-3; value rel diff < 1e-3; fixed-bank/rollout OOB 비증가 |
| G5 repeated audit | final 3개 audit seed가 모두 hard gates를 통과 |

### 추가 보고값

다음은 우선 report-only로 두고 pilot 후 hard threshold 여부를 정한다.

- `p95`와 `max |Δu|`
- mean transition-count 차이
- empirical order `beta_hat`
- grid-level convexity violation fraction
- Bellman residual quantiles

Switching surface 부근에서는 작은 state 이동이 큰 action jump를 만들 수 있으므로 `max |Δu|`를 곧바로 hard gate로 쓰는 것은 권하지 않는다.

## 5.2 Domain sensitivity 위치

### 최종 추천

1. **개발 중:** r2 matched-spacing domain screening 실행
   
   ```text
   r2D = (39,7,31,L=3.5)
   ```

2. **최종 paper certification:** r3 matched-spacing domain run을 한 번 실행

   ```text
   r3D = (47,9,41,L=3.5)
   ```

3. 기존 `r3'=(41,9,41,L=3.5)`는 primary G4에서 제외

이렇게 해야 “domain을 넓혀도 interior reference가 변하지 않는다”는 해석이 가능하다.

## 5.3 Audit bank와 seed 예산

### 개발 단계

```text
N_s = 256
N_p = 2000
seed = 1개
```

### 최종 certification

```text
state bank: 256 × 3 seeds
rollout:    2000 × 3 seeds
```

DP solve는 deterministic이므로 seed마다 DP를 다시 풀 필요가 없다. 각 rung은 한 번만 풀고, evaluation bank만 3개 사용한다.

### State bank 구성 권고

한 종류의 `make_hist`만 모든 시점에 재사용하지 말고, seed당 256개를 다음처럼 나눈다.

```text
128: generic interior history templates
128: fixed generator policy로 만든 reachable rollout snapshots
```

모든 rung과 모든 domain run에 정확히 같은 bank를 사용한다.

이유:

- generic bank는 off-trajectory interior smoothness를 본다.
- reachable bank는 실제 rollout에서 중요한 영역을 본다.
- initial-history template만 모든 k에 놓으면 후반 시점의 실제 reachable geometry를 충분히 반영하지 못할 수 있다.

## 5.4 Storage 결정

### 최종 추천

> **u-streaming + 2-value-buffer를 기본으로 하고, policy memmap은 optional/auto fallback으로 둔다.**

구체적으로:

- candidate objective 계산: float64
- value recursion: current/next 2-buffer
- policy table: float32
- regime table: int8
- audit-bank value readout: 각 k에서 즉시 저장
- full value tensor history: `store="all"` debugging에서만
- policy memmap: 예상 peak가 사용 가능 RAM의 예를 들어 70%를 넘을 때 자동 활성화
- raw `.dat`보다 `numpy.lib.format.open_memmap` 기반 `.npy` 권장

현재 설계의 r3 메모리 추정은 대략 다음과 같다.

| 항목 | r3 대략 크기 |
|---|---:|
| full `tots`, float32 | 442 MB |
| all policy tables, float32 | 162 MB |
| all value tables, float64 | 345 MB |
| one value/work tensor, float64 | 21.6 MB |

Full `tots`와 all-values를 동시에 유지하지 않으면 peak를 크게 줄일 수 있다.

---

# 6. 권장 `dp_ladder.py` 구조

## 6.1 별도 ladder YAML

새 package resource를 권한다.

```text
src/pgdpo_delay/configs/p1/dp_ladder.yaml
```

예시:

```yaml
base_variant: dp_small
rungs:
  - {name: r1, nx: 25, ngh: 5, nu: 21, L: 3.0}
  - {name: r2, nx: 33, ngh: 7, nu: 31, L: 3.0}
  - {name: r3, nx: 41, ngh: 9, nu: 41, L: 3.0}
domain_rungs:
  - {name: r2D, nx: 39, ngh: 7, nu: 31, L: 3.5}
  - {name: r3D, nx: 47, ngh: 9, nu: 41, L: 3.5}
audit:
  state_count_per_seed: 256
  rollout_count_per_seed: 2000
  seeds: [101, 211, 307]
  interior_radius: 2.0
gates:
  action_rmse: 0.005
  value_rel_rmse: 0.001
  occupancy_pp: 1.0
  regime_disagreement: 0.02
  first_switch_steps: 1.0
storage:
  mode: audit
  candidate_streaming: true
  policy_memmap: auto
```

## 6.2 권장 public API

```python
# problems/p1/dp_ladder.py

def run(config="dp_ladder", outdir="outputs/verify/p1/ladder") -> dict:
    ...
```

내부 함수는 다음처럼 나눈다.

```python
@dataclass(frozen=True)
class RungSpec:
    name: str
    n_x: int
    n_gh: int
    n_u: int
    L: float


def build_audit_banks(cfg, ladder_cfg): ...
def solve_rung(cfg, spec, banks, outdir): ...
def evaluate_rung(cfg, dp, banks): ...
def compare_rungs(left, right): ...
def compare_domain(base, enlarged): ...
def apply_gates(comparisons, gate_cfg): ...
def write_ladder_artifacts(...): ...
```

## 6.3 실행 흐름

```text
1. load dp_small + dp_ladder YAML
2. validate odd nx/ngh/nu, H=3, common bounds/h/N
3. build and save fixed audit banks once
4. solve r1, r2, r3
5. evaluate each rung on identical banks
6. compute r1-r2 and r2-r3 Cauchy differences
7. run r2D screening
8. run r3D for final certification
9. apply gates
10. save manifest, rung metrics, pair metrics, gate verdict
11. nonzero exit or explicit FAIL status when a hard gate fails
```

## 6.4 CLI 연결

현재 flat CLI를 유지하려면 최소 변경은 다음과 같다.

```python
ap.add_argument("--ladder", action="store_true")
```

규율:

- `--ladder`는 `verb=verify`, `problem=p1`에서만 허용
- `--full`과 동시 사용 금지
- `--all --ladder` 금지
- output은 `outputs/verify/p1/ladder/`

Registry에는 일반 verify와 분리된 callable을 둔다.

```python
PROBLEM_REGISTRY = {
    "p1": {
        "verify": _p1_verify,
        "ladder": _p1_ladder,
    },
    ...
}
```

Ladder를 generic `solver="exact-reference"` manifest로 기록하지 않는다. 예:

```text
method = p1c-dp-ladder
solver = numerical-tensor-dp
reference_status = preliminary | certified
```

---

# 7. `dp_small.py` 권장 수정안

## 7.1 함수 signature

```python
def dp_reference(
    cfg,
    *,
    n_x=None,
    n_gh=None,
    n_u=None,
    L=None,
    bounds=None,
    store="audit",              # "audit" | "all"
    candidate_streaming=True,
    policy_memmap=None,
    audit_states=None,
    interpolation="linear",
):
    ...
```

## 7.2 반환 contract

```python
return {
    "policy": ...,
    "regime": ...,
    "value_readouts": ...,
    "values": ... or None,
    "xg": xg,
    "ug": ug,
    "grid_quadrature_oob_frac": ...,
    "runtime": ...,
    "peak_memory_estimate": ...,
    "api_version": P1C_DP_API_VERSION,
    "spec": ...,
}
```

## 7.3 Streaming 구현 시 중요한 점

3-slice streaming을 쓰되 다음 두 가지를 지켜야 한다.

1. candidate objective slice와 parabolic 계산은 float64
2. 최종 선택한 variable action에서 Bellman objective를 한 번 재평가하여 stored action/value consistency 확인

단순히 parabola predicted minimum만 저장하면 continuation objective가 완전한 quadratic이 아닐 때 작은 bias가 남을 수 있다.

## 7.4 Bellman interpolation의 실제 구조

Grid-node Bellman recursion에서는

```text
Z' = (X_{k+1}, X_k, X_{k-1}, X_{k-2})
```

이고 뒤 세 좌표는 기존 grid node에 정확히 놓인다. 따라서 중간 step의 off-grid interpolation은 본질적으로 **새 current coordinate 한 축에 대한 1D interpolation**이다.

이는 좋은 소식이다.

- full 4D cubic interpolation이 필요하지 않다.
- linear interpolation이 ladder에서 실패하면 첫 대안은 first-axis local quadratic/convexity-preserving interpolation이다.
- 무검증 generic cubic spline을 넣을 이유가 없다.

### 실패 시 interpolation upgrade 순서

1. first-axis local quadratic with curvature guard
2. second-difference 기반 convexity-preserving piecewise quadratic
3. 필요할 때만 shape-preserving cubic 계열 검토

---

# 8. Evaluation API 권장 수정

## 8.1 `active_set_stats`는 paired path 정보를 반환해야 한다

현재는 summary scalar만 반환한다. Ladder의 first-switch paired difference를 계산하려면 다음이 필요하다.

```python
return {
    "occ": ...,
    "transitions_mean": ...,
    "switched_frac": ...,
    "first_switch_mean": ...,
    "first_switch_per_path": ...,
    "transition_count_per_path": ...,
}
```

## 8.2 별도 paired switching evaluator

```python
def compare_switching_paired(cfg, policy_a, regime_a, policy_b, regime_b,
                             initial_states, brownian_bank):
    ...
```

보고:

- occupancy difference by regime
- state-bank label disagreement
- rollout regime disagreement by time
- switched-fraction difference
- jointly-switching first-time MAE
- transition-count difference

## 8.3 Batch readout

현재 `dp_policy`는 state마다 Python loop를 돌며 `map_coordinates`를 호출한다. `Np=2000`, `N=15`, 다중 seed에서 불필요하게 느릴 수 있다.

다음 batch API를 권한다.

```python
dp_value_batch(dp, k, Z)
dp_action_interpolated_batch(dp, k, Z)
dp_regime_batch(dp, k, Z)
```

`map_coordinates`에 `(4, Np)` coordinate array를 한 번에 넘기면 된다.

---

# 9. Artifact 규약

권장 output:

```text
outputs/verify/p1/ladder/
├── manifest.json
├── ladder_config.json
├── audit_bank.npz
├── ladder_table.csv
├── pairwise_metrics.csv
├── domain_metrics.csv
├── gate_results.json
├── verdict.txt
├── r1/
│   ├── solve_summary.json
│   ├── policy.npy
│   ├── regime.npy
│   └── audit_readouts.npz
├── r2/
├── r3/
├── r2D/
└── r3D/
```

`ladder_table.csv` 권장 columns:

```text
rung,n_x,n_gh,n_u,L,dx,runtime_s,peak_mem_mb,
grid_quadrature_oob_frac,audit_oob_frac,rollout_boundary_hit_frac,
last_step_rmse,last_step_max_error,bellman_residual_rms,
occ_lower,occ_interior,occ_upper,transitions,switched_frac,first_switch
```

`pairwise_metrics.csv`:

```text
left,right,action_rmse,action_p95,value_rel_rmse,
regime_disagreement,occ_diff_lower_pp,occ_diff_interior_pp,occ_diff_upper_pp,
transition_diff,switched_frac_diff_pp,first_switch_paired_mae
```

Manifest에는 최소 다음을 추가한다.

- code/API version
- git commit
- NumPy/SciPy versions
- dtype
- interpolation mode
- streaming/memmap mode
- rung config hashes
- audit seed list
- bank checksum
- PASS/FAIL

---

# 10. 권장 테스트

## 10.1 Unit tests

1. `exact_last_step_action` vs bounded scalar minimization
2. direct final-step policy table vs exact formula
3. terminal value exactness
4. stored action/value Bellman consistency
5. lower/interior/upper label table 정확성
6. float32 upper-bound regression test
7. batch readout vs scalar readout
8. streaming vs full-`tots` tiny-grid equivalence
9. `store="audit"` vs `store="all"` audit readout equivalence
10. reoptimized action이 bounds 안에 있고 local objective를 개선

## 10.2 Ladder logic tests

1. synthetic metrics에서 gate PASS
2. G1/G2/G3/G4 각각의 단독 FAIL
3. r2-r3만 통과하고 Cauchy trend가 나빠지는 warning
4. domain rung의 dx matching validation
5. same bank checksum across all rungs
6. `--ladder` CLI tiny-rung smoke test
7. failed ladder도 artifacts와 verdict를 남기는지 확인

---

# 11. 구현 순서

## Phase 0 — Correctness blockers

1. final step을 exact formula로 직접 채움
2. explicit int8 regime table 추가
3. active tolerance regression test 추가
4. action policy와 regime readout 분리
5. `dp_action_reoptimized_at` 구현
6. current OOB metric rename 및 fixed-bank boundary audit 추가

## Phase 1 — Memory와 API

7. candidate objective streaming
8. 2-buffer value recursion
9. audit-bank value readout 저장
10. optional policy memmap
11. batch readout
12. P1-C DP API version 추가

## Phase 2 — Ladder engine

13. `dp_ladder.yaml`
14. fixed generic/reachable banks 생성
15. r1/r2/r3 solve/evaluate
16. pairwise comparison
17. matched-spacing domain rungs
18. gate engine
19. artifacts/manifest
20. CLI/registry 연결

## Phase 3 — 최종 실행과 분기

21. 개발 1-seed ladder
22. 실패 원인 분리 sweep
23. final 3-seed audit
24. PASS 시 reference artifact freeze
25. Appendix table/certification sentence 작성

---

# 12. Ladder 실패 시 분기

## G1/G2 실패

바로 전체 알고리즘을 바꾸지 말고 원인을 분리한다.

```text
state grid only: (41,7,31)
GH only:         (33,9,31)
action only:     (33,7,41)
```

가장 큰 변화 축을 찾는다.

- state-grid 병목이면 first-axis convexity-preserving interpolation
- GH 병목이면 `n_GH` 추가
- action 병목이면 `n_u` 또는 bounded local reoptimization 개선

## G3만 실패

- 먼저 explicit regime table과 paired-label contract 재확인
- switching surface 근처 state 비율 확인
- action RMSE는 작고 label mismatch만 크다면 switching-boundary localization 결과로 보고, grid를 추가 정련

## G4 실패

- L=3.5 reference 채택
- 필요하면 matched-spacing L=4.0 rung 추가
- 단순히 same-`n_x` L=3.5 결과만 보고 L을 바꾸지 않음

---

# 13. 논문에서 허용할 명칭

## Ladder PASS 전

> preliminary tensor-grid numerical DP implementation under refinement

## 모든 final gate PASS 후

> refinement-certified high-accuracy tensor-grid DP reference

다만 PASS 후에도 다음 표현은 피한다.

- exact constrained oracle
- proved convergence rate
- H=16 global constrained reference

P1-C small은 어디까지나 `H=3` numerical audit이고, main `H=16` experiment와 직접 수치 비교하지 않는다.

---

# 14. 최종 결정안

아래를 현재의 권장 확정안으로 제안한다.

1. **Gate scale:** 기존 `5e-3 / 1e-3 / 1 pp / 2% / h` 승인. 다만 explicit label, paired first-switch, Cauchy trend를 추가한다.
2. **Domain:** 기존 same-`n_x` r3′ 대신 matched-spacing `r2D=(39,7,31,3.5)`, `r3D=(47,9,41,3.5)` 사용.
3. **예산:** 개발은 1 seed, 최종 인증은 `N_s=256`, `N_p=2000`을 3 seeds에서 평가. DP는 seed마다 재계산하지 않는다.
4. **Bank:** generic history 50% + reachable snapshots 50%.
5. **Storage:** u-streaming/2-buffer가 기본, policy memmap은 optional/auto.
6. **Dtype:** objective/value 계산 float64, policy float32, regime int8.
7. **Canonical off-grid policy:** local reoptimized action. Interpolated action은 cheap Cauchy diagnostic.
8. **실행 순서:** correctness patches가 모두 끝난 뒤 ladder를 돌린다.

---

# 15. 최종 판정

```text
P1-U exact reference       : CLOSED
P1-U estimator contract    : CLOSED
P2 structured reference    : CLOSED
P1-C small DP core         : PARTIALLY CLOSED
P1-C ladder design         : SOUND WITH REVISIONS
P1-C ladder implementation : NOT STARTED
P1-C certification         : NOT YET
Common Stage I/II solver   : SKELETON
P3/P4 reference            : NOT STARTED
```

가장 먼저 수정할 한 가지를 고르면 **explicit regime table + rollout action/regime 분리**다. 현재 상태에서는 upper-bound occupancy와 switching 통계가 실제로 왜곡될 수 있으므로, 이 patch 없이 ladder를 실행하면 rungs 자체가 잘 계산되어도 G3 판정이 신뢰할 수 없다.
