# P1 end-to-end output contract (schema 2)

## 목적

P1의 Stage-I LSTM-DPO, Stage-II PGDPO, P1-U exact Riccati 또는 P1-C
paired baseline을 한 형식으로 비교한다. 구현은
`reporting/p1_output_contract.py`에 있다.

## 핵심 분리

- `seed_results`: 학습 seed마다 달라지는 policy/objective/KKT/runtime 지표
- `shared_diagnostics`: 고정 evaluation bank에서 모든 학습 seed가 공유하는
  exact-oracle/value anchor
- `metric_schema`: 각 metric의 scope와 uncertainty role
- `problem_config_hash`: Stage I, Stage II, reference가 동일한 P1 dynamics,
  grid, initial-law 설정을 사용했는지 묶는 공통 scientific-config identity

따라서 exact `J_exact`, `J_oracle_mc`, MC anchor gap을 seed 수만큼 복제하여
가짜 seed SD/CI를 만들지 않는다. `dJ_se`는 한 policy 안에서 paired path로
계산한 MC SE이고, seed 간 sample SD와 별개의 불확실성 축이다.
방법별 optimizer/protocol `config_hash`는 서로 달라도 되지만, 각 seed와
reference의 `problem_config_hash`는 top-level 값과 반드시 같아야 한다.
이 값은 `problems.p1.config.scientific_config_hash(cfg)`로 만들며, 임의의
문자열이나 method fingerprint를 대신 넣지 않는다. Stage-I worker가 이를
manifest/config/status/checkpoint에 기록하고 aggregator가 seed 간 일치를 확인한다.
`from_stage1_aggregate()`는 aggregate에 실제 저장된 값을 사용하며 caller가 다른
문자열로 다시 표시(re-stamp)하는 것을 거부한다. Estimator budget, optimizer,
network/input schema는 이 scientific hash가 아니라 method run fingerprint에 속한다.

## P1-U

최소 learned-policy 비교 지표는 다음이다.

- `J_policy`
- `control_nrmse`
- `dJ_paired`, `dJ_se`

공유 exact 진단은 `J_exact`, `J_oracle_mc`, `mc_anchor_gap`,
`mc_anchor_gap_se`이다.

중요하게, learned warm-up의 BPTT adjoint는 그 learned policy를 평가하는
객체이므로 optimal Riccati adjoint와 직접 nRMSE를 계산하지 않는다. `p`, `q`,
`Pi`, `zeta` estimator nRMSE는 오직 `policy=exact_oracle_affine_feedback`로
동결된 별도 `estimator_audit_bank`에서만 `audit_*` 공유 진단으로 저장한다.
`p_cur`는 manuscript Path-A 좌표이고 `p_nxt`는 exact Euler 진단이므로,
둘 사이 finite-grid action floor도 별도 이름으로 저장한다.

Complete comparison에서 objective 차이는 독립적인 자유 입력이 아니다. Contract는
각 seed에 대해

```text
dJ_paired = J_policy - J_oracle_mc
```

를 검증하고, 공유 P1-U anchor도

```text
mc_anchor_gap = J_oracle_mc - J_exact
```

를 만족해야 한다. 또한 complete artifact의 paired rollout은 반드시
`common_random_numbers=true`여야 한다. 검증에는 부동소수점 합산 순서만 허용하는
작은 absolute/relative tolerance를 사용한다.

## P1-C

Global exact oracle을 주장하지 않는다. 최소 지표는 다음이다.

- common-noise `J_policy`, `J_baseline`, `dJ_paired`, `dJ_se`
- `constraint_violation_rate`, `max_constraint_violation`
- recovery bank와 독립인 `holdout_kkt_rms`

선택적으로 lower/interior/upper occupancy, switch count, first-switch time,
regime disagreement를 같은 seed record에 추가한다.
P1-C에서도 `dJ_paired = J_policy - J_baseline` 산술 관계를 contract가
직접 검증한다.

## Stage II 전용 구분

- `solver_r_num_*`: 동일 projected tuple에 대한 local solver tolerance
- `holdout_kkt_*`: recovery와 독립인 branch bank에서의 통계적 품질
- projection activation/displacement, feasibility, recovery denominator
- runtime 및 exact/inexact solver metadata

현재 P1-U paired rollout의 `Np=50000`을 Stage-II branch state 수로 상속하지
않는다. 이 값이 다른 protocol에서 바뀌더라도 두 예산은 항상 별개다.
Stage II는 action을 만드는 `stage2_recovery_bank`와 그 action을 독립적으로
평가하는 `holdout_kkt_bank`를 별도 기록한다. 두 bank는 각각 `states`, `seed`,
`bank_id`, `M`, `M_out`, `M_in`, `branch_batch_size`를 가지며, holdout에는
`independent_of_recovery=true`가 필수다. 두 bank의 `bank_id`와 `seed`는 모두
달라야 한다. 앞의 통계 예산과 메모리용 `branch_batch_size`는 역할이 다르다.

Stage-II seed record는 projection provenance도 반드시 가진다.

```yaml
projection:
  mode: identity-audit | numerical-product-set
  api_version: ...
  config_hash: ...
```

따라서 activation fraction이 0이더라도 명시적 identity audit인지, 실제 numerical
product-set projector가 연결됐지만 활성화되지 않은 것인지 구분할 수 있다.

## Publication

JSON과 tidy CSV는 private staging에 생성·검증한 뒤 immutable bundle로 이동하고,
마지막에 `current-comparison.json` 포인터 하나만 원자적으로 교체한다. Reader는
JSON schema/content hash, CSV의 JSON 일치, manifest/config hash를 모두 재검증한다.
Stage-I의 기존 `Stage1AggregationResult`는 `from_stage1_aggregate`로 변환하며,
반복 저장된 shared exact diagnostic을 한 번만 남긴다.
