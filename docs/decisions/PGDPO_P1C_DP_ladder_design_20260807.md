# P1-C DP Refinement Ladder — 설계 확정안 (2026-08-07)

- **목적:** P1-C small(H=3) tensor-grid DP의 잔여 수치 오차(중간 step 선형 보간 편향, domain 절단, action 격자)를 정량화하는 refinement ladder + 정확도 gate. 통과 시 "preliminary tensor-grid numerical DP implementation under refinement" 명칭을 "refinement-certified high-accuracy tensor-grid DP reference"로 회복.
- **전제:** last-step 해석적 처리 + parabolic sub-grid refinement는 patch_20260807로 반영 완료(last-step RMSE 6.3e-7, regime 불일치 0%). 본 ladder는 k < N−1 중간 step 오차를 담당.
- **규율 유지:** H=3↔16 수치 직접 비교 금지. 수렴 rate 증명 주장 금지("empirically consistent with …" 수위만).

---

## 1. 사다리 단 (rungs)

| 단 | (n_x, n_GH, n_u) | L | 런타임 | 근거 |
|---|---|---|---|---|
| r1 | (25, 5, 21) | 3.0 | 12 s (실측, 2026-08-07 환경) | 현행 canonical (dp_small.yaml) |
| r2 | (33, 7, 31) | 3.0 | 90 s (실측) | oob 5.63% |
| r3 | (41, 9, 41) | 3.0 | ~6 min (r2 × 4.0 외삽) | 인증 후보 |
| r3′ | (41, 9, 41) | 3.5 | ~6 min | domain 민감도 전용 |

- 실측 스케일링: 이론 복잡도 N·n_u·n_GH·n_x⁴, r2/r1 예측 6.3× vs 실측 7.5× — r3 외삽 신뢰 가능.
- 전체 사다리 총 ~15분(단일 스레드), 비차단 배치. n_u·n_GH 홀수 유지.
- grid 조건: h = δ/taps = 0.0667, N = 15, H = 3, box = [−0.531, 0.650] (V3 calibration 동결, yaml 단일 출처).

## 2. 비교 프로토콜

**공통 평가점 (격자 비종속):**
- 고정 seed로 make_hist 분포에서 N_s = 256 히스토리 상태를 뽑고 interior 필터 ‖z‖_∞ ≤ 2.0 적용 (L=3.0 대비 여유폭 1.0).
- 전 시점 k = 0..N−1에서 평가. 동일 bank를 모든 단·모든 L에 재사용.

**Readout 역할 분리 (기존 규율 계승):**
- action 오차: `dp_action_interpolated_at` (다중선형; 매끈한 궤적 비교 전용)
- regime/점유율/switching: `dp_action_label_at` (nearest-node; active-set 라벨 전용 — 보간 정책은 regime 평균화로 라벨 통계 오염)
- value 오차: `dp_value_at`, 상대화 ΔV_rel = RMSE(V_r − V_{r+1}) / RMS(V_{r+1})

**Rollout 지표 (기존 평가기 재사용):**
- `active_set_stats(cfg, dp_policy(dp, label=True), Np=2000, common seed)` → 점유율 3-벡터, transition 수, first-switch 시각, switched fraction. 단별 재실행 후 인접 단 차이 보고.
- `regime_disagreement`를 bank 상태에서 인접 단 쌍에 적용.
- 단별 oob_frac 기록 (비증가 확인).

**보고 전용 (주장 금지):**
- 경험적 수렴 차수 β̂ = log(Δ₁₂/Δ₂₃)/log(해상도 비) — appendix 각주 수위.

## 3. 정확도 Gate (마지막 쌍 r2–r3 기준)

| Gate | 조건 | 임계 (제안) |
|---|---|---|
| G1 action | RMSE_u(r2, r3) on bank | < ε_u = 5e-3 (box 폭 1.181의 0.42%) |
| G2 value | ΔV_rel(r2, r3) | < 1e-3 |
| G3 regime | \|Δocc\| per regime / label 불일치율 / \|Δ first-switch\| | < 1 pp / < 2% / ≤ h |
| G4 domain | RMSE_u(r3, r3′) < ε_u AND oob 비증가 | L=3.0 채택 근거 |
| G5 상시 | 각 단 last-step exact gate < 1e-6, bounds, terminal exactness | 기존 fast gate 재실행 |

**판정 분기:**
- 전부 통과 → r3 (41,9,41,L=3.0) 인증 채택, 명칭 회복. appendix 사다리 표 + 인증 문장.
- G1/G2 실패 → 선형 보간 병목 신호 → convexity-preserving 보간 교체(§9.3 트랙, 예: 축별 monotone cubic) 선행 후 재사다리.
- G4 실패 → L=3.5 채택, L=4.0 단 추가 후 재판정.

**인증 문장 템플릿 (appendix):**
> Across the final two rungs the interpolated action differs by RMSE X (Y% of the control box width), regime labels agree on Z% of audit states, and rollout occupancy shifts by at most W pp per regime; we therefore adopt the (41, 9, 41, L = 3.0) solution as the refinement-certified DP reference. (수렴 rate 주장 없음.)

## 4. 저장 구조 (엔지니어링)

실측: r3의 병목은 Vs 전 step 저장(362 MB)이 아니라 `tots` 작업 배열(n_u × n_x⁴ float32 = 463 MB).

- `dp_reference(..., store="all"|"snapshots")`: value를 2-buffer(현재/다음)로 진행, 비교용 snapshot(전 k 또는 audit k)만 float32 보관. 기존 시그니처 호환 (기본값 "all" = 현행 동작).
- `tots` u-축 3-slice rolling 스트리밍 옵션: parabolic stencil이 j_c−1, j_c, j_c+1만 필요하므로 가능. peak 463 MB → ~70 MB. (running argmin + 이웃 2슬라이스 보관)
- 정책 테이블(r3: 170 MB)은 RAM 기본, `--memmap` 시 `outputs/verify/p1/ladder/<rung>/pol.dat` (float32, memmap).
- r3 peak ~1 GB는 로컬 머신 허용 범위 → memmap은 옵션이지 필수 아님.

## 5. CLI·산출물 (스캐폴드 규약)

```bash
python main.py verify --problem p1 --ladder        # fast/full과 독립, 비차단 배치
```

- `outputs/verify/p1/ladder/`: manifest(config hash·git·seed·API version), 단별 요약 json, `ladder_table.csv`(appendix 표 원천), 판정 요약(gate PASS/FAIL 목록).
- figures/tables는 규약대로 CSV만 소비. 신규 파일: `problems/p1/dp_ladder.py` (드라이버 + gate), registry에 `--ladder` 분기 등록, dp_small.py는 store 옵션만 추가(계약 불변).
- P3-R과 병렬 가능: DP 함수 시그니처·verify gate·평가기 인터페이스 불변.

## 6. 확정 필요한 결정 (사용자)

1. Gate 임계 승인: ε_u = 5e-3 / ΔV_rel 1e-3 / occ 1 pp / 불일치율 2% / first-switch ≤ h
2. L 민감도 위치: r3′만 (제안, +6분) vs r2′ (33,7,31,L=3.5) 저렴 선행 추가 여부
3. 예산: state bank N_s = 256, rollout N_p = 2000, 공통 seed 1개(고정) vs 3-seed 반복
4. r3 저장: u-streaming만으로 충분(제안) vs memmap까지 기본 탑재
