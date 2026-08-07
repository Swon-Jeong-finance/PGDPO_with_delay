# PGDPO Delay Stage-I multi-GPU runner contract (2026-08-07)

## 1. 목적과 범위

Stage-I의 독립 training seed들을 GPU 슬롯 큐로 실행한다. 예를 들어 seed
5개와 GPU 3개를 주면 처음 3개를 동시에 실행하고, 먼저 끝난 GPU에 다음
seed를 즉시 배정한다. 스케줄러·artifact·집계 계층은 problem-independent이며,
현재 production worker 등록은 P1-U부터 시작한다.

이 구현은 P2 eigenbasis 정렬과 P3 boundary-capable chart를 수정하지 않는다.
P2/P3 production 학습 전에 해당 problem-specific correction과 worker 등록이
별도로 필요하다.

## 2. 실행 모델

- seed마다 독립 OS subprocess를 사용한다. thread나 한 프로세스 안의 복수
  CUDA context를 사용하지 않는다.
- 각 GPU에는 동시에 최대 한 subprocess만 배정한다.
- CUDA device는 `cuda:0`처럼 index를 반드시 명시한다. 한 run에서 CPU와
  CUDA 슬롯을 섞지 않는다.
- seed별 model/history/train-noise/validation RNG split은 기존 Stage-I trainer의
  `SeedSequence` 계약을 그대로 사용한다.
- scheduler는 solver/Torch를 import하지 않는다. worker만 PyTorch와 problem
  adapter를 import한다.
- Ctrl-C, `SIGTERM`, `SIGHUP` 시 실행 중 child를 terminate하고, grace timeout
  뒤에도 남으면 kill한다. 실패 staging과 로그는 보존한다.

## 3. 설치와 명령

```bash
python -m pip install -e '.[solver]'
```

GPU 및 artifact 경로만 확인하는 5-seed smoke:

```bash
python main.py run \
  --problem p1 \
  --stage 1 \
  --protocol p1_u_smoke \
  --seeds 1,2,3,4,5 \
  --devices cuda:0,cuda:1,cuda:2 \
  --run-name p1u_smoke_5seed
```

`p1_u_smoke`는 artifact 경로 검사 전용이며 논문 결과로 사용하지 않는다.

P1-U Phase-B production protocol:

```bash
python main.py run \
  --problem p1 \
  --stage 1 \
  --protocol p1_u \
  --seeds 1,2,3 \
  --devices cuda:0,cuda:1,cuda:2 \
  --run-name p1u_phaseb
```

위 `1,2,3`은 실행 예시다. Training seed roster는 protocol YAML에 숨겨 넣지
않고 명령의 `--seeds`에서 명시하며, 논문용 roster는 pilot 전에 별도로 동결한다.

`p1_u`는 다음 예산을 동결한다.

```text
iters=3000, batch=1024, lr=5e-5, hidden=256, num_layers=2
clip_grad=1, log_every=100, val_every=100, val_batch=1024
evaluation Np=50000, common evaluation seed=123, policy batch_size=4096
```

입력은 oldest-to-newest `(B,H+1,3)` token
`[state, kh/T, relative_lag]`이고 모델은 2-layer LSTM 뒤에
`Linear-Tanh-Linear` head를 사용한다. 상세 schema와 checkpoint 호환성은
`PGDPO_P1_STAGE1_feature_model_v2_20260807.md`를 따른다.

실제 subprocess를 시작하지 않고 protocol/fingerprint/경로만 확인하려면
`--dry-run`을 붙인다.

### 실시간 로그

Stage-I CLI는 각 subprocess의 `run.log`를 실행 중에 terminal로 전달한다.
병렬 출력은 다음처럼 seed와 device prefix로 구분된다.

```text
[seed=1 device=cuda:0] [stage1] iter   600  J_train = 0.303565  J_val = 0.301842  *
```

`J_val`은 validation 시점에만 표시하고 `*`는 새 best-validation weight가
선택됐다는 뜻이다. Prefix는 terminal에만 추가되며 seed별 `run.log`에는
원래 trainer 출력이 그대로 저장된다. 따라서 실행이 끝나기 전에도 진행과
best 갱신을 볼 수 있고, 저장 artifact의 로그 형식은 바뀌지 않는다.

## 4. Resume와 overwrite 규칙

중단 후에는 seed roster, protocol, run name을 그대로 두고 실행한다.

```bash
python main.py run \
  --problem p1 --stage 1 --protocol p1_u \
  --seeds 1,2,3 \
  --devices cuda:0,cuda:1,cuda:2 \
  --run-name p1u_phaseb --resume
```

Resume는 `COMPLETE`이며 동일 run fingerprint인 seed만 건너뛴다. 이때 status뿐
아니라 manifest의 seed/problem/method, 실제 config hash, checkpoint의
config/chart/protocol/source/initial-law binding까지 교차검증한다. 불완전하거나
변조됐거나 다른 fingerprint인 seed 디렉터리는 재사용하거나 덮어쓰지 않는다.
설정을 바꿀 때는 새 protocol 이름 또는 새 `--run-name`을 사용한다.

## 5. Artifact 계약

기본 경로는 다음과 같다.

```text
outputs/runs/p1/stage1/<run-name>/
├── run_spec.json
├── run_summary.json
├── per_seed.csv
├── summary.csv
├── summary.json
├── seed1/
│   ├── manifest.json
│   ├── config.json
│   ├── status.json
│   ├── stage1_state.pt
│   ├── stage1_spec.json
│   ├── training_trace.npz
│   ├── metrics.json
│   └── run.log
└── failed/
    └── seedN-<attempt-id>/...
```

Worker는 `status.json=COMPLETE`를 마지막에 쓴다. Scheduler는 필수 파일과
fingerprint를 검증한 뒤 private staging을 `seedN/`으로 원자적으로 이동한다.
체크포인트에는 problem, problem config, chart, protocol, run fingerprint,
source-tree hash와 명시적인 initial-law 규약이 저장된다. Git metadata가 없는
ZIP 실행도 source hash로 구·신 코드의 seed 혼합을 차단한다. Production load는
반드시 expected binding을 주며, 불일치를 거부한다.

`metrics.json`은 flat numeric schema다. P1-U에는 exact Riccati 대비 control
nRMSE, common-random-number paired objective regret와 SE, MC/value anchor,
anchor gap의 paired MC SE, best iteration, clipping fraction,
train/evaluation/total runtime이 들어간다. 필수 metric roster는 protocol별로
고정되며 하나라도 누락되거나 NaN/Inf이면 worker는 COMPLETE를 게시하지 않는다.
manifest에는 NumPy/SciPy/Torch/CUDA/cuDNN 버전과 deterministic/TF32 backend
상태도 기록한다. 현재는 statistical seeded reproducibility이며 CUDA bitwise
determinism을 강제로 켜지는 않는다.

## 6. Seed 집계

모든 seed가 끝나면 자동으로 다음을 생성한다.

- `per_seed.csv`: seed별 모든 metric과 device
- `summary.csv`: metric별 역할, `n`, mean과 적용 가능한 sample SD, SE,
  Student-t 95% CI
- `summary.json`: 동일 통계·불확실성 축과 identity/failure metadata

problem, method, config hash, run fingerprint 또는 metric schema가 seed 사이에서
다르면 집계를 거부한다. 평가 seed 123은 모든 training seed에서 공통으로
사용하므로 평가 bank가 동일하다.

따라서 sample SD/SE/Student-t 95% CI는 **공통 평가 bank에 조건부인 독립
training-seed 변동성**이다. 각 정책의 common-noise paired-path MC SE는
`dJ_se`로 별도 저장한다. `J_exact`, `J_oracle_mc`, `mc_anchor_gap`처럼 모든
training seed가 공유하는 diagnostic은 독립 반복이 아니므로 seed SD/CI를
표시하지 않는다. 3-seed 본문 요약에서는 mean과 sample SD를 우선 사용한다.

## 7. P2/P3 확장 지점

공통 scheduler와 aggregator는 수정하지 않는다. 새 problem은 다음만 추가한다.

1. `configs/stage1/<protocol>.yaml`
2. `core/stage1_run.py`의 torch-free problem snapshot 등록
3. 해당 Stage-I adapter 생성 함수
4. problem-specific reference/benchmark evaluator
5. worker smoke 및 binding 회귀 테스트

P2는 eigenbasis/spectrum ordering correction 후 등록하고, P3는 main chart 규율을
결과 확인 전에 predeclare한 뒤 등록한다.
