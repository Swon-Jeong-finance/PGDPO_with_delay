# P1 Stage-I feature/model v2 decision (2026-08-07)

## 1. 적용 범위

이 문서는 P1 Stage-I buffer-scan LSTM 정책의 입력, 모델, checkpoint 및
P1-U Phase-B production protocol을 동결한다. P1-C도 같은 P1 adapter의 입력
규약을 사용하지만, 현재 등록된 production multi-seed protocol은 P1-U의
`p1_u`이다. `p1_u_smoke`는 실행·artifact 확인용이며 논문 결과가 아니다.

## 2. 입력 feature v2

물리 state buffer `Z`는 내부적으로 newest-first이지만, LSTM에는
oldest-to-newest 순서로 넣는다. `j=0,...,H`번째 scan token은

```text
[ X_{k-H+j},  k h / T,  (j-H) / H ]
```

이며 전체 입력 shape은 `(B, H+1, 3)`이다. 세 채널의 의미는 다음과 같다.

1. `state_value`: 해당 history tap의 state 값. 별도 scaling은 하지 않는다.
2. `global_current_time_kh_over_T`: 정책이 결정을 내리는 현재 시각. 한
   window 안에서는 모든 token이 같은 값을 가진다.
3. `relative_lag_minus1_to_0`: oldest `-1`에서 current `0`까지의 tap 위치.

따라서 state가 서로 다른 과거 시점에서 왔다는 정보가 current-time 채널에
묻히지 않는다. 등간격 grid에서는 token의 실제 정규화 시각도
`kh/T + (Hh/T) * relative_lag`로 복원할 수 있다. 이 분리는 global decision
time과 window 안의 상대 위치를 각각 명시한다.

Run spec에는 `input_schema.api=p1.stage1_features-v2`, `feat_dim=3`,
`feature_schema=state_global_time_relative_lag_v2` 및 sequence order를 저장한다.
Checkpoint binding에도 같은 schema를 포함하므로 2-feature checkpoint나 다른
tap 순서의 checkpoint를 새 실행으로 resume할 수 없다.

## 3. 모델 v2

정책은 매 호출마다 명시적인 `(H+1)` history window 전체를 scan한다. 물리
시간 step 사이에 LSTM hidden state를 전달하지 않으므로 정책은
`u_k = pol(k, Z_k)`로 닫히며, 동시에 history buffer 자체를 읽으므로
memoryless 정책은 아니다.

Production 구조는 다음과 같다.

```text
LSTM(input=3, hidden=256, num_layers=2)
  -> last scan output
  -> Linear(256, 256) -> Tanh -> Linear(256, 1)
```

LSTM input weight는 Xavier uniform, recurrent weight는 gate별 orthogonal로
초기화한다. 두 LSTM bias의 forget-gate slice를 각각 `0.5`로 두어 effective
forget bias를 `1.0`으로 만든다. 첫 MLP weight는 tanh gain의 Xavier uniform,
마지막 weight는 gain `0.1`의 Xavier uniform이며 bias는 각각 `0`이다.

Checkpoint의 `model_schema`는 `buffer_scan_lstm_mlp_v2`이다. `num_layers`,
head 구조와 initialization schema를 `stage1_spec.json`에 명시하고 load 때
검증한다. 단일 linear head를 사용한 legacy checkpoint는 구조가 다르므로
묵시적으로 변환하지 않고 재학습해야 한다.

## 4. P1-U Phase-B production protocol

Canonical `p1_u` 설정은 다음과 같다.

```text
dtype=float32
iters=3000, batch=1024, lr=5e-5
hidden=256, num_layers=2, clip_grad=1
log_every=100, val_every=100, val_batch=1024
evaluation Np=50000, seed=123, policy batch_size=4096
```

Validation은 training RNG와 분리된 고정 history/Brownian bank를 사용한다.
Checkpoint 선택 규칙은 기존과 동일하게 validation 시점의 `J_val`이 이전
최솟값보다 **엄격히 작을 때** 갱신하며, 최종 iterate가 아니라 그
best-validation weight를 평가한다.

`evaluation.batch_size=4096`은 `Np=50000` policy forward만 나누는 실행용
chunk이다. 초기 history, Brownian path 및 Monte-Carlo 통계 예산은 여전히
50,000개 전체이므로 chunk 크기가 결과의 통계적 정의를 바꾸지 않는다.

Protocol schema는 `2`이며 `num_layers`와 evaluation `batch_size`가 필수다.
이 설정, input schema, 모델 구조 또는 source identity가 달라지면 run
fingerprint가 달라지므로 기존 seed directory와 섞거나 resume하지 않는다.

## 5. 실시간 로그와 저장 로그

Trainer는 출력마다 flush한다. Validation이 없는 줄에는 현재 stochastic
training batch의 `J_train`만, validation 시점에는 고정 bank의 `J_val`도 쓴다.
엄격한 새 validation 최솟값에는 `*`를 붙인다.

```text
[stage1] iter   600  J_train = 0.303565  J_val = 0.301842  *
```

Multi-GPU scheduler는 seed subprocess의 private `run.log`에서 완성된 줄을
실행 중에 읽어 parent terminal로 전달한다. Terminal에만
`[seed=<n> device=<slot>]` prefix를 붙이며, seed별 `run.log` 원문은 변경하지
않는다. 따라서 병렬 seed의 진행을 구분해 볼 수 있으면서 artifact는 단독
worker 실행과 같은 형식을 유지한다.

## 6. 구현 위치

- 입력 adapter: `problems/p1/stage1_torch.py`
- 모델/checkpoint: `core/stage1_models.py`
- training 및 validation 선택: `core/stage1.py`
- protocol/run binding: `core/stage1_run.py`
- GPU slot scheduler와 live log: `core/runner.py`
- canonical protocol: `configs/stage1/p1_u.yaml`
