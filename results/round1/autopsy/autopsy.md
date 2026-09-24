# Round-1 autopsy - seed 541098

Seed median drift 14.32 % over 40 scenarios.

| # | trip | dom | dur s | dist m | drift % | pure % | AT m | CT m | speed err m | creep m | stop s | v ratio | turn deg | seed hdg err | scale eng x ekf = eff | main cause |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 40 | S-S4 | Arterial | 30 | 181 | 101.0 | 131.8 | -174.4 | -56.0 | 72.8 | 0.0 | 0 | 1.332 | 6 | 88.8 | 1.152 x 1.000 = 1.152 | SPEED/SCALE (73 m) |
| 9 | S-S2 | Arterial | 75 | 872 | 91.0 | 107.4 | -759.0 | 232.2 | -314.5 | 0.2 | 2 | 0.668 | 171 | 146.7 | 0.912 x 1.000 = 0.912 | SPEED/SCALE (315 m) |
| 35 | S-S4 | Arterial | 75 | 466 | 47.6 | 47.6 | 140.5 | -171.4 | 313.6 | 0.0 | 0 | 1.731 | 227 | 104.0 | 1.250 x 1.000 = 1.250 | SPEED/SCALE (314 m) |
| 38 | S-S4 | Arterial | 60 | 932 | 34.6 | 36.3 | -316.4 | -62.2 | -343.6 | 0.0 | 0 | 0.669 | 12 | 33.2 | 1.250 x 1.000 = 1.250 | SPEED/SCALE (344 m) |
| 37 | S-S4 | Arterial | 30 | 678 | 32.7 | 34.9 | -221.5 | -0.8 | -264.4 | 0.0 | 0 | 0.648 | 4 | 1.4 | 1.250 x 1.000 = 1.250 | SPEED/SCALE (264 m) |

## Double speed-scale check (T9)
Across all 40 scenarios: median engine scale 1.201, median effective (engine x ekf) 1.201. If effective differs from engine by more than ~2 %, the correction is being applied twice.

Plain-text maths: speed_err_m = sum(v_est - v_can) * 0.1 ; creep_m = sum(v_est where v_can < 0.3) * 0.1
