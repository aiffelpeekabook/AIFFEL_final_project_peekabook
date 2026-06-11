import wandb
import pandas as pd

api = wandb.Api()

# sweep_id는 대시보드 URL에서 확인
# https://wandb.ai/{entity}/{project}/sweeps/{sweep_id}
sweep = api.sweep("jjeong3150-aiffel/peekabook-crs/sweeps/v5seoftk")

rows = []
for run in sweep.runs:
    rows.append({
        "persona_name":    run.config.get("persona_name"),
        "collection_name": run.config.get("collection_name"),
        "run_index":       run.config.get("run_index"),
        "judge_score":     run.summary.get("judge_score"),
    })

df = pd.DataFrame(rows)

print(df)

# 논문 테이블 형태로 피벗
pivot = (
    df.groupby(["collection_name", "persona_name"])["judge_score"]
    .agg(mean="mean", std="std")
    .round(2)
    .reset_index()
)

# mean ± std 문자열로 합치기
pivot["score"] = pivot.apply(
    lambda r: f"{r['mean']:.2f} ± {r['std']:.2f}", axis=1
)

# 피벗 테이블
final_table = pivot.pivot(
    index="collection_name",
    columns="persona_name",
    values="score"
)

print(final_table)
# persona_name       대학생_문학팬   직장인_SF팬   중년_역사_비문학
# collection_name
# books_intro_48k    8.00 ± 0.50  4.20 ± 0.30   6.00 ± 0.40
# books_merged_48k   ?            ?              ?