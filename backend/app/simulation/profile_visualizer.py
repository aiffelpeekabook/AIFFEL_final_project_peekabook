"""
inspect_viz.py: ChromaDB 사용자 프로파일 검사 및 클러스터 시각화.

원본 노트북: peekareader_sim_multi_v3.ipynb (cell 13, 20)

함수:
- inspect_profile_by_session: SQLite 직접 접근으로 세션별 프로파일 dump
- visualize_profile_clusters: 임베딩을 KMeans + PCA로 2D 시각화 후 PNG 저장
- log_clusters_to_wandb:       (선택) wandb.Image로 W&B에 push
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional


def inspect_profile_by_session(chroma_path: str) -> list[dict]:
    """
    Chroma SQLite를 직접 읽어 세션별 사용자 프로파일을 콘솔에 출력하고 list로 반환함.

    프로파일은 ChromaDB의 embedding_fulltext_search_content 테이블에 JSON 문자열로 저장됨.

    Args:
        chroma_path: Chroma persist_directory 경로 (예: backend/chroma_db_runs/...)

    Returns:
        [{"session_seq": int, "profile": dict | str}, ...]
    """
    db_file = Path(chroma_path) / "chroma.sqlite3"
    if not db_file.exists():
        print(f"[inspect 스킵] Chroma DB 없음: {db_file}")
        return []

    conn = sqlite3.connect(str(db_file))
    cur  = conn.cursor()

    try:
        cur.execute("SELECT id, c0 FROM embedding_fulltext_search_content ORDER BY id")
        rows = cur.fetchall()
    finally:
        conn.close()

    print(f"\n{'='*60}")
    print(f"세션별 유저 프로파일 ({len(rows)}개 세션)")
    print(f"chroma_path: {chroma_path}")
    print(f"{'='*60}")

    profiles: list[dict] = []
    for seq_id, content in rows:
        try:
            data    = json.loads(content)
            profile = data.get("profile", {})
            print(f"\n[세션 {seq_id}]")
            for k, v in profile.items():
                print(f"  {k}: {v}")
            profiles.append({"session_seq": seq_id, "profile": profile})
        except Exception:
            print(f"\n[세션 {seq_id}] {content[:100]}...")
            profiles.append({"session_seq": seq_id, "profile": content})

    return profiles


def visualize_profile_clusters(chroma_path:     str,
                               persona_id:      str,
                               collection_name: str = "user_profile_memory",
                               save_path:       Optional[str] = None) -> Optional[str]:
    """
    Chroma에 저장된 세션별 프로파일 임베딩을 KMeans + PCA로 2D 시각화함.

    엘보우 방법으로 최적 k를 자동 선택.

    Args:
        chroma_path:     Chroma persist_directory
        persona_id:      플롯 제목용
        collection_name: ChromaDB collection 이름 (default 'user_profile_memory')
        save_path:       PNG 저장 경로. None이면 chroma_path 안에 저장함

    Returns:
        저장된 PNG 경로 (또는 임베딩 없으면 None)
    """
    # heavy 의존성은 함수 안에서 import (사용자가 이 함수 안 부르면 sklearn/matplotlib 안 깔려 있어도 OK)
    import chromadb
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    client     = chromadb.PersistentClient(path=chroma_path)
    try:
        collection = client.get_collection(collection_name)
    except Exception as e:
        print(f"[viz 스킵] collection '{collection_name}' 없음: {e}")
        return None

    result     = collection.get(include=["embeddings", "documents"])
    embeddings = result.get("embeddings")
    documents  = result.get("documents")

    if embeddings is None or len(embeddings) == 0:
        print("[viz 스킵] 임베딩 없음 — 시뮬레이션 먼저 실행 필요")
        return None

    vectors = np.array(embeddings)

    # 1) 레이블 추출 (세션 번호 + preferred_genre 앞 10자)
    labels = []
    for i, doc in enumerate(documents):
        try:
            data  = json.loads(doc)
            genre = data.get("profile", {}).get("preferred_genre", f"세션 {i+1}")
        except Exception:
            genre = f"세션 {i+1}"
        labels.append(f"S{i+1}: {genre[:10]}")

    # 2) 엘보우 방법으로 최적 k 자동 선택
    n = len(vectors)
    k_range  = range(2, min(11, n))
    if len(k_range) == 0:
        print(f"[viz 스킵] 세션 수 부족 ({n}개)")
        return None

    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(vectors)
        inertias.append(km.inertia_)

    if len(inertias) < 2:
        best_k = k_range[0]
    else:
        diffs  = [inertias[i] - inertias[i+1] for i in range(len(inertias)-1)]
        best_k = list(k_range)[diffs.index(max(diffs)) + 1]
    print(f"[엘보우 방법] 최적 클러스터 수: {best_k}")

    # 3) 최적 k로 클러스터링
    km          = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_ids = km.fit_predict(vectors)

    # 4) PCA 2D 축소
    pca     = PCA(n_components=2)
    vecs_2d = pca.fit_transform(vectors)

    # 5) 시각화
    fig, ax = plt.subplots(figsize=(14, 9))
    colors  = plt.cm.tab10.colors

    for i, (x, y) in enumerate(vecs_2d):
        c = colors[cluster_ids[i] % len(colors)]
        ax.scatter(x, y, color=c, s=120, zorder=3)
        ax.annotate(
            labels[i], (x, y),
            fontsize=8, xytext=(6, 6), textcoords="offset points",
        )

    ax.set_title(
        f"{persona_id} 세션별 프로파일 클러스터링\n"
        f"(PCA + KMeans, k={best_k}, 총 {n}세션)",
        fontsize=12,
    )
    ax.set_xlabel(f"PC1 (설명력: {pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 (설명력: {pca.explained_variance_ratio_[1]:.1%})")
    plt.tight_layout()

    if save_path is None:
        save_path = str(Path(chroma_path) / f"cluster_{persona_id}.png")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[viz 저장] {save_path}")

    return save_path


def log_clusters_to_wandb(png_path: str, key: str = "cluster_plot") -> None:
    """
    클러스터 PNG를 W&B에 wandb.Image로 push함.
    wandb.run이 None이면 no-op.
    """
    import wandb
    if wandb.run is None:
        return
    wandb.log({key: wandb.Image(png_path)})
