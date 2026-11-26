import base64
import io
import math
import os
import random
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# ==========================================
# 1. 기본 설정 및 상수
# ==========================================
st.set_page_config(page_title="V-League 몬테카를로 분석기", page_icon="🏐", layout="wide")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/130.0.0.0 Safari/537.36",
    "Referer": "https://kovo.co.kr/",
    "Origin": "https://kovo.co.kr",
}

WEIGHTS = {
    "1": {  # 남자부
        "base_so": 0.65,
        "atk": 2.30,
        "opp_drtg": 1.35,
        "rec": 0.85,
        "srv": 0.14,
        "blk": 0.10,
        "err": 0.08,
        "home": 0.05,
    },
    "2": {  # 여자부
        "base_so": 0.59,
        "atk": 1.95,
        "opp_drtg": 1.45,
        "rec": 1.40,
        "srv": 0.12,
        "blk": 0.11,
        "err": 0.10,
        "home": 0.04,
    },
}

DB_FILENAME = "vleague_betting_db.csv"
DB_COLUMNS = [
    "경기ID", "날짜", "부문", "홈팀", "원정팀", "홈코드", "원정코드",
    "세트핸디기준", "언오버기준",
    "몬테_승패", "몬테_승패확률", "몬테_핸디", "몬테_핸디확률", "몬테_언오버", "몬테_언오버확률",
    "라스_승패", "라스_승패확률", "라스_핸디", "라스_핸디확률", "라스_언오버", "라스_언오버확률",
    "제미_승패", "제미_승패확률", "제미_핸디", "제미_핸디확률", "제미_언오버", "제미_언오버확률",
    "실제_홈세트", "실제_원정세트", "실제_총점", "메모",
]


# ==========================================
# 2. DB 로드 / 저장 (로컬 + GitHub 자동 백업)
# ==========================================
def _get_github_config() -> Tuple[Optional[str], Optional[str], str]:
    try:
        token = st.secrets.get("GITHUB_TOKEN")
        repo = st.secrets.get("GITHUB_REPO") or st.secrets.get("REPO_NAME")
        branch = st.secrets.get("GITHUB_BRANCH", "main")
        return token, repo, branch
    except Exception:
        return None, None, "main"


def load_db() -> pd.DataFrame:
    token, repo, branch = _get_github_config()
    if token and repo:
        url = f"https://api.github.com/repos/{repo}/contents/{DB_FILENAME}"
        gh_headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
        try:
            r = requests.get(url, headers=gh_headers, params={"ref": branch}, timeout=8)
            if r.status_code == 200:
                content = base64.b64decode(r.json()["content"]).decode("utf-8-sig")
                df = pd.read_csv(io.StringIO(content), dtype={"홈코드": str, "원정코드": str})
                for col in DB_COLUMNS:
                    if col not in df.columns:
                        df[col] = np.nan
                return df[DB_COLUMNS]
        except Exception:
            pass

    if os.path.exists(DB_FILENAME):
        df = pd.read_csv(DB_FILENAME, dtype={"홈코드": str, "원정코드": str})
        for col in DB_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan
        return df[DB_COLUMNS]

    return pd.DataFrame(columns=DB_COLUMNS)


def save_db(df: pd.DataFrame) -> str:
    df = df.copy()
    for col in DB_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[DB_COLUMNS]
    csv_text = df.to_csv(index=False, encoding="utf-8-sig")

    with open(DB_FILENAME, "w", encoding="utf-8-sig", newline="") as f:
        f.write(csv_text)

    token, repo, branch = _get_github_config()
    if token and repo:
        url = f"https://api.github.com/repos/{repo}/contents/{DB_FILENAME}"
        gh_headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
        sha = None
        try:
            r_get = requests.get(url, headers=gh_headers, params={"ref": branch}, timeout=8)
            if r_get.status_code == 200:
                sha = r_get.json().get("sha")
            payload = {
                "message": f"Update {DB_FILENAME} ({datetime.now().strftime('%Y-%m-%d %H:%M')})",
                "content": base64.b64encode(csv_text.encode("utf-8-sig")).decode("utf-8"),
                "branch": branch,
            }
            if sha:
                payload["sha"] = sha
            r_put = requests.put(url, headers=gh_headers, json=payload, timeout=10)
            if r_put.status_code in (200, 201):
                return "GitHub 클라우드 백업 완료"
            return f"로컬 저장 완료 (GitHub 오류: {r_put.status_code})"
        except Exception as e:
            return f"로컬 저장 완료 (GitHub 연결 실패: {e})"

    return "로컬 저장 완료 (GitHub Secrets 미설정 - CSV 백업 권장)"


# ==========================================
# 3. KOVO API 수집 모듈
# ==========================================
@st.cache_data(ttl=600)
def fetch_team_records(gender: str, season: str = "023", round_num: int = 0) -> pd.DataFrame:
    url = "https://user-api.kovo.co.kr/stat/team-records"
    params = {
        "page": 0,
        "size": 30,
        "sort": "atsp,desc",
        "gender": gender,
        "round": round_num,
        "seasonCodes": season,
        "leagueCodes": "201",
    }
    resp = requests.get(url, params=params, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    items = resp.json().get("payload", {}).get("content", [])

    rows = []
    for it in items:
        s_cnt = max(int(it.get("s_count", 1) or 1), 1)
        att = max(int(it.get("att", 1) or 1), 1)
        rt = max(int(it.get("rt", 1) or 1), 1)
        rows.append({
            "tcode": str(it["tcode"]),
            "team": it["tsname"],
            "logo": it.get("emblumUrl", ""),
            "games": int(it.get("g_count", 0)),
            "sets": s_cnt,
            "attack_pct": float(it.get("atsp", 0.0)),
            "attack_eff": (it.get("ats", 0) - it.get("ate", 0) - it.get("atf", 0)) / att,
            "receive_eff": (it.get("rs", 0) - it.get("rf", 0)) / rt,
            "serve_ace_ps": float(it.get("ssp", 0.0)),
            "block_ps": float(it.get("bsp", 0.0)),
            "dig_ps": float(it.get("dsp", 0.0)),
            "error_ps": float(it.get("err", 0.0)) / s_cnt,
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=600)
def fetch_schedule(season: str = "023") -> List[Dict]:
    url = "https://user-api.kovo.co.kr/stat/game-schedule"
    resp = requests.get(url, params={"gcode": "001", "seasonCode": season}, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    payload = resp.json().get("payload", [])
    return payload if isinstance(payload, list) else payload.get("content", [])


@st.cache_data(ttl=3600)
def fetch_game_detail(gnum: int, season: str = "023", league_code: str = "201") -> Dict:
    url = f"https://user-api.kovo.co.kr/stat/game-schedule/{gnum}"
    params = {"gcode": "001", "seasonCode": season, "leagueCode": league_code}
    resp = requests.get(url, params=params, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    return resp.json().get("payload", {})


# ==========================================
# 4. L5 / 상대 허용 수비력(DRtg) / 피로도 / H2H
# ==========================================
def extract_match_team_stats(detail: Dict, tcode: str) -> Optional[Dict]:
    game = detail.get("game", {})
    sets = max(int(game.get("hspoint", 0) or 0) + int(game.get("aspoint", 0) or 0), 1)
    tcode_str = str(tcode)

    team_list = detail.get("team", []) or []
    t_row = next((t for t in team_list if str(t.get("tcode", "")) == tcode_str), None)

    if t_row and (t_row.get("att", 0) or 0) > 0:
        return {
            "sets": sets,
            "att": max(int(t_row.get("att", 1)), 1),
            "ats": int(t_row.get("ats", 0) or 0),
            "ate": int(t_row.get("ate", 0) or 0),
            "atf": int(t_row.get("atf", 0) or 0),
            "rt": max(int(t_row.get("rt", 1) or 1), 1),
            "rs": int(t_row.get("rs", 0) or 0),
            "rf": int(t_row.get("rf", 0) or 0),
            "ss": int(t_row.get("ss", 0) or 0),
            "bs": int(t_row.get("bs", 0) or 0),
            "err": int(t_row.get("err", 0) or 0),
        }

    players = [p for p in (detail.get("player", []) or []) if str(p.get("tcode", "")) == tcode_str]
    if not players:
        return None

    return {
        "sets": sets,
        "att": max(sum(int(p.get("att", 0) or 0) for p in players), 1),
        "ats": sum(int(p.get("ats", 0) or 0) for p in players),
        "ate": sum(int(p.get("ate", 0) or 0) for p in players),
        "atf": sum(int(p.get("atf", 0) or 0) for p in players),
        "rt": max(sum(int(p.get("rt", 0) or 0) for p in players), 1),
        "rs": sum(int(p.get("rs", 0) or 0) for p in players),
        "rf": sum(int(p.get("rf", 0) or 0) for p in players),
        "ss": sum(int(p.get("ss", 0) or 0) for p in players),
        "bs": sum(int(p.get("bs", 0) or 0) for p in players),
        "err": sum(int(p.get("err", 0) or p.get("ate", 0) or 0) for p in players),
    }


def build_team_l5_profile(
    tcode: str,
    season_row: pd.Series,
    finished_games: List[Dict],
    target_date_str: str,
    season: str = "023",
    l5_weight: float = 0.45,
) -> Dict:
    tcode_str = str(tcode)
    team_games = [
        g for g in finished_games
        if (str(g.get("hcode")) == tcode_str or str(g.get("acode")) == tcode_str)
        and g.get("gdate", "") <= target_date_str
    ]
    team_games.sort(key=lambda x: (x.get("gdate", ""), int(x.get("gnum", 0))))
    recent = team_games[-5:]

    rest_days = 4
    prev_was_5set = False
    l5_wins, l5_losses = 0, 0
    if recent:
        last_g = recent[-1]
        try:
            d1 = datetime.strptime(target_date_str, "%Y-%m-%d")
            d0 = datetime.strptime(last_g["gdate"], "%Y-%m-%d")
            rest_days = max((d1 - d0).days, 1)
        except Exception:
            rest_days = 4
        prev_was_5set = (int(last_g.get("hspoint", 0)) + int(last_g.get("aspoint", 0))) == 5

        for g in recent:
            is_h = str(g.get("hcode")) == tcode_str
            won = (g["hspoint"] > g["aspoint"]) if is_h else (g["aspoint"] > g["hspoint"])
            if won:
                l5_wins += 1
            else:
                l5_losses += 1

    my_agg, opp_agg = Counter(), Counter()
    top_players = []

    for idx, g in enumerate(recent):
        try:
            detail = fetch_game_detail(int(g["gnum"]), season, str(g.get("leagueCode", "201")))
        except Exception:
            continue
        opp_code = str(g["acode"]) if str(g["hcode"]) == tcode_str else str(g["hcode"])

        my_s = extract_match_team_stats(detail, tcode_str)
        opp_s = extract_match_team_stats(detail, opp_code)
        if my_s:
            for k, v in my_s.items():
                my_agg[k] += v
        if opp_s:
            for k, v in opp_s.items():
                opp_agg[k] += v

        if idx == len(recent) - 1:
            ps = [p for p in (detail.get("player", []) or []) if str(p.get("tcode", "")) == tcode_str]
            ps.sort(key=lambda x: int(x.get("point", 0) or 0), reverse=True)
            for p in ps[:3]:
                top_players.append({
                    "선수명": p.get("pname", ""),
                    "포지션": p.get("position", ""),
                    "득점": int(p.get("point", 0) or 0),
                    "공격성공률(%)": round(float(p.get("atsp", 0.0) or 0.0), 1),
                    "공격시도": int(p.get("att", 0) or 0),
                })

    if my_agg["att"] > 0 and my_agg["sets"] > 0:
        l5_atk = (my_agg["ats"] - my_agg["ate"] - my_agg["atf"]) / my_agg["att"]
        l5_rec = (my_agg["rs"] - my_agg["rf"]) / max(my_agg["rt"], 1)
        l5_srv = my_agg["ss"] / my_agg["sets"]
        l5_blk = my_agg["bs"] / my_agg["sets"]
        l5_err = (my_agg["err"] / my_agg["sets"]) if my_agg["err"] > 0 else float(season_row["error_ps"])
    else:
        l5_atk = float(season_row["attack_eff"])
        l5_rec = float(season_row["receive_eff"])
        l5_srv = float(season_row["serve_ace_ps"])
        l5_blk = float(season_row["block_ps"])
        l5_err = float(season_row["error_ps"])

    opp_atk_allowed = (
        (opp_agg["ats"] - opp_agg["ate"] - opp_agg["atf"]) / opp_agg["att"]
        if opp_agg["att"] > 0
        else float(season_row["attack_eff"])
    )

    w_s = 1.0 - l5_weight
    return {
        "tcode": tcode_str,
        "team": season_row["team"],
        "logo": season_row["logo"],
        "l5_record": f"{l5_wins}승 {l5_losses}패",
        "rest_days": rest_days,
        "prev_was_5set": prev_was_5set,
        "top_players": top_players,
        "season_atk": float(season_row["attack_eff"]),
        "l5_atk": l5_atk,
        "season_rec": float(season_row["receive_eff"]),
        "l5_rec": l5_rec,
        "opp_atk_allowed": opp_atk_allowed,
        "attack_eff": w_s * float(season_row["attack_eff"]) + l5_weight * l5_atk,
        "receive_eff": w_s * float(season_row["receive_eff"]) + l5_weight * l5_rec,
        "serve_ace_ps": w_s * float(season_row["serve_ace_ps"]) + l5_weight * l5_srv,
        "block_ps": w_s * float(season_row["block_ps"]) + l5_weight * l5_blk,
        "error_ps": w_s * float(season_row["error_ps"]) + l5_weight * l5_err,
    }


def get_h2h_summary(finished_games: List[Dict], home_code: str, away_code: str) -> Dict:
    hc, ac = str(home_code), str(away_code)
    h2h = [g for g in finished_games if {str(g.get("hcode")), str(g.get("acode"))} == {hc, ac}]
    h_wins, a_wins, h_sets, a_sets = 0, 0, 0, 0
    logs = []
    for g in sorted(h2h, key=lambda x: x.get("gdate", "")):
        if str(g["hcode"]) == hc:
            hs, as_ = int(g["hspoint"]), int(g["aspoint"])
        else:
            hs, as_ = int(g["aspoint"]), int(g["hspoint"])
        h_sets += hs
        a_sets += as_
        if hs > as_:
            h_wins += 1
        else:
            a_wins += 1
        logs.append(f"{g['gdate']} ({hs}:{as_})")
    return {"h_wins": h_wins, "a_wins": a_wins, "h_sets": h_sets, "a_sets": a_sets, "logs": logs}


# ==========================================
# 5. 풀세트 랠리 단위 몬테카를로 시뮬레이션
# ==========================================
def logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def run_volleyball_mc(
    home: Dict,
    away: Dict,
    league_avg: pd.Series,
    gender: str,
    h2h: Dict,
    home_adj: float,
    away_adj: float,
    ou_line: float,
    set_handicap: float = -1.5,
    n_sims: int = 10000,
    seed: int = 42,
) -> Dict:
    w = WEIGHTS[gender]

    def fatigue_pen(t: Dict) -> float:
        pen = 0.0
        if t["prev_was_5set"] and t["rest_days"] <= 3:
            pen -= 0.06
        if t["rest_days"] <= 2:
            pen -= 0.04
        return pen

    total_h2h_sets = h2h["h_sets"] + h2h["a_sets"]
    h2h_edge = 0.0
    if total_h2h_sets >= 3:
        h2h_edge = max(-0.04, min(0.04, ((h2h["h_sets"] / total_h2h_sets) - 0.5) * 0.12))

    def calc_so(off: Dict, opp: Dict, is_home: bool, adj: float, edge: float) -> float:
        z = logit(w["base_so"])
        z += w["atk"] * (off["attack_eff"] - league_avg["attack_eff"])
        z += w["opp_drtg"] * (opp["opp_atk_allowed"] - league_avg["attack_eff"])
        z += w["rec"] * (off["receive_eff"] - league_avg["receive_eff"])
        z -= w["srv"] * (opp["serve_ace_ps"] - league_avg["serve_ace_ps"])
        z -= w["blk"] * (opp["block_ps"] - league_avg["block_ps"])
        z -= w["err"] * (off["error_ps"] - league_avg["error_ps"])
        z += (w["home"] if is_home else 0.0) + fatigue_pen(off) + edge + adj
        return max(0.35, min(0.85, sigmoid(z)))

    p_so_h = calc_so(home, away, True, home_adj, h2h_edge)
    p_so_a = calc_so(away, home, False, away_adj, -h2h_edge)
    p_brk_h = 1.0 - p_so_a

    random.seed(seed)
    rand = random.random
    scores = Counter()
    totals = []
    h_wins, set_handi_hits, over_hits = 0, 0, 0

    for _ in range(n_sims):
        hs, as_, h_tot, a_tot = 0, 0, 0, 0
        s1_srv = 0 if rand() < 0.5 else 1
        set_no = 1
        while hs < 3 and as_ < 3:
            target = 15 if set_no == 5 else 25
            srv = (0 if rand() < 0.5 else 1) if set_no == 5 else (s1_srv if set_no % 2 == 1 else 1 - s1_srv)
            hp, ap = 0, 0
            while True:
                if srv == 0:
                    if rand() < p_brk_h:
                        hp += 1
                    else:
                        ap += 1
                        srv = 1
                else:
                    if rand() < p_so_h:
                        hp += 1
                        srv = 0
                    else:
                        ap += 1
                if (hp >= target or ap >= target) and abs(hp - ap) >= 2:
                    break
            h_tot += hp
            a_tot += ap
            if hp > ap:
                hs += 1
            else:
                as_ += 1
            set_no += 1

        tot = h_tot + a_tot
        scores[f"{hs}-{as_}"] += 1
        totals.append(tot)
        if hs == 3:
            h_wins += 1
        if (hs - as_ + set_handicap) > 0:
            set_handi_hits += 1
        if tot > ou_line:
            over_hits += 1

    totals_arr = np.array(totals)
    home_sh_prob = set_handi_hits / n_sims
    return {
        "p_so_h": p_so_h,
        "p_so_a": p_so_a,
        "home_win": h_wins / n_sims,
        "away_win": 1.0 - (h_wins / n_sims),
        "home_set_handi": home_sh_prob,
        "away_set_handi": 1.0 - home_sh_prob,
        "over_prob": over_hits / n_sims,
        "under_prob": 1.0 - (over_hits / n_sims),
        "exp_total": float(totals_arr.mean()),
        "med_total": float(np.median(totals_arr)),
        "totals_arr": totals_arr,
        "set_scores": {k: scores[k] / n_sims for k in ["3-0", "3-1", "3-2", "2-3", "1-3", "0-3"]},
    }


# ==========================================
# 6. 백테스팅 적중률 분석 엔진 (개별 / 2자 / 3자 + 50/60/70%+)
# ==========================================
def analyze_vleague_backtest(
    df: pd.DataFrame,
    market_filter: str = "전체",
    division_filter: str = "전체",
    combo_prob_mode: str = "모두 이상 (엄격)",
) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    개별(몬테/라스/제미), 2자 일치(몬+라, 몬+제, 라+제), 3자 만장일치의
    전체 / 50%+ / 60%+ / 70%+ 적중률을 계산.
    """
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), 0

    work = df.copy()
    if division_filter != "전체":
        work = work[work["부문"] == division_filter]

    for c in ["실제_홈세트", "실제_원정세트", "실제_총점", "세트핸디기준", "언오버기준"]:
        work[c] = pd.to_numeric(work[c], errors="coerce")

    completed = work[
        work["실제_홈세트"].notna()
        & work["실제_원정세트"].notna()
        & ((work["실제_홈세트"] + work["실제_원정세트"]) >= 3)
    ].copy()

    if completed.empty:
        return pd.DataFrame(), pd.DataFrame(), 0

    markets = ["승패", "핸디", "언오버"] if market_filter == "전체" else [market_filter]
    records = []

    for _, row in completed.iterrows():
        h_s, a_s = row["실제_홈세트"], row["실제_원정세트"]
        tot = row["실제_총점"]
        h_line, ou_line = row["세트핸디기준"], row["언오버기준"]

        actuals = {
            "승패": "홈승" if h_s > a_s else "원정승",
            "핸디": ("홈핸디" if (h_s - a_s + h_line) > 0 else "원정핸디") if pd.notna(h_line) else None,
            "언오버": (
                "오버" if tot > ou_line else ("언더" if tot < ou_line else None)
            ) if (pd.notna(tot) and tot > 0 and pd.notna(ou_line)) else None,
        }

        for m in markets:
            act = actuals[m]
            if act is None:
                continue

            preds = {}
            for src in ["몬테", "라스", "제미"]:
                d = str(row.get(f"{src}_{m}", "") or "").strip()
                p = pd.to_numeric(row.get(f"{src}_{m}확률"), errors="coerce")
                if d and d not in ("패스", "nan", "None") and pd.notna(p):
                    preds[src] = (d, float(p))

            # 개별 소스 평가
            for src, label in [("몬테", "몬테카를로"), ("라스", "라스(시장)"), ("제미", "제미나이")]:
                if src in preds:
                    d, p = preds[src]
                    records.append({"구분": label, "마켓": m, "확률": p, "적중": int(d == act)})

            # 2자 방향 일치 평가
            pairs = [
                ("몬테", "라스", "2자 일치: 몬테 + 라스"),
                ("몬테", "제미", "2자 일치: 몬테 + 제미나이"),
                ("라스", "제미", "2자 일치: 라스 + 제미나이"),
            ]
            for s1, s2, label in pairs:
                if s1 in preds and s2 in preds and preds[s1][0] == preds[s2][0]:
                    p_comb = min(preds[s1][1], preds[s2][1]) if combo_prob_mode.startswith("모두") else (preds[s1][1] + preds[s2][1]) / 2.0
                    records.append({"구분": label, "마켓": m, "확률": p_comb, "적중": int(preds[s1][0] == act)})

            # 3자 만장일치 평가
            if all(s in preds for s in ["몬테", "라스", "제미"]):
                if preds["몬테"][0] == preds["라스"][0] == preds["제미"][0]:
                    ps = [preds["몬테"][1], preds["라스"][1], preds["제미"][1]]
                    p_comb = min(ps) if combo_prob_mode.startswith("모두") else float(np.mean(ps))
                    records.append({"구분": "3자 만장일치 (몬+라+제)", "마켓": m, "확률": p_comb, "적중": int(preds["몬테"][0] == act)})

    if not records:
        return pd.DataFrame(), pd.DataFrame(), len(completed)

    rec_df = pd.DataFrame(records)
    group_order = [
        "몬테카를로",
        "라스(시장)",
        "제미나이",
        "2자 일치: 몬테 + 라스",
        "2자 일치: 몬테 + 제미나이",
        "2자 일치: 라스 + 제미나이",
        "3자 만장일치 (몬+라+제)",
    ]

    def fmt_rate(sub: pd.DataFrame) -> str:
        n = len(sub)
        if n == 0:
            return "-"
        hits = int(sub["적중"].sum())
        return f"{hits / n * 100:.1f}% ({hits}/{n})"

    # 1) 구간별 적중률 표 (전체 / 50%+ / 60%+ / 70%+)
    tier_rows = []
    for g in group_order:
        sub = rec_df[rec_df["구분"] == g]
        tier_rows.append({
            "구분": g,
            "전체": fmt_rate(sub),
            "50% 이상": fmt_rate(sub[sub["확률"] >= 50.0]),
            "60% 이상": fmt_rate(sub[sub["확률"] >= 60.0]),
            "70% 이상": fmt_rate(sub[sub["확률"] >= 70.0]),
        })
    tier_df = pd.DataFrame(tier_rows)

    # 2) 마켓별 상세 적중률 표
    mkt_rows = []
    for g in group_order:
        sub = rec_df[rec_df["구분"] == g]
        mkt_rows.append({
            "구분": g,
            "승패": fmt_rate(sub[sub["마켓"] == "승패"]),
            "세트핸디": fmt_rate(sub[sub["마켓"] == "핸디"]),
            "언오버": fmt_rate(sub[sub["마켓"] == "언오버"]),
        })
    mkt_df = pd.DataFrame(mkt_rows)

    return tier_df, mkt_df, len(completed)


# ==========================================
# 7. 사이드바 공통 설정 및 데이터 로딩
# ==========================================
st.title("🏐 V-리그 몬테카를로 분석 & 백테스팅 허브")

with st.sidebar:
    st.header("경기 및 베트맨 기준점")
    gender_label = st.radio("부문 선택", ["남자부", "여자부"], horizontal=True)
    gender = "1" if gender_label == "남자부" else "2"
    season = st.text_input("시즌 코드", value="023")

    try:
        df_teams = fetch_team_records(gender, season)
        all_sched = fetch_schedule(season)
    except Exception as e:
        st.error(f"KOVO API 호출 실패: {e}")
        st.stop()

    div_sched = [g for g in all_sched if str(g.get("gender")) == gender]
    finished = [g for g in div_sched if (int(g.get("hspoint", 0) or 0) + int(g.get("aspoint", 0) or 0)) >= 3]
    upcoming = [g for g in div_sched if (int(g.get("hspoint", 0) or 0) + int(g.get("aspoint", 0) or 0)) < 3]

    code_to_name = dict(zip(df_teams["tcode"], df_teams["team"]))
    name_to_code = dict(zip(df_teams["team"], df_teams["tcode"]))

    mode = st.radio("매치업 선택", ["예정 경기 불러오기", "직접 팀 선택"], horizontal=True)
    target_date = datetime.now().strftime("%Y-%m-%d")

    if mode == "예정 경기 불러오기" and upcoming:
        upcoming_sorted = sorted(upcoming, key=lambda x: (x.get("gdate", ""), x.get("gstime", "")))[:14]
        options = [
            f"{g['gdate']} {g.get('gstime','')} | {code_to_name.get(str(g['hcode']), g['hname'])} vs {code_to_name.get(str(g['acode']), g['aname'])}"
            for g in upcoming_sorted
        ]
        sel_idx = st.selectbox("경기 선택", range(len(options)), format_func=lambda i: options[i])
        sel_game = upcoming_sorted[sel_idx]
        home_code, away_code = str(sel_game["hcode"]), str(sel_game["acode"])
        target_date = sel_game.get("gdate", target_date)
    else:
        team_names = df_teams["team"].tolist()
        h_name = st.selectbox("홈 팀", team_names, index=0)
        a_name = st.selectbox("원정 팀", team_names, index=min(1, len(team_names) - 1))
        home_code, away_code = name_to_code[h_name], name_to_code[a_name]

    st.divider()
    st.subheader("베트맨 풀세트 기준점")
    set_handicap = st.selectbox(
        "홈 세트 핸디캡",
        options=[-2.5, -1.5, 1.5, 2.5],
        index=1,
        format_func=lambda x: f"홈 {x:+.1f} 세트",
    )
    default_ou = 183.5 if gender == "1" else 179.5
    ou_line = st.number_input("총득점 언오버 기준점", value=default_ou, step=0.5)
    l5_weight = st.slider("최근 5경기(L5) 반영 비중", 0.0, 0.8, 0.45, 0.05)
    n_sims = st.select_slider("시뮬레이션 횟수", [5000, 10000, 20000], value=10000)

    st.divider()
    st.subheader("핵심 결장 / 컨디션 보정")
    h_out_foreign = st.checkbox("홈 외국인/에이스 결장 (-0.16)")
    h_out_setter = st.checkbox("홈 주전 세터 결장 (-0.12)")
    a_out_foreign = st.checkbox("원정 외국인/에이스 결장 (-0.16)")
    a_out_setter = st.checkbox("원정 주전 세터 결장 (-0.12)")
    home_adj = (-0.16 if h_out_foreign else 0.0) + (-0.12 if h_out_setter else 0.0)
    away_adj = (-0.16 if a_out_foreign else 0.0) + (-0.12 if a_out_setter else 0.0)

if home_code == away_code:
    st.warning("홈 팀과 원정 팀을 서로 다르게 선택해 주세요.")
    st.stop()

# 시뮬레이션 계산
with st.spinner("KOVO 최근 5경기 박스스코어 및 시뮬레이션 계산 중..."):
    h_row = df_teams[df_teams["tcode"] == home_code].iloc[0]
    a_row = df_teams[df_teams["tcode"] == away_code].iloc[0]

    home_prof = build_team_l5_profile(home_code, h_row, finished, target_date, season, l5_weight)
    away_prof = build_team_l5_profile(away_code, a_row, finished, target_date, season, l5_weight)
    h2h = get_h2h_summary(finished, home_code, away_code)
    league_avg = df_teams[["attack_eff", "receive_eff", "serve_ace_ps", "block_ps", "error_ps"]].mean()

    sim = run_volleyball_mc(
        home_prof, away_prof, league_avg, gender, h2h,
        home_adj, away_adj, ou_line, set_handicap, n_sims=n_sims
    )

# 몬테카를로 방향 & 확률 자동 추출
mc_ml_dir = "홈승" if sim["home_win"] >= 0.5 else "원정승"
mc_ml_prob = round(max(sim["home_win"], sim["away_win"]) * 100, 1)
mc_hd_dir = "홈핸디" if sim["home_set_handi"] >= 0.5 else "원정핸디"
mc_hd_prob = round(max(sim["home_set_handi"], sim["away_set_handi"]) * 100, 1)
mc_ou_dir = "오버" if sim["over_prob"] >= 0.5 else "언더"
mc_ou_prob = round(max(sim["over_prob"], sim["under_prob"]) * 100, 1)

# ==========================================
# 8. 3개 메인 탭 구성
# ==========================================
tab_sim, tab_prompt, tab_track = st.tabs([
    "🏐 매치업 시뮬레이션",
    "✨ 제미나이 프롬프트 생성",
    "📊 예측 기록 & 백테스팅",
])

# ------------------------------------------
# TAB 1: 매치업 시뮬레이션
# ------------------------------------------
with tab_sim:
    col_h, col_vs, col_a = st.columns([4, 2, 4])
    with col_h:
        if home_prof["logo"]:
            st.image(home_prof["logo"], width=64)
        st.subheader(f"🏠 {home_prof['team']} (홈)")
        badge_h = "⚠️ 직전 5세트 접전" if home_prof["prev_was_5set"] else "체력 정상"
        st.caption(f"L5: {home_prof['l5_record']} | 휴식일: {home_prof['rest_days']}일 | {badge_h}")

    with col_vs:
        st.markdown("<h3 style='text-align:center; margin-top:25px;'>VS</h3>", unsafe_allow_html=True)
        st.markdown(
            f"<p style='text-align:center; font-size:13px;'>{target_date}<br>시즌 H2H <b>{h2h['h_wins']}승 {h2h['a_wins']}패</b> (세트 {h2h['h_sets']}:{h2h['a_sets']})</p>",
            unsafe_allow_html=True,
        )

    with col_a:
        if away_prof["logo"]:
            st.image(away_prof["logo"], width=64)
        st.subheader(f"✈️ {away_prof['team']} (원정)")
        badge_a = "⚠️ 직전 5세트 접전" if away_prof["prev_was_5set"] else "체력 정상"
        st.caption(f"L5: {away_prof['l5_record']} | 휴식일: {away_prof['rest_days']}일 | {badge_a}")

    st.divider()

    # 베트맨 풀세트 3대 마켓 KPI
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric(f"{home_prof['team']} 승률", f"{sim['home_win']:.1%}", f"SO {sim['p_so_h']:.1%}")
    k2.metric(f"{away_prof['team']} 승률", f"{sim['away_win']:.1%}", f"SO {sim['p_so_a']:.1%}")
    k3.metric(
        f"홈 {set_handicap:+.1f} 세트",
        f"{sim['home_set_handi']:.1%}",
        f"원정 {-set_handicap:+.1f}: {sim['away_set_handi']:.1%}",
    )
    k4.metric(f"오버 ({ou_line})", f"{sim['over_prob']:.1%}", f"기대 {sim['exp_total']:.1f}점")
    k5.metric(f"언더 ({ou_line})", f"{sim['under_prob']:.1%}", f"중앙값 {sim['med_total']:.0f}점")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 세트스코어 확률 분포")
        df_sc = pd.DataFrame({
            "세트스코어": list(sim["set_scores"].keys()),
            "확률(%)": [round(v * 100, 1) for v in sim["set_scores"].values()],
        })
        fig_bar = px.bar(df_sc, x="세트스코어", y="확률(%)", text="확률(%)")
        fig_bar.update_traces(textposition="outside")
        fig_bar.update_layout(height=330, margin=dict(t=20, b=20))
        st.plotly_chart(fig_bar, use_container_width=True)

    with c2:
        st.markdown("#### 총득점 분포 vs 언오버 기준점")
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=sim["totals_arr"], nbinsx=40, name="시뮬레이션 총득점"))
        fig_hist.add_vline(x=ou_line, line_dash="dash", line_color="red", annotation_text=f"기준점 {ou_line}")
        fig_hist.add_vline(x=sim["exp_total"], line_dash="dot", line_color="green", annotation_text=f"평균 {sim['exp_total']:.1f}")
        fig_hist.update_layout(height=330, margin=dict(t=20, b=20), xaxis_title="합산 총득점", yaxis_title="빈도")
        st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("#### 팀 전력 비교 (시즌 vs L5 & 상대 허용 수비력)")
    comp_df = pd.DataFrame({
        "지표": [
            "공격효율 (시즌)", "공격효율 (L5)", "상대 허용 공격효율 (L5 DRtg, 낮을수록 수비 강함)",
            "리시브효율 (시즌)", "리시브효율 (L5)", "세트당 서브 에이스", "세트당 블로킹", "세트당 범실",
        ],
        home_prof["team"]: [
            f"{home_prof['season_atk']:.1%}", f"{home_prof['l5_atk']:.1%}", f"{home_prof['opp_atk_allowed']:.1%}",
            f"{home_prof['season_rec']:.1%}", f"{home_prof['l5_rec']:.1%}",
            f"{home_prof['serve_ace_ps']:.2f}", f"{home_prof['block_ps']:.2f}", f"{home_prof['error_ps']:.2f}",
        ],
        away_prof["team"]: [
            f"{away_prof['season_atk']:.1%}", f"{away_prof['l5_atk']:.1%}", f"{away_prof['opp_atk_allowed']:.1%}",
            f"{away_prof['season_rec']:.1%}", f"{away_prof['l5_rec']:.1%}",
            f"{away_prof['serve_ace_ps']:.2f}", f"{away_prof['block_ps']:.2f}", f"{away_prof['error_ps']:.2f}",
        ],
    })
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    p1, p2 = st.columns(2)
    with p1:
        st.markdown(f"**{home_prof['team']} 직전 경기 득점 Top 3**")
        st.dataframe(pd.DataFrame(home_prof["top_players"]), use_container_width=True, hide_index=True)
    with p2:
        st.markdown(f"**{away_prof['team']} 직전 경기 득점 Top 3**")
        st.dataframe(pd.DataFrame(away_prof["top_players"]), use_container_width=True, hide_index=True)


# ------------------------------------------
# TAB 2: 제미나이 예측 프롬프트 생성
# ------------------------------------------
with tab_prompt:
    st.subheader("✨ 제미나이 분석용 프롬프트 자동 생성")
    st.caption("아래 라스(시장) 예측치나 당일 이슈를 넣고 생성된 프롬프트를 오른쪽 상단 복사 버튼으로 제미나이에 바로 붙여넣으세요.")

    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        las_p_ml = st.text_input("라스(시장) 승패 참고 (선택)", placeholder="예: 홈승 64% (배당 1.48)")
    with pc2:
        las_p_hd = st.text_input(f"라스 홈 {set_handicap:+.1f}세트 참고 (선택)", placeholder="예: 홈핸디 58%")
    with pc3:
        las_p_ou = st.text_input(f"라스 언오버({ou_line}) 참고 (선택)", placeholder="예: 오버 54%")

    extra_news = st.text_area(
        "당일 라인업 / 부상 / 교체 등 추가 이슈 메모 (선택)",
        placeholder="예: 홈 외국인 아포짓 무릎 테이핑, 원정 주전 세터 복귀전 등",
        height=80,
    )

    h_players_str = ", ".join(
        f"{p['선수명']}({p['포지션']}, {p['득점']}점/{p['공격성공률(%)']}%)" for p in home_prof["top_players"]
    ) or "데이터 없음"
    a_players_str = ", ".join(
        f"{p['선수명']}({p['포지션']}, {p['득점']}점/{p['공격성공률(%)']}%)" for p in away_prof["top_players"]
    ) or "데이터 없음"

    adj_notes = []
    if h_out_foreign: adj_notes.append("홈 외국인/에이스 결장 반영")
    if h_out_setter: adj_notes.append("홈 주전 세터 결장 반영")
    if a_out_foreign: adj_notes.append("원정 외국인/에이스 결장 반영")
    if a_out_setter: adj_notes.append("원정 주전 세터 결장 반영")
    adj_str = ", ".join(adj_notes) if adj_notes else "결장 보정 없음"

    set_dist_str = " | ".join(f"{k}: {v*100:.1f}%" for k, v in sim["set_scores"].items())

    gemini_prompt = f"""너는 한국 프로배구(V-리그) 정량 스포츠 애널리스트야.
아래 KOVO 공식 데이터, 몬테카를로 {n_sims:,}회 시뮬레이션 결과, 그리고 베트맨(풀세트) 기준점을 바탕으로 이 경기를 정밀 분석하고 최종 예측 픽을 내려줘.

[1. 매치업 및 기준점]
- 일자/부문: {target_date} / V-리그 {gender_label}
- 매치업: {home_prof['team']}(홈) vs {away_prof['team']}(원정)
- 베트맨 기준점: 홈 세트 핸디캡 ({set_handicap:+.1f} 세트) / 총득점 언오버 ({ou_line}점)
- 시즌 상대전적(H2H): 홈 기준 {h2h['h_wins']}승 {h2h['a_wins']}패 (세트 {h2h['h_sets']}:{h2h['a_sets']}) / 로그: {', '.join(h2h['logs']) if h2h['logs'] else '첫 맞대결'}
- 시뮬레이션 결장 보정: {adj_str}

[2. 몬테카를로 시뮬레이션 ({n_sims:,}회)]
- 사이드아웃 확률: 홈 {sim['p_so_h']:.1%} vs 원정 {sim['p_so_a']:.1%}
- 승패 확률: 홈 {sim['home_win']:.1%} vs 원정 {sim['away_win']:.1%} (몬테 픽: {mc_ml_dir} {mc_ml_prob}%)
- 홈 {set_handicap:+.1f} 세트 핸디: 홈핸디 {sim['home_set_handi']:.1%} vs 원정핸디 {sim['away_set_handi']:.1%} (몬테 픽: {mc_hd_dir} {mc_hd_prob}%)
- 총득점 언오버({ou_line}): 오버 {sim['over_prob']:.1%} vs 언더 {sim['under_prob']:.1%} / 기대총점 {sim['exp_total']:.1f}점, 중앙값 {sim['med_total']:.0f}점 (몬테 픽: {mc_ou_dir} {mc_ou_prob}%)
- 세트스코어 분포: {set_dist_str}

[3. 양 팀 전력 비교 (시즌 vs 최근 5경기 L5)]
- {home_prof['team']}(홈): L5 {home_prof['l5_record']} | 휴식 {home_prof['rest_days']}일 | 직전 5세트 여부: {home_prof['prev_was_5set']}
  · 공격효율: 시즌 {home_prof['season_atk']:.1%} / L5 {home_prof['l5_atk']:.1%} | 상대 허용 공격효율(DRtg): {home_prof['opp_atk_allowed']:.1%}
  · 리시브효율: 시즌 {home_prof['season_rec']:.1%} / L5 {home_prof['l5_rec']:.1%} | 서브 {home_prof['serve_ace_ps']:.2f} / 블로킹 {home_prof['block_ps']:.2f} / 범실 {home_prof['error_ps']:.2f}
  · 직전 경기 핵심 득점원: {h_players_str}
- {away_prof['team']}(원정): L5 {away_prof['l5_record']} | 휴식 {away_prof['rest_days']}일 | 직전 5세트 여부: {away_prof['prev_was_5set']}
  · 공격효율: 시즌 {away_prof['season_atk']:.1%} / L5 {away_prof['l5_atk']:.1%} | 상대 허용 공격효율(DRtg): {away_prof['opp_atk_allowed']:.1%}
  · 리시브효율: 시즌 {away_prof['season_rec']:.1%} / L5 {away_prof['l5_rec']:.1%} | 서브 {away_prof['serve_ace_ps']:.2f} / 블로킹 {away_prof['block_ps']:.2f} / 범실 {away_prof['error_ps']:.2f}
  · 직전 경기 핵심 득점원: {a_players_str}

[4. 라스(시장) 참고 및 추가 이슈]
- 라스 승패: {las_p_ml or '미입력'} / 라스 세트핸디: {las_p_hd or '미입력'} / 라스 언오버: {las_p_ou or '미입력'}
- 추가 현장 이슈: {extra_news.strip() or '없음 (최신 부상·교체·컨디션 뉴스가 있다면 웹 검색으로 확인 반영해 줘)'}

[요청 출력 형식]
1. 핵심 매치업 포인트 (공격 vs 수비 DRtg, 리시브·서브 상성, 5세트 피로도/일정 변수)
2. 몬테카를로 결과에서 놓치기 쉬운 리스크나 반전 포인트
3. 최종 예측 요약표 (아래 형식을 반드시 지켜서 50~95% 사이 신뢰도 숫자로 제시해 줘):
   - 승패: [홈승 또는 원정승] / 신뢰도 [XX]%
   - 세트핸디(홈 {set_handicap:+.1f}): [홈핸디 또는 원정핸디] / 신뢰도 [XX]%
   - 언오버({ou_line}): [오버 또는 언더] / 신뢰도 [XX]%
   - 예상 세트스코어: [예: 3-1]
"""

    st.code(gemini_prompt, language="markdown")


# ------------------------------------------
# TAB 3: 예측 기록 & 백테스팅 대시보드
# ------------------------------------------
with tab_track:
    st.subheader("📊 예측 결과 기록 및 적중률 백테스팅")
    db_df = load_db()

    # 3-1. 신규 예측 기록 폼
    with st.expander("➕ 현재 선택 경기 예측 기록 저장하기 (몬테 / 라스 / 제미나이)", expanded=True):
        st.caption(
            f"대상 경기: **{target_date} [{gender_label}] {home_prof['team']}(홈) vs {away_prof['team']}(원정)** "
            f"| 홈 세트핸디 **{set_handicap:+.1f}** | 언오버 **{ou_line}**"
        )
        with st.form("save_prediction_form"):
            fc1, fc2, fc3 = st.columns(3)

            with fc1:
                st.markdown("**1️⃣ 몬테카를로 (자동 입력)**")
                in_mc_ml = st.selectbox("몬테 승패", ["홈승", "원정승", "패스"], index=0 if mc_ml_dir == "홈승" else 1)
                in_mc_ml_p = st.number_input("몬테 승패 확률(%)", 0.0, 100.0, float(mc_ml_prob), 0.1)
                in_mc_hd = st.selectbox(f"몬테 홈{set_handicap:+.1f}핸디", ["홈핸디", "원정핸디", "패스"], index=0 if mc_hd_dir == "홈핸디" else 1)
                in_mc_hd_p = st.number_input("몬테 핸디 확률(%)", 0.0, 100.0, float(mc_hd_prob), 0.1)
                in_mc_ou = st.selectbox(f"몬테 언오버({ou_line})", ["오버", "언더", "패스"], index=0 if mc_ou_dir == "오버" else 1)
                in_mc_ou_p = st.number_input("몬테 언오버 확률(%)", 0.0, 100.0, float(mc_ou_prob), 0.1)

            with fc2:
                st.markdown("**2️⃣ 라스 / 시장 예측**")
                in_las_ml = st.selectbox("라스 승패", ["패스", "홈승", "원정승"], index=0)
                in_las_ml_p = st.number_input("라스 승패 확률(%)", 0.0, 100.0, 55.0, 0.5)
                in_las_hd = st.selectbox(f"라스 홈{set_handicap:+.1f}핸디", ["패스", "홈핸디", "원정핸디"], index=0)
                in_las_hd_p = st.number_input("라스 핸디 확률(%)", 0.0, 100.0, 55.0, 0.5)
                in_las_ou = st.selectbox(f"라스 언오버({ou_line})", ["패스", "오버", "언더"], index=0)
                in_las_ou_p = st.number_input("라스 언오버 확률(%)", 0.0, 100.0, 55.0, 0.5)

            with fc3:
                st.markdown("**3️⃣ 제미나이 예측**")
                in_gem_ml = st.selectbox("제미나이 승패", ["패스", "홈승", "원정승"], index=0)
                in_gem_ml_p = st.number_input("제미나이 승패 신뢰도(%)", 0.0, 100.0, 60.0, 1.0)
                in_gem_hd = st.selectbox(f"제미나이 홈{set_handicap:+.1f}핸디", ["패스", "홈핸디", "원정핸디"], index=0)
                in_gem_hd_p = st.number_input("제미나이 핸디 신뢰도(%)", 0.0, 100.0, 60.0, 1.0)
                in_gem_ou = st.selectbox(f"제미나이 언오버({ou_line})", ["패스", "오버", "언더"], index=0)
                in_gem_ou_p = st.number_input("제미나이 언오버 신뢰도(%)", 0.0, 100.0, 60.0, 1.0)

            memo_in = st.text_input("메모 (선택)", placeholder="예: 3자 만장일치 주력픽")
            submitted = st.form_submit_button("💾 예측 기록 DB에 저장", use_container_width=True)

            if submitted:
                game_id = f"{target_date}_{home_code}_{away_code}"
                new_row = {
                    "경기ID": game_id,
                    "날짜": target_date,
                    "부문": gender_label,
                    "홈팀": home_prof["team"],
                    "원정팀": away_prof["team"],
                    "홈코드": home_code,
                    "원정코드": away_code,
                    "세트핸디기준": set_handicap,
                    "언오버기준": ou_line,
                    "몬테_승패": in_mc_ml, "몬테_승패확률": in_mc_ml_p,
                    "몬테_핸디": in_mc_hd, "몬테_핸디확률": in_mc_hd_p,
                    "몬테_언오버": in_mc_ou, "몬테_언오버확률": in_mc_ou_p,
                    "라스_승패": in_las_ml, "라스_승패확률": in_las_ml_p if in_las_ml != "패스" else np.nan,
                    "라스_핸디": in_las_hd, "라스_핸디확률": in_las_hd_p if in_las_hd != "패스" else np.nan,
                    "라스_언오버": in_las_ou, "라스_언오버확률": in_las_ou_p if in_las_ou != "패스" else np.nan,
                    "제미_승패": in_gem_ml, "제미_승패확률": in_gem_ml_p if in_gem_ml != "패스" else np.nan,
                    "제미_핸디": in_gem_hd, "제미_핸디확률": in_gem_hd_p if in_gem_hd != "패스" else np.nan,
                    "제미_언오버": in_gem_ou, "제미_언오버확률": in_gem_ou_p if in_gem_ou != "패스" else np.nan,
                    "실제_홈세트": np.nan, "실제_원정세트": np.nan, "실제_총점": np.nan,
                    "메모": memo_in,
                }
                # 동일 경기ID가 있으면 기존 실제 결과는 유지한 채 업데이트
                if not db_df.empty and (db_df["경기ID"] == game_id).any():
                    idx_match = db_df.index[db_df["경기ID"] == game_id][0]
                    for res_col in ["실제_홈세트", "실제_원정세트", "실제_총점"]:
                        if pd.notna(db_df.at[idx_match, res_col]):
                            new_row[res_col] = db_df.at[idx_match, res_col]
                    for k, v in new_row.items():
                        db_df.at[idx_match, k] = v
                else:
                    db_df = pd.concat([db_df, pd.DataFrame([new_row])], ignore_index=True)

                msg = save_db(db_df)
                st.success(f"저장되었습니다! ({msg})")
                st.rerun()

    st.divider()

    # 3-2. 적중률 분석 대시보드
    st.markdown("#### 🎯 구간별 & 조합별 적중률 분석 (50% · 60% · 70% 이상)")
    f1, f2, f3 = st.columns(3)
    with f1:
        mkt_sel = st.selectbox("마켓 필터", ["전체", "승패", "핸디", "언오버"])
    with f2:
        div_sel = st.selectbox("부문 필터", ["전체", "남자부", "여자부"])
    with f3:
        prob_mode = st.radio(
            "2자/3자 일치 확률 기준",
            ["모두 이상 (엄격)", "평균 이상"],
            horizontal=True,
            help="예: 60% 이상 필터 시 일치하는 모델이 모두 60% 이상인지, 평균이 60% 이상인지 선택",
        )

    tier_df, mkt_df, n_done = analyze_vleague_backtest(db_df, mkt_sel, div_sel, prob_mode)
    if n_done == 0 or tier_df.empty:
        st.info("아직 실제 경기 결과가 입력된 백테스팅 데이터가 없습니다. 아래 표에서 경기 종료 후 스코어를 채워주세요.")
    else:
        st.caption(f"집계 완료 경기 수: **{n_done}경기** (적중률은 `적중률 (적중/전체)` 형식)")
        st.dataframe(tier_df, use_container_width=True, hide_index=True)

        with st.expander("📌 마켓별(승패 / 세트핸디 / 언오버) 적중률 상세 보기"):
            st.dataframe(mkt_df, use_container_width=True, hide_index=True)

    st.divider()

    # 3-3. 경기 결과 자동 채우기 & 직접 수정 테이블
    st.markdown("#### 📝 기록 관리 및 실제 스코어 입력")
    b1, b2, b3 = st.columns([2, 1, 1])

    with b1:
        if st.button("🔄 KOVO API로 종료 경기 스코어 자동 채우기", use_container_width=True):
            if db_df.empty:
                st.warning("저장된 예측 기록이 없습니다.")
            else:
                sched_map = {}
                for g in all_sched:
                    hs, as_ = int(g.get("hspoint", 0) or 0), int(g.get("aspoint", 0) or 0)
                    if (hs + as_) >= 3:
                        tot_pts = sum(
                            int(g.get(f"hs{i}point", 0) or 0) + int(g.get(f"as{i}point", 0) or 0)
                            for i in range(1, 6)
                        )
                        key = f"{g.get('gdate')}_{g.get('hcode')}_{g.get('acode')}"
                        sched_map[key] = (hs, as_, tot_pts)

                updated_cnt = 0
                for idx, row in db_df.iterrows():
                    if pd.isna(row["실제_홈세트"]) or pd.isna(row["실제_총점"]):
                        k = str(row["경기ID"])
                        if k in sched_map:
                            hs, as_, tot_pts = sched_map[k]
                            db_df.at[idx, "실제_홈세트"] = hs
                            db_df.at[idx, "실제_원정세트"] = as_
                            db_df.at[idx, "실제_총점"] = tot_pts
                            updated_cnt += 1

                if updated_cnt > 0:
                    msg = save_db(db_df)
                    st.success(f"{updated_cnt}경기 스코어를 자동으로 채웠습니다! ({msg})")
                    st.rerun()
                else:
                    st.info("새로 업데이트할 종료 경기가 없습니다.")

    with b2:
        if not db_df.empty:
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                db_df.to_excel(writer, index=False, sheet_name="VLeague_DB")
            st.download_button(
                "📥 엑셀 백업",
                data=buf.getvalue(),
                file_name=f"vleague_betting_db_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

    with b3:
        uploaded = st.file_uploader("CSV 복원", type=["csv"], label_visibility="collapsed")
        if uploaded is not None:
            restored = pd.read_csv(uploaded, dtype={"홈코드": str, "원정코드": str})
            msg = save_db(restored)
            st.success(f"CSV 복원 완료 ({msg})")
            st.rerun()

    if not db_df.empty:
        edited_df = st.data_editor(
            db_df,
            use_container_width=True,
            num_rows="dynamic",
            key="vleague_db_editor",
        )
        if st.button("💾 수정된 테이블(스코어/삭제 포함) 저장", use_container_width=True):
            msg = save_db(edited_df)
            st.success(f"변경 사항이 저장되었습니다! ({msg})")
            st.rerun()
