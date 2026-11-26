import io
import math
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
# 1. 기본 설정 및 KOVO API 수집 모듈
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
        "opp_drtg": 1.35,  # 상대 허용 공격효율(수비력) 반영
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
# 2. 경기별 L5 / 상대 허용 스탯(DRtg) / 피로도 계산
# ==========================================
def extract_match_team_stats(detail: Dict, tcode: str) -> Optional[Dict]:
    """team 배열 우선 파싱, 키 불일치 시 player 배열 합산으로 안전하게 폴백"""
    game = detail.get("game", {})
    sets = max(int(game.get("hspoint", 0) or 0) + int(game.get("aspoint", 0) or 0), 1)
    tcode_str = str(tcode)

    team_list = detail.get("team", []) or []
    t_row = next((t for t in team_list if str(t.get("tcode", "")) == tcode_str), None)

    if t_row and (t_row.get("att", 0) or 0) > 0:
        att = max(int(t_row.get("att", 1)), 1)
        ats = int(t_row.get("ats", 0) or 0)
        ate = int(t_row.get("ate", 0) or 0)
        atf = int(t_row.get("atf", 0) or 0)
        rt = max(int(t_row.get("rt", 1) or 1), 1)
        rs = int(t_row.get("rs", 0) or 0)
        rf = int(t_row.get("rf", 0) or 0)
        ss = int(t_row.get("ss", 0) or 0)
        bs = int(t_row.get("bs", 0) or 0)
        err = int(t_row.get("err", 0) or 0)
        return {
            "sets": sets, "att": att, "ats": ats, "ate": ate, "atf": atf,
            "rt": rt, "rs": rs, "rf": rf, "ss": ss, "bs": bs, "err": err,
        }

    players = [p for p in (detail.get("player", []) or []) if str(p.get("tcode", "")) == tcode_str]
    if not players:
        return None

    att = max(sum(int(p.get("att", 0) or 0) for p in players), 1)
    ats = sum(int(p.get("ats", 0) or 0) for p in players)
    ate = sum(int(p.get("ate", 0) or 0) for p in players)
    atf = sum(int(p.get("atf", 0) or 0) for p in players)
    rt = max(sum(int(p.get("rt", 0) or 0) for p in players), 1)
    rs = sum(int(p.get("rs", 0) or 0) for p in players)
    rf = sum(int(p.get("rf", 0) or 0) for p in players)
    ss = sum(int(p.get("ss", 0) or 0) for p in players)
    bs = sum(int(p.get("bs", 0) or 0) for p in players)
    err = sum(int(p.get("err", 0) or p.get("ate", 0) or 0) for p in players)

    return {
        "sets": sets, "att": att, "ats": ats, "ate": ate, "atf": atf,
        "rt": rt, "rs": rs, "rf": rf, "ss": ss, "bs": bs, "err": err,
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

    # 피로도 & 최근 5경기 승패
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

    # L5 상세 박스스코어 & 상대 허용 공격효율(DRtg) 집계
    my_agg = Counter()
    opp_agg = Counter()
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

        # 가장 최근 경기의 핵심 선수 Top 3 추출
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
        # 시즌 vs L5 비교용 원본
        "season_atk": float(season_row["attack_eff"]),
        "l5_atk": l5_atk,
        "season_rec": float(season_row["receive_eff"]),
        "l5_rec": l5_rec,
        "opp_atk_allowed": opp_atk_allowed,
        # 시뮬레이션 투입용 가중 평균 지표
        "attack_eff": w_s * float(season_row["attack_eff"]) + l5_weight * l5_atk,
        "receive_eff": w_s * float(season_row["receive_eff"]) + l5_weight * l5_rec,
        "serve_ace_ps": w_s * float(season_row["serve_ace_ps"]) + l5_weight * l5_srv,
        "block_ps": w_s * float(season_row["block_ps"]) + l5_weight * l5_blk,
        "error_ps": w_s * float(season_row["error_ps"]) + l5_weight * l5_err,
    }


def get_h2h_summary(finished_games: List[Dict], home_code: str, away_code: str) -> Dict:
    hc, ac = str(home_code), str(away_code)
    h2h = [
        g for g in finished_games
        if {str(g.get("hcode")), str(g.get("acode"))} == {hc, ac}
    ]
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
# 3. 랠리 단위 몬테카를로 시뮬레이션 엔진
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
    pts_spread_home: float,
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

    # 상대 전적(H2H) 미세 상성 보정 (최대 ±0.04)
    total_h2h_sets = h2h["h_sets"] + h2h["a_sets"]
    h2h_edge = 0.0
    if total_h2h_sets >= 3:
        h2h_edge = max(-0.04, min(0.04, ((h2h["h_sets"] / total_h2h_sets) - 0.5) * 0.12))

    def calc_so(off: Dict, opp: Dict, is_home: bool, adj: float, edge: float) -> float:
        z = logit(w["base_so"])
        z += w["atk"] * (off["attack_eff"] - league_avg["attack_eff"])
        # 상대가 평소 남들에게 높은 공격효율을 허용하면(수비 약함) 내 사이드아웃 확률 상승
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
    diffs = []
    h_wins, h_minus_1_5, over_hits, spread_hits = 0, 0, 0, 0

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
        diff = h_tot - a_tot
        scores[f"{hs}-{as_}"] += 1
        totals.append(tot)
        diffs.append(diff)
        if hs == 3:
            h_wins += 1
        if (hs - as_) >= 2:
            h_minus_1_5 += 1
        if tot > ou_line:
            over_hits += 1
        if (diff + pts_spread_home) > 0:
            spread_hits += 1

    totals_arr = np.array(totals)
    return {
        "p_so_h": p_so_h,
        "p_so_a": p_so_a,
        "home_win": h_wins / n_sims,
        "away_win": 1.0 - (h_wins / n_sims),
        "home_minus_1_5": h_minus_1_5 / n_sims,
        "over_prob": over_hits / n_sims,
        "under_prob": 1.0 - (over_hits / n_sims),
        "spread_prob": spread_hits / n_sims,
        "exp_total": float(totals_arr.mean()),
        "med_total": float(np.median(totals_arr)),
        "exp_diff": float(np.mean(diffs)),
        "totals_arr": totals_arr,
        "set_scores": {k: scores[k] / n_sims for k in ["3-0", "3-1", "3-2", "2-3", "1-3", "0-3"]},
    }


# ==========================================
# 4. Streamlit UI 구성
# ==========================================
st.title("🏐 V-리그 몬테카를로 매치업 분석기")

with st.sidebar:
    st.header("경기 및 모델 설정")
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

    mode = st.radio("매치업 선택 방식", ["예정 경기 불러오기", "직접 팀 선택"], horizontal=True)
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
    st.subheader("기준점 및 시뮬레이션")
    default_ou = 183.5 if gender == "1" else 179.5
    ou_line = st.number_input("언오버 기준점 (총득점)", value=default_ou, step=0.5)
    pts_spread = st.number_input("홈 점수 핸디캡 (예: -5.5)", value=-4.5, step=0.5)
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

    st.divider()
    openai_key = st.text_input("OpenAI API Key (선택: AI 프리뷰용)", type="password")

if home_code == away_code:
    st.warning("홈 팀과 원정 팀을 서로 다르게 선택해 주세요.")
    st.stop()

# 데이터 집계 및 시뮬레이션 실행
with st.spinner("최근 5경기 박스스코어 및 상대 허용 스탯 분석 중..."):
    h_row = df_teams[df_teams["tcode"] == home_code].iloc[0]
    a_row = df_teams[df_teams["tcode"] == away_code].iloc[0]

    home_prof = build_team_l5_profile(home_code, h_row, finished, target_date, season, l5_weight)
    away_prof = build_team_l5_profile(away_code, a_row, finished, target_date, season, l5_weight)
    h2h = get_h2h_summary(finished, home_code, away_code)
    league_avg = df_teams[["attack_eff", "receive_eff", "serve_ace_ps", "block_ps", "error_ps"]].mean()

    sim = run_volleyball_mc(
        home_prof, away_prof, league_avg, gender, h2h,
        home_adj, away_adj, ou_line, pts_spread, n_sims=n_sims
    )

# 상단 매치업 헤더
col_h, col_vs, col_a = st.columns([4, 2, 4])
with col_h:
    if home_prof["logo"]:
        st.image(home_prof["logo"], width=64)
    st.subheader(f"🏠 {home_prof['team']} (홈)")
    badge_h = "⚠️ 직전 5세트 접전" if home_prof["prev_was_5set"] else "체력 정상"
    st.caption(f"L5: {home_prof['l5_record']} | 휴식일: {home_prof['rest_days']}일 | {badge_h}")

with col_vs:
    st.markdown(f"<h3 style='text-align:center; margin-top:30px;'>VS</h3>", unsafe_allow_html=True)
    st.markdown(
        f"<p style='text-align:center; font-size:13px;'>시즌 H2H<br><b>{h2h['h_wins']}승 {h2h['a_wins']}패</b> (세트 {h2h['h_sets']}:{h2h['a_sets']})</p>",
        unsafe_allow_html=True,
    )

with col_a:
    if away_prof["logo"]:
        st.image(away_prof["logo"], width=64)
    st.subheader(f"✈️ {away_prof['team']} (원정)")
    badge_a = "⚠️ 직전 5세트 접전" if away_prof["prev_was_5set"] else "체력 정상"
    st.caption(f"L5: {away_prof['l5_record']} | 휴식일: {away_prof['rest_days']}일 | {badge_a}")

st.divider()

# 핵심 KPI 지표
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric(f"{home_prof['team']} 승률", f"{sim['home_win']:.1%}", f"SO {sim['p_so_h']:.1%}")
k2.metric(f"{away_prof['team']} 승률", f"{sim['away_win']:.1%}", f"SO {sim['p_so_a']:.1%}")
k3.metric("홈 -1.5 세트승", f"{sim['home_minus_1_5']:.1%}")
k4.metric(f"오버 ({ou_line})", f"{sim['over_prob']:.1%}", f"기대 {sim['exp_total']:.1f}점")
k5.metric(f"홈 핸디 ({pts_spread:+})", f"{sim['spread_prob']:.1%}", f"득실차 {sim['exp_diff']:+.1f}")

# 시각화 차트 2종
c1, c2 = st.columns(2)
with c1:
    st.markdown("#### 세트스코어 확률 분포")
    df_sc = pd.DataFrame({
        "세트스코어": list(sim["set_scores"].keys()),
        "확률(%)": [round(v * 100, 1) for v in sim["set_scores"].values()],
    })
    fig_bar = px.bar(df_sc, x="세트스코어", y="확률(%)", text="확률(%)")
    fig_bar.update_traces(textposition="outside")
    fig_bar.update_layout(height=340, margin=dict(t=20, b=20))
    st.plotly_chart(fig_bar, use_container_width=True)

with c2:
    st.markdown("#### 총득점 분포 vs 언오버 기준점")
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=sim["totals_arr"], nbinsx=40, name="시뮬레이션 총득점"))
    fig_hist.add_vline(x=ou_line, line_dash="dash", line_color="red", annotation_text=f"기준점 {ou_line}")
    fig_hist.add_vline(x=sim["exp_total"], line_dash="dot", line_color="green", annotation_text=f"평균 {sim['exp_total']:.1f}")
    fig_hist.update_layout(height=340, margin=dict(t=20, b=20), xaxis_title="양 팀 합산 총득점", yaxis_title="빈도")
    st.plotly_chart(fig_hist, use_container_width=True)

# 상세 스탯 비교표 & 직전 경기 핵심 선수
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

# AI 프리뷰 & 엑셀 저장
st.divider()
col_ai, col_xl = st.columns([3, 1])

with col_ai:
    if st.button("🤖 AI 경기 프리뷰 & 추천 픽 생성"):
        if not openai_key:
            st.warning("사이드바에 OpenAI API Key를 입력해 주세요.")
        else:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            prompt = f"""
            V-리그 {gender_label} 매치업 데이터 기반 프리뷰와 최종 추천 픽(승패/세트핸디/언오버)을 간결하게 작성해줘.
            - 매치업: {home_prof['team']}(홈) vs {away_prof['team']}(원정) ({target_date})
            - 시뮬레이션 승률: 홈 {sim['home_win']:.1%} vs 원정 {sim['away_win']:.1%} (홈 -1.5세트 {sim['home_minus_1_5']:.1%})
            - 언오버({ou_line}): 오버 {sim['over_prob']:.1%} / 기대 총점 {sim['exp_total']:.1f}점
            - 세트스코어 분포: {sim['set_scores']}
            - 홈 L5 공격효율 {home_prof['l5_atk']:.1%}, 상대 허용 {home_prof['opp_atk_allowed']:.1%}, 휴식 {home_prof['rest_days']}일(직전5세트:{home_prof['prev_was_5set']})
            - 원정 L5 공격효율 {away_prof['l5_atk']:.1%}, 상대 허용 {away_prof['opp_atk_allowed']:.1%}, 휴식 {away_prof['rest_days']}일(직전5세트:{away_prof['prev_was_5set']})
            - 시즌 상대전적: 홈 기준 {h2h['h_wins']}승 {h2h['a_wins']}패 ({h2h['logs']})
            """
            with st.spinner("AI 프리뷰 생성 중..."):
                res = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                )
                st.markdown(res.choices[0].message.content)

with col_xl:
    export_df = pd.DataFrame([{
        "날짜": target_date,
        "부문": gender_label,
        "홈팀": home_prof["team"],
        "원정팀": away_prof["team"],
        "홈승률": round(sim["home_win"], 4),
        "홈-1.5세트": round(sim["home_minus_1_5"], 4),
        "언오버기준점": ou_line,
        "오버확률": round(sim["over_prob"], 4),
        "기대총점": round(sim["exp_total"], 1),
        "홈핸디기준": pts_spread,
        "홈핸디적중률": round(sim["spread_prob"], 4),
        "실제세트스코어": "",
        "실제총점": "",
    }])
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        export_df.to_excel(writer, index=False, sheet_name="VLeague_Pick")
    st.download_button(
        "📥 백테스팅용 엑셀 저장",
        data=buf.getvalue(),
        file_name=f"vleague_{target_date}_{home_prof['team']}_vs_{away_prof['team']}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
