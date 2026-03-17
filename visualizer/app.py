"""
Match analysis visualizer. Drop match log, CSV, and bot log in visualizer/data (or provide paths),
set "We are Team 0/1", and run: streamlit run visualizer/app.py
"""
import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from parsers.match_log import parse_match_runner_log
from parsers.match_csv import parse_match_csv
from parsers.bot_log import parse_bot_log
from analysis.strategy_shifts import detect_opponent_shifts
from analysis.read_accuracy import compute_read_accuracy
from analysis.mistakes import extract_mistakes

st.set_page_config(page_title="Match analysis", layout="wide")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def load_files(match_log_path, csv_path, bot_log_path):
    match_data = csv_data = bot_data = None
    err = []
    if match_log_path and os.path.isfile(match_log_path):
        try:
            match_data = parse_match_runner_log(match_log_path)
        except Exception as e:
            err.append(f"Match log: {e}")
    if csv_path and os.path.isfile(csv_path):
        try:
            csv_data = parse_match_csv(csv_path)
        except Exception as e:
            err.append(f"CSV: {e}")
    if bot_log_path and os.path.isfile(bot_log_path):
        try:
            bot_data = parse_bot_log(bot_log_path)
        except Exception as e:
            err.append(f"Bot log: {e}")
    return match_data, csv_data, bot_data, err


def main():
    st.title("Match analysis visualizer")
    st.markdown("Drop match runner log, match CSV, and bot log; set **We are Team 0** or **Team 1**; then load and view.")

    data_dir = st.text_input("Data directory (or leave default)", DATA_DIR)
    if not data_dir or not os.path.isdir(data_dir):
        st.info("Enter a directory containing match log, CSV, and bot log, or use file pickers below.")
    else:
        files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
        st.caption(f"Files in directory: {', '.join(files) if files else 'none'}")

    col1, col2, col3 = st.columns(3)
    with col1:
        match_log_path = st.text_input("Match log path", os.path.join(data_dir or DATA_DIR, "match_25757.txt"))
    with col2:
        csv_path = st.text_input("Match CSV path", os.path.join(data_dir or DATA_DIR, "match_25770.csv"))
    with col3:
        bot_log_path = st.text_input("Bot log path", os.path.join(data_dir or DATA_DIR, "bot.log"))

    our_team_id = st.radio("We are", [0, 1], format_func=lambda x: f"Team {x}", horizontal=True)

    if st.button("Load and analyze"):
        match_data, csv_data, bot_data, err = load_files(match_log_path, csv_path, bot_log_path)
        if err:
            for e in err:
                st.error(e)
        if not bot_data and not csv_data and not match_data:
            st.warning("Load at least one file (bot log + CSV recommended for full analysis).")
            return

        st.session_state["match_data"] = match_data
        st.session_state["csv_data"] = csv_data
        st.session_state["bot_data"] = bot_data
        st.session_state["our_team_id"] = our_team_id
        st.session_state["loaded"] = True

    if not st.session_state.get("loaded"):
        st.stop()

    match_data = st.session_state.get("match_data")
    csv_data = st.session_state.get("csv_data")
    bot_data = st.session_state.get("bot_data")
    our_team_id = st.session_state.get("our_team_id", 0)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Losses & mistakes", "Opponent recon", "Read accuracy", "Hand drill-down"])

    with tab1:
        st.header("Overview")
        if match_data and match_data.get("bankroll_checkpoints"):
            df = pd.DataFrame(match_data["bankroll_checkpoints"])
            our_col = "team_0_bankroll" if our_team_id == 0 else "team_1_bankroll"
            opp_col = "team_1_bankroll" if our_team_id == 0 else "team_0_bankroll"
            df["our_bankroll"] = df[our_col]
            fig = px.line(df, x="hand_number", y="our_bankroll", title="Our bankroll (from match log checkpoints)")
            st.plotly_chart(fig, use_container_width=True)
        if match_data:
            f0 = match_data.get("final_team_0_bankroll")
            f1 = match_data.get("final_team_1_bankroll")
            if f0 is not None and f1 is not None:
                our_final = f0 if our_team_id == 0 else f1
                st.metric("Our final bankroll", our_final)
            t0 = match_data.get("time_used_0")
            t1 = match_data.get("time_used_1")
            if t0 is not None and t1 is not None:
                st.write(f"Time used - Team 0: {t0:.1f}s, Team 1: {t1:.1f}s")
        if csv_data and csv_data.get("hands"):
            st.write(f"Total hands in CSV: {len(csv_data['hands'])}")
        if bot_data and bot_data.get("hand_results"):
            st.write(f"Hand results in bot log: {len(bot_data['hand_results'])}")
        # Position validation: we are SB for hand N iff (our_team_id == 0 and N%2==0) or (our_team_id == 1 and N%2==1)
        if bot_data and bot_data.get("hand_results"):
            mismatches = []
            for r in bot_data["hand_results"][:100]:
                h = r.get("hand")
                pos = (r.get("position") or "").strip().upper()
                if h is None or not pos:
                    continue
                expected_sb = (our_team_id == 0 and h % 2 == 0) or (our_team_id == 1 and h % 2 == 1)
                expected_pos = "SB" if expected_sb else "BB"
                if pos != expected_pos:
                    mismatches.append({"hand": h, "logged": pos, "expected": expected_pos})
            if mismatches:
                st.warning(f"Position mismatch (check We are Team): {mismatches[:5]}")

    with tab2:
        st.header("Losses & mistakes")
        if bot_data and bot_data.get("hand_results"):
            mistakes = extract_mistakes(bot_data["hand_results"])
            st.subheader("Invalid action (auto-fold)")
            st.write(mistakes.get("invalid_action_hands", []))
            st.subheader("We folded in big pots")
            st.dataframe(pd.DataFrame(mistakes.get("we_fold_big_pot", [])))
            st.subheader("Loss breakdown (by street_ended, end_type, position)")
            lb = mistakes.get("loss_breakdown", {})
            rows = [{"street_ended": k[0], "end_type": k[1], "position": k[2], "count": v["count"], "total_reward": v["total_reward"]} for k, v in lb.items()]
            if rows:
                st.dataframe(pd.DataFrame(rows))
            lost = [r for r in bot_data["hand_results"] if r.get("lost") or (r.get("reward") or 0) < 0]
            st.subheader("Losing hands (sample)")
            if lost:
                st.dataframe(pd.DataFrame(lost[:50]))
        else:
            st.info("Load bot log to see losses and mistakes.")

    with tab3:
        st.header("Opponent recon over time")
        if bot_data and bot_data.get("opp_recon"):
            recon = bot_data["opp_recon"]
            df = pd.DataFrame(recon)
            for col in ["vpip", "pfr", "af"]:
                if col in df.columns:
                    fig = px.line(df, x="hand", y=col, title=f"OPP_RECON {col}")
                    shifts = detect_opponent_shifts(recon)
                    shift_hands = [s["hand_number"] for s in shifts if s.get("metric") == col]
                    if shift_hands:
                        for h in shift_hands:
                            fig.add_vline(x=h, line_dash="dash", line_color="gray")
                    st.plotly_chart(fig, use_container_width=True)
            st.subheader("Detected strategy shifts")
            shifts = detect_opponent_shifts(recon)
            if shifts:
                st.dataframe(pd.DataFrame(shifts))
            else:
                st.caption("No significant shifts detected (try smaller window or thresholds).")
        else:
            st.info("Load bot log to see opponent recon.")

    with tab4:
        st.header("Read accuracy (our recon vs actual from CSV)")
        if bot_data and csv_data and bot_data.get("opp_recon") and csv_data.get("hands"):
            acc = compute_read_accuracy(bot_data["opp_recon"], csv_data["hands"], our_team_id)
            if acc:
                df = pd.DataFrame(acc)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df["hand_number"], y=df["our_vpip"], name="our vpip", mode="lines+markers"))
                fig.add_trace(go.Scatter(x=df["hand_number"], y=df["actual_vpip"], name="actual vpip (trailing 50)", mode="lines+markers"))
                fig.update_layout(title="VPIP: our read vs actual")
                st.plotly_chart(fig, use_container_width=True)
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=df["hand_number"], y=df["our_pfr"], name="our pfr", mode="lines+markers"))
                fig2.add_trace(go.Scatter(x=df["hand_number"], y=df["actual_pfr"], name="actual pfr (trailing 50)", mode="lines+markers"))
                fig2.update_layout(title="PFR: our read vs actual")
                st.plotly_chart(fig2, use_container_width=True)
                st.dataframe(df.head(100))
            else:
                st.caption("No overlap between bot log hands and CSV hands.")
        else:
            st.info("Load both bot log and CSV for read accuracy.")

    with tab5:
        st.header("Hand drill-down")
        if bot_data and csv_data:
            hand_nums = []
            if bot_data.get("hand_result_by_hand"):
                hand_nums = sorted(bot_data["hand_result_by_hand"].keys())
            elif csv_data.get("hands"):
                hand_nums = [h["hand_number"] for h in csv_data["hands"]]
            if hand_nums:
                hand = st.selectbox("Select hand", hand_nums)
                if bot_data.get("hand_result_by_hand") and hand in bot_data["hand_result_by_hand"]:
                    st.subheader("HAND_RESULT")
                    st.json(bot_data["hand_result_by_hand"][hand])
                if bot_data.get("opp_recon_by_hand") and hand in bot_data["opp_recon_by_hand"]:
                    st.subheader("OPP_RECON")
                    st.json(bot_data["opp_recon_by_hand"][hand])
                if csv_data.get("hands"):
                    hand_actions = [h for h in csv_data["hands"] if h["hand_number"] == hand]
                    if hand_actions:
                        st.subheader("CSV actions")
                        st.dataframe(pd.DataFrame(hand_actions[0]["actions"]))
            else:
                st.caption("No hands to select.")
        else:
            st.info("Load bot log and CSV for drill-down.")


if __name__ == "__main__":
    main()
