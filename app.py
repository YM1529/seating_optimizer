import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from main import SeatingOptimizer

# ページの設定
st.set_page_config(page_title="席替え最適化ツール", layout="wide")

st.title("席替え自動最適化アプリ")
st.write("Googleスプレッドシートと連携して、最適な席順を算出します")

# サイドバー（設定パネル）
st.sidebar.header("設定パラメータ")
spreadsheet_url = st.sidebar.text_input("スプレッドシートのURL", value="")

iterations = st.sidebar.number_input("計算回数", min_value=10000, max_value=1000000, value=100000, step=10000)
use_initial = st.sidebar.checkbox("前回の結果を初期配置として引き継ぐ", value=False)

# メイン画面のボタン
if st.button("最適化を実行", type="primary"):
    if not spreadsheet_url:
        st.error("スプレッドシートのURLを入力してください。")
    else:
        try:
            with st.spinner("スプレッドシートを読み込み中..."):
                optimizer = SeatingOptimizer(url=spreadsheet_url)
                optimizer.load_fixed_seats_from_sheet()

            st.info("計算を実行中...")
            
            # 最適化の実行
            optimizer.optimize(
                iterations=iterations,
                use_current_as_initial=use_initial
            )

            st.success("計算が完了しました！")

            # 結果の保存
            with st.spinner("スプレッドシートへ書き込み中..."):
                optimizer.save_results()
            st.success("結果をスプレッドシートに保存しました！")

            # グラフの表示
            st.subheader("最適化の推移")
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            ax1.plot(optimizer.loss_history_iter, optimizer.loss_history_val, color='crimson')
            ax1.set_ylabel('Penalty Loss')
            ax1.grid(True)

            ax2.plot(optimizer.loss_history_iter, optimizer.bias_history_val, color='royalblue', linestyle='--')
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Bias Prob')
            ax2.grid(True)

            st.pyplot(fig, use_container_width=True)
            plt.close(fig)  # メモリ開放

            # 重複レポートの表示
            st.subheader("重複ペア分析")
            eff_counts = np.maximum(0.0, optimizer.pair_counts - optimizer.fixed_pair_weights)
            duplicate_found = False
            
            for i in range(optimizer.num_members):
                for j in range(i + 1, optimizer.num_members):
                    c = eff_counts[i, j]
                    if c > 1.0:
                        p1 = optimizer.id_to_member[i]
                        p2 = optimizer.id_to_member[j]
                        pen = optimizer._calc_pair_cost(c)
                        st.warning(f"⚠️ **{p1} & {p2}**: 重み合計 {c:.1f} (ペナルティ: {pen:.2f})")
                        duplicate_found = True
            
            if not duplicate_found:
                st.success("重複ペナルティが発生しているペアはありません！")

        except Exception as e:
            st.error(f"エラーが発生しました: {e}")
