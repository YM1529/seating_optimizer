# ==========================================
# 席替え最適化クラス (SeatingOptimizer) - 固定席ペアのコスト除外対応版
# ==========================================
class SeatingOptimizer:
    def __init__(self, url, names_sheet='names_list', layout_sheet='table_layout', 
                 output_sheet='meal_seat_assignments', fixed_sheet='fixed_seats'):
        """
        初期化: シートの取得・データ読み込み・座席グラフ構造の構築
        """
        self.url = url
        self.names_sheet_name = names_sheet
        self.layout_sheet_name = layout_sheet
        self.output_sheet_name = output_sheet
        self.fixed_sheet_name = fixed_sheet
        
        self.sh = gc.open_by_url(self.url)
        
        # 内部状態・属性の初期化
        self.seat_positions = {}
        self.seat_neighbors = []
        self.num_seats = 0
        self.meals = []
        self.num_meals = 0
        self.all_members = []
        self.num_members = 0
        self.member_to_id = {}
        self.id_to_member = {}
        self.present_members_by_meal = []
        
        # 内部保持変数: 固定座席情報 & 固定ペアの隣接重み
        self.fixed_seats = {}
        self.fixed_pair_weights = None  # {(p1, p2): total_fixed_weight}
        
        # 状態保持用
        self.current_state = []
        self.best_state = []
        self.pair_counts = None
        self.member_penalties = None
        self.current_cost = 0.0
        self.best_cost = float('inf')
        
        # ログ保持用
        self.loss_history_iter = []
        self.loss_history_val = []
        self.bias_history_val = []

        # 準備実行
        self._load_and_parse_sheets()
        self._build_neighborhood_graph()

    def _load_and_parse_sheets(self):
        """スプレッドシートからデータを読み込み、メンバーIDリストを構築"""
        try:
            names_sheet = self.sh.worksheet(self.names_sheet_name)
            layout_sheet = self.sh.worksheet(self.layout_sheet_name)
        except Exception as e:
            print(f"エラー: シート名 '{self.names_sheet_name}' または '{self.layout_sheet_name}' が見つかりません。")
            raise e

        raw_names_data = names_sheet.get_all_values()
        if not raw_names_data:
            raise ValueError(f"'{self.names_sheet_name}' シートが空です。")

        self.meals = [str(col).strip() for col in raw_names_data[0] if str(col).strip() != '']
        self.num_meals = len(self.meals)

        df_names = pd.DataFrame(raw_names_data[1:], columns=raw_names_data[0])
        self.all_members = sorted(list(set(df_names.values.flatten()) - {''}))
        self.num_members = len(self.all_members)
        self.member_to_id = {name: i for i, name in enumerate(self.all_members)}
        self.id_to_member = {i: name for i, name in enumerate(self.all_members)}

        self.present_members_by_meal = []
        for m_name in self.meals:
            p_names = [str(n).strip() for n in df_names[m_name].dropna().values if str(n).strip() != '']
            p_ids = [self.member_to_id[n] for n in p_names if n in self.member_to_id]
            self.present_members_by_meal.append(p_ids)

        self.layout_matrix = np.array(layout_sheet.get_all_values())

    def _build_neighborhood_graph(self):
        """座席位置および隣接グラフの構築"""
        rows, cols = self.layout_matrix.shape
        for r in range(rows):
            for c in range(cols):
                val = str(self.layout_matrix[r, c]).strip()
                if val.isdigit():
                    self.seat_positions[int(val)] = (r, c)

        self.num_seats = len(self.seat_positions)
        self.seat_neighbors = [[] for _ in range(self.num_seats)]

        for s1, (r1, c1) in self.seat_positions.items():
            for s2, (r2, c2) in self.seat_positions.items():
                if s1 < s2:
                    dr, dc = abs(r1 - r2), abs(c1 - c2)
                    if dr + dc == 1:   # タテ・ヨコ隣接
                        self.seat_neighbors[s1].append((s2, 1.0))
                        self.seat_neighbors[s2].append((s1, 1.0))
                    elif dr == 1 and dc == 1: # 斜め
                        self.seat_neighbors[s1].append((s2, 0.5))
                        self.seat_neighbors[s2].append((s1, 0.5))

    def create_or_reset_fixed_seats_sheet(self):
        """固定座席入力用の専用シートを作成・リセット"""
        try:
            ws = self.sh.worksheet(self.fixed_sheet_name)
            ws.clear()
            print(f"固定座席入力シート '{self.fixed_sheet_name}' をリセットしました。")
        except Exception:
            ws = self.sh.add_worksheet(title=self.fixed_sheet_name, rows="100", cols="20")
            print(f"固定座席入力シート '{self.fixed_sheet_name}' を新規作成しました。")

        headers = ['席番号'] + self.meals
        rows = [[s] + [''] * self.num_meals for s in range(self.num_seats)]
        ws.update([headers] + rows)
        print("-> 席番号一覧のテンプレートをセットしました。")

    def load_fixed_seats_from_sheet(self):
        """固定座席指定シートからの読み取り & 固定ペアの隣接重み算出"""
        self.fixed_seats = {}
        try:
            ws = self.sh.worksheet(self.fixed_sheet_name)
            data = ws.get_all_values()
            if len(data) <= 1:
                print("固定座席シートデータが空のため、固定座席なしで読み込みました。")
                self._identify_fixed_pairs()
                return

            headers = [str(h).strip() for h in data[0]]
            count = 0

            for row in data[1:]:
                if not row or str(row[0]).strip() == '': continue
                seat_str = str(row[0]).strip()
                if not seat_str.isdigit(): continue
                seat_num = int(seat_str)

                for col_idx in range(1, len(headers)):
                    m_name = headers[col_idx]
                    val = str(row[col_idx]).strip() if col_idx < len(row) else ''
                    
                    if val and val in self.member_to_id and m_name in self.meals:
                        m_idx = self.meals.index(m_name)
                        self.fixed_seats[(m_idx, seat_num)] = self.member_to_id[val]
                        count += 1
                        
            print(f"固定座席シートから {count} 件の指定席を内部変数 (self.fixed_seats) に読み込みました。")
            self._identify_fixed_pairs()

        except Exception as e:
            print(f"固定座席シート '{self.fixed_sheet_name}' の読み込み中にエラーが発生しました: {e}")
            self._identify_fixed_pairs()

    def _identify_fixed_pairs(self):
        """
        固定座席同士で発生している不可避な隣接重みを事前計算
        """
        self.fixed_pair_weights = np.zeros((self.num_members, self.num_members), dtype=float)
        
        for m in range(self.num_meals):
            # 該当回での固定席のマップ {seat_num: person_id}
            meal_fixed = {s: p for (f_m, s), p in self.fixed_seats.items() if f_m == m}
            
            for s1, p1 in meal_fixed.items():
                for s2, weight in self.seat_neighbors[s1]:
                    if s1 < s2 and s2 in meal_fixed:
                        p2 = meal_fixed[s2]
                        if p1 != p2:
                            self.fixed_pair_weights[p1, p2] += weight
                            self.fixed_pair_weights[p2, p1] += weight

    def _initialize_state(self, use_current_as_initial=False):
        """初期配置の作成"""
        self.current_state = []
        effective_fixed = dict(self.fixed_seats)

        if use_current_as_initial:
            try:
                out_sheet = self.sh.worksheet(self.output_sheet_name)
                data = out_sheet.get_all_values()
                if len(data) > 1:
                    headers = [str(h).strip() for h in data[0]]
                    for row in data[1:]:
                        if not row or not str(row[0]).strip().isdigit(): continue
                        seat_num = int(str(row[0]).strip())
                        for col_idx in range(1, len(headers)):
                            m_name = headers[col_idx]
                            val = str(row[col_idx]).strip() if col_idx < len(row) else ''
                            if val and val != '（空席）' and val != '空席' and m_name in self.meals and val in self.member_to_id:
                                m_idx = self.meals.index(m_name)
                                effective_fixed[(m_idx, seat_num)] = self.member_to_id[val]
                    print("前回の最適化結果を初期配置として引き継ぎました。")
            except Exception:
                print("前回の出力シートが存在しないため、新規ランダム初期化を実行します。")

        for m in range(self.num_meals):
            seats = [-1] * self.num_seats
            present = self.present_members_by_meal[m][:]
            
            for (f_m, f_s), f_p in effective_fixed.items():
                if f_m == m:
                    seats[f_s] = f_p
                    if f_p in present:
                        present.remove(f_p)
                        
            random.shuffle(present)
            p_idx = 0
            for s in range(self.num_seats):
                if s < len(self.present_members_by_meal[m]):
                    if seats[s] == -1:
                        seats[s] = present[p_idx]
                        p_idx += 1
            self.current_state.append(seats)

        if self.fixed_pair_weights is None:
            self._identify_fixed_pairs()

        self.pair_counts = np.zeros((self.num_members, self.num_members), dtype=float)
        for m in range(self.num_meals):
            seats = self.current_state[m]
            for s1 in range(self.num_seats):
                p1 = seats[s1]
                if p1 == -1: continue
                for s2, weight in self.seat_neighbors[s1]:
                    if s1 < s2:
                        p2 = seats[s2]
                        if p2 != -1:
                            self.pair_counts[p1, p2] += weight
                            self.pair_counts[p2, p1] += weight

        self.current_cost = self._calc_total_cost(self.pair_counts)
        self.member_penalties = np.zeros(self.num_members, dtype=float)
        
        # 固定席同士の重みを引いた実効カウントでペナルティ初期化
        eff_counts = np.maximum(0.0, self.pair_counts - self.fixed_pair_weights)
        for i in range(self.num_members):
            for j in range(self.num_members):
                if i != j and eff_counts[i, j] > 1.0:
                    self.member_penalties[i] += self._calc_pair_cost(eff_counts[i, j])

        self.best_state = [s[:] for s in self.current_state]
        self.best_cost = self.current_cost

    @staticmethod
    def _calc_pair_cost(c):
        """実効同席重みにおけるペナルティ計算 (1.0を超えた分を2乗)"""
        return (c - 1.0) ** 2 if c > 1.0 else 0.0

    def _calc_total_cost(self, p_counts):
        """固定席同士の隣接重みを除外した実効コストを算出"""
        total = 0.0
        eff_counts = np.maximum(0.0, p_counts - self.fixed_pair_weights)
        for i in range(self.num_members):
            for j in range(i + 1, self.num_members):
                c = eff_counts[i, j]
                if c > 1.0:
                    total += (c - 1.0) ** 2
        return total

    def optimize(
        self,
        iterations=300000,
        warmup_steps=90000,
        min_bias_prob=0.05,
        max_bias_prob=0.50,
        initial_temp=15.0,
        final_temp=0.01,
        plt_freq=3000,
        use_current_as_initial=False
    ):
        """
        最適化メイン処理
        """
        self._initialize_state(use_current_as_initial=use_current_as_initial)
        
        if not use_current_as_initial:
            self.loss_history_iter = []
            self.loss_history_val = []
            self.bias_history_val = []
            start_step_offset = 0
        else:
            start_step_offset = self.loss_history_iter[-1] if self.loss_history_iter else 0
            max_bias_prob = min(max_bias_prob, 0.15) 
            min_bias_prob = min(min_bias_prob, 0.02)
            initial_temp = min(initial_temp, 2.0)
            warmup_steps = int(warmup_steps * 0.3)

        print(f"最適化が開始されました ({'継続最適化' if use_current_as_initial else 'リセット・再最適化'})  初期コスト総計: {self.current_cost:.2f}")
        start_time = time.time()

        for step in range(iterations):
            temp = initial_temp * ((final_temp / initial_temp) ** (step / iterations))
            
            if step < warmup_steps:
                ratio = step / max(1, warmup_steps)
                current_bias_prob = min_bias_prob + ratio * (max_bias_prob - min_bias_prob)
            else:
                remain_steps = max(1, iterations - warmup_steps)
                ratio = (step - warmup_steps) / remain_steps
                current_bias_prob = max_bias_prob - ratio * (max_bias_prob - min_bias_prob)

            m = random.randint(0, self.num_meals - 1)
            num_p = len(self.present_members_by_meal[m])
            if num_p <= 1: continue
                
            movable_seats = [s for s in range(num_p) if (m, s) not in self.fixed_seats]
            if len(movable_seats) < 2: continue

            s1, s2 = None, None
            if random.random() < current_bias_prob and np.sum(self.member_penalties) > 0:
                candidates = [(s, self.current_state[m][s]) for s in movable_seats if self.current_state[m][s] != -1]
                if candidates:
                    seats_list, members_list = zip(*candidates)
                    weights = np.array([self.member_penalties[p] for p in members_list])
                    if np.sum(weights) > 0:
                        probs = weights / np.sum(weights)
                        chosen_idx = np.random.choice(len(seats_list), p=probs)
                        s1 = seats_list[chosen_idx]

            if s1 is None:
                s1 = random.choice(movable_seats)
                
            other_seats = [s for s in movable_seats if s != s1]
            s2 = random.choice(other_seats)

            p1 = self.current_state[m][s1]
            p2 = self.current_state[m][s2]

            if p1 == p2: continue

            # --- 差分計算 (Delta Calculation) ---
            deltas = {}
            def add_delta(a, b, w):
                if a == -1 or b == -1 or a == b: return
                if a > b: a, b = b, a
                deltas[(a, b)] = deltas.get((a, b), 0.0) + w

            for neighbor_seat, w in self.seat_neighbors[s1]:
                if neighbor_seat == s2: continue
                np_id = self.current_state[m][neighbor_seat]
                if np_id != -1:
                    add_delta(p1, np_id, -w)
                    add_delta(p2, np_id, +w)

            for neighbor_seat, w in self.seat_neighbors[s2]:
                if neighbor_seat == s1: continue
                np_id = self.current_state[m][neighbor_seat]
                if np_id != -1:
                    add_delta(p2, np_id, -w)
                    add_delta(p1, np_id, +w)

            cost_diff = 0.0
            for (a, b), dw in deltas.items():
                old_raw = self.pair_counts[a, b]
                new_raw = old_raw + dw
                
                fixed_w = self.fixed_pair_weights[a, b]
                old_eff = max(0.0, old_raw - fixed_w)
                new_eff = max(0.0, new_raw - fixed_w)
                
                cost_diff += self._calc_pair_cost(new_eff) - self._calc_pair_cost(old_eff)

            # 採択判定
            if cost_diff < 0 or random.random() < math.exp(-cost_diff / temp):
                self.current_state[m][s1], self.current_state[m][s2] = p2, p1
                self.current_cost += cost_diff
                
                for (a, b), dw in deltas.items():
                    old_raw = self.pair_counts[a, b]
                    new_raw = old_raw + dw
                    
                    fixed_w = self.fixed_pair_weights[a, b]
                    old_eff = max(0.0, old_raw - fixed_w)
                    new_eff = max(0.0, new_raw - fixed_w)
                    
                    old_pen = self._calc_pair_cost(old_eff)
                    new_pen = self._calc_pair_cost(new_eff)
                    pen_diff = new_pen - old_pen
                    
                    self.pair_counts[a, b] = new_raw
                    self.pair_counts[b, a] = new_raw
                    
                    self.member_penalties[a] += pen_diff
                    self.member_penalties[b] += pen_diff

                if self.current_cost < self.best_cost:
                    self.best_cost = self.current_cost
                    self.best_state = [s[:] for s in self.current_state]

            if step % plt_freq == 0 or step == iterations - 1:
                self.loss_history_iter.append(start_step_offset + step)
                self.loss_history_val.append(self.best_cost)
                self.bias_history_val.append(current_bias_prob)

        execution_time = time.time() - start_time
        print(f"計算終了 実行時間: {execution_time:.2f} 秒 ({iterations/execution_time:.0f} iter/sec)")
        print(f"最終コスト総計: {self.best_cost:.2f}")

    def save_results(self):
        """最適化結果を出力用スプレッドシートへ保存"""
        out_sheet = self.sh.worksheet(self.output_sheet_name)
        result_data = []
        for m_idx, m_name in enumerate(self.meals):
            for s in range(self.num_seats):
                p_id = self.best_state[m_idx][s]
                assigned_person = self.id_to_member[p_id] if p_id != -1 else "空席"
                result_data.append({
                    '食事の回': m_name,
                    '席番号': s,
                    '着席者': assigned_person
                })

        df_result = pd.DataFrame(result_data)
        df_pivot = df_result.pivot(index='席番号', columns='食事の回', values='着席者')
        df_pivot = df_pivot.reindex(columns=self.meals).reset_index()

        out_sheet.clear()
        out_sheet.update([df_pivot.columns.values.tolist()] + df_pivot.values.tolist())
        print(f"結果をシート '{self.output_sheet_name}' に保存しました")

    def plot_history(self):
        """最適化の推移グラフを描画"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

        ax1.plot(self.loss_history_iter, self.loss_history_val, label='Best Penalty Loss', color='crimson')
        ax1.set_ylabel('Penalty Loss')
        ax1.set_title('Optimization Loss & Dynamic Bias Schedule')
        ax1.grid(True)
        ax1.legend(loc='upper right')

        ax2.plot(self.loss_history_iter, self.bias_history_val, label='Bias Probability', color='royalblue', linestyle='--')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Bias Prob')
        ax2.grid(True)
        ax2.legend(loc='upper right')

        plt.tight_layout()
        plt.show()

    def analyze_duplicates(self):
        """同席重複の内訳をテキスト表示 (固定席由来のカウントは除外して判定)"""
        print("\n--- 重複ペア分析レポート ---")
        duplicate_found = False
        eff_counts = np.maximum(0.0, self.pair_counts - self.fixed_pair_weights)
        for i in range(self.num_members):
            for j in range(i + 1, self.num_members):
                c = eff_counts[i, j]
                if c > 1.0:
                    p1 = self.id_to_member[i]
                    p2 = self.id_to_member[j]
                    pen = self._calc_pair_cost(c)
                    print(f"- {p1} & {p2}: 重み合計 {c:.1f} (ペナルティ: {pen:.2f})")
                    duplicate_found = True
        if not duplicate_found:
            print("重複ペナルティが発生しているペアはありません")
