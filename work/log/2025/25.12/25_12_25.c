// ===================== 配置 =====================
#define N_ACTIONS 5

static const float DV_SET[N_ACTIONS] = {
    -DV_L,   // 大减
    -DV_S,   // 小减
     0.0f,   // 不变
    +DV_S,   // 小加
    +DV_L    // 大加
};

typedef struct {
    float T_last;     // 上一锤峰值扭矩
    float v_last;     // 上一锤速度
    float T_target;   // 目标
    float k_hat;      // T_max = k*v 的在线估计
    float beta_hat;   // z_{t+1} = z_t*beta 的在线估计
    float z_hat;      // 当前缺口 z
} MPCState;

static inline float clampf(float x, float lo, float hi){
    if(x < lo) return lo;
    if(x > hi) return hi;
    return x;
}

// ===================== 1) 在线参数更新 =====================
// 用测得 T_meas(当前这锤) 来更新 z_hat、k_hat、beta_hat
void mpc_update_params(MPCState* s, float T_meas, float v_used)
{
    // 1) 用当前 k_hat 估计 T_max，并由测量反推出 z
    float Tmax_est = s->k_hat * v_used;
    Tmax_est = clampf(Tmax_est, 1e-3f, 1e9f);  // 防止除0

    float z_meas = 1.0f - (T_meas / Tmax_est);
    z_meas = clampf(z_meas, 0.0f, 1.0f);

    // 2) 更新 beta：beta_meas ≈ z_meas / z_prev
    //    注意：如果 z_prev 很小，beta 会不稳定，做保护
    float z_prev = s->z_hat;
    float beta_meas = s->beta_hat;
    if (z_prev > 0.05f) {
        beta_meas = clampf(z_meas / z_prev, 0.2f, 0.999f);
    }
    // 低通平滑
    s->beta_hat = clampf( (1.0f - ETA_BETA)*s->beta_hat + ETA_BETA*beta_meas, 0.2f, 0.999f);

    // 3) 更新 k：让预测的扭矩更贴近测量
    //    T_meas ≈ (k*v_used)*(1 - z_prev*beta_hat)   （用上一时刻缺口推进一拍）
    float one_minus_z = 1.0f - (z_prev * s->beta_hat);
    one_minus_z = clampf(one_minus_z, 0.05f, 1.0f);

    float T_pred_from_k = (s->k_hat * v_used) * one_minus_z;
    float e = T_meas - T_pred_from_k;

    // 梯度式更新（足够工程化，避免RLS复杂度）
    // k += eta * e / (v_used*one_minus_z)
    float denom = v_used * one_minus_z;
    denom = clampf(denom, 1e-3f, 1e9f);

    s->k_hat = clampf(s->k_hat + ETA_K * (e / denom), K_MIN, K_MAX);

    // 4) 更新 z_hat（用测量值低通）
    s->z_hat = clampf((1.0f - ETA_Z)*s->z_hat + ETA_Z*z_meas, 0.0f, 1.0f);

    // 5) 更新记录
    s->T_last = T_meas;
    s->v_last = v_used;
}

// ===================== 2) 一步预测模型 =====================
// 给定当前状态 s，选择下一锤速度 v_next，预测下一锤峰值扭矩
static inline float predict_next_T(const MPCState* s, float v_next, float* z_next_out)
{
    float z_next = clampf(s->z_hat * s->beta_hat, 0.0f, 1.0f);
    float Tmax_next = s->k_hat * v_next;

    float T_pred = Tmax_next * (1.0f - z_next);

    if (z_next_out) *z_next_out = z_next;
    return T_pred;
}

// ===================== 3) 代价函数（两段逻辑） =====================
static inline float stage_cost(float T_now, float T_pred, float T_target,
                               float dv, float v_next)
{
    float e_now  = T_target - T_now;
    float e_pred = T_target - T_pred;

    // 过冲量（预测）
    float overshoot = (T_pred > T_target) ? (T_pred - T_target) : 0.0f;

    // 远离目标：鼓励增长（等价于“奖励 ΔT_pred”）
    float dT_pred = T_pred - T_now;

    // 权重切换：误差越大越“激进”，越接近越“保守”
    float far = clampf(e_now / E_FAR, 0.0f, 1.0f);   // e_now>=E_FAR -> 1
    float near = 1.0f - far;

    float cost = 0.0f;

    // 基础：预测误差
    cost += W_ERR * fabsf(e_pred);

    // 过冲强惩罚（接近目标时更强）
    cost += (W_OVS_BASE + W_OVS_NEAR * near) * overshoot * overshoot;

    // 远离目标时：鼓励扭矩快速增长（减少敲击次数）
    // 这里用 " -W_GROW * dT_pred " 作为负代价（即奖励增长）
    cost += - (W_GROW * far) * dT_pred;

    // 接近目标时：惩罚正向加速（慎重加速）
    if (dv > 0) {
        cost += W_DV_POS * near * dv * dv;
    }

    // 平滑/能量惩罚（可选）
    cost += W_DV_ALL * dv * dv;
    cost += W_V     * v_next * v_next;

    // 每敲一次固定代价（越大越倾向少敲）
    cost += W_HIT;

    return cost;
}

// ===================== 4) MPC 穷举（H=2 示例） =====================
float mpc_choose_next_speed(const MPCState* s0)
{
    float bestJ = 1e30f;
    float best_v1 = s0->v_last;

    // Horizon = 2：枚举 (a0, a1)
    for(int i0=0;i0<N_ACTIONS;i0++){
        float dv0 = DV_SET[i0];
        float v1  = clampf(s0->v_last + dv0, V_MIN, V_MAX);

        MPCState s1 = *s0;
        float z1;
        float T1 = predict_next_T(&s1, v1, &z1);

        // 推进到下一步的“预测状态”
        s1.T_last = T1;
        s1.v_last = v1;
        s1.z_hat  = z1;

        float J0 = stage_cost(s0->T_last, T1, s0->T_target, dv0, v1);

        for(int i1=0;i1<N_ACTIONS;i1++){
            float dv1 = DV_SET[i1];
            float v2  = clampf(s1.v_last + dv1, V_MIN, V_MAX);

            float z2;
            float T2 = predict_next_T(&s1, v2, &z2);

            float J1 = stage_cost(s1.T_last, T2, s1.T_target, dv1, v2);

            float J = J0 + GAMMA_H * J1;

            if(J < bestJ){
                bestJ = J;
                best_v1 = v1; // receding horizon：只输出第一步
            }
        }
    }

    return best_v1;
}
