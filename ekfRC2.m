function [x_out, P_out] = ekf(u, y, T, x, P)

Ts = 0.01;

% ---- Parameters ----
R0_ref = 0.005;
R1_ref = 0.01;
R2_ref = 0.02;

alpha0 = 0.03;
alpha1 = 0.05;
alpha2 = 0.05;

R0 = R0_ref*(1 + alpha0*(25 - T));
R1 = R1_ref*(1 + alpha1*(25 - T));
R2 = R2_ref*(1 + alpha2*(25 - T));

Qb = 7200;

C1 = 200;
C2 = 500;

a1 = exp(-Ts/(R1*C1));
a2 = exp(-Ts/(R2*C2));

% ---- States ----
SOC = x(1);
V1  = x(2);
V2  = x(3);
bias = x(4);   % NEW

I_meas = u;

% ---- Prediction ----
SOC_pred = SOC - Ts/Qb * (I_meas - bias);   % corrected current
V1_pred  = a1*V1 + R1*(1-a1)*(I_meas - bias);
V2_pred  = a2*V2 + R2*(1-a2)*(I_meas - bias);

bias_pred = bias;  % bias assumed slowly varying

x_pred = [SOC_pred; V1_pred; V2_pred; bias_pred];

% ---- OCV lookup ----
soc_bp = [0 0.1 0.2 0.4 0.6 0.8 1];
ocv_bp = [3.0 3.3 3.5 3.7 3.85 4.0 4.2];

OCV = interp1(soc_bp, ocv_bp, SOC_pred, 'linear', 'extrap');

% ---- Output prediction ----
y_pred = OCV - R0*(I_meas - bias) - V1_pred - V2_pred;

% ---- dOCV/dSOC ----
dSOC = 1e-4;
OCV_p = interp1(soc_bp, ocv_bp, SOC_pred + dSOC,'linear','extrap');
OCV_m = interp1(soc_bp, ocv_bp, SOC_pred - dSOC,'linear','extrap');

dOCV = (OCV_p - OCV_m)/(2*dSOC);

% ---- Jacobians ----
F = [1 0 0 Ts/Qb;
    0 a1 0 -R1*(1-a1);
    0 0 a2 -R2*(1-a2);
    0 0 0 1];

H = [dOCV  -1  -1   R0];

% ---- Noise tuning ----
Qk = diag([1e-6, 5e-7, 5e-7, 5e-5]);   % bias gets higher uncertainty
Rk = 5e-5;

% ---- Covariance ----
P_pred = F*P*F' + Qk;

% ---- Kalman Gain ----
K = P_pred*H'/(H*P_pred*H' + Rk);

% ---- Update ----
x_out = x_pred + K*(y - y_pred);

P_out = (eye(4) - K*H)*P_pred;
x_out(4) = max(min(x_out(4),0.1),-0.1);

end