% version1.m - Baseline monitoring per Algorithm 1
% Path A: static X-chart (Phase I from first N_CAL points, 3-sigma)
% Path B: sliding-window 3rd-order polynomial regression + 10-step forecast
% Reads sensor_data.csv periodically; plots T (top) and H (bottom).

CSV_PATH   = fullfile('..', 'analysis', 'sensor_data.csv');
N_CAL      = 50;
WIN_SIZE   = 110;
FIT_SIZE   = 100;
PRED_SIZE  = 10;
POLY_ORDER = 3;
REFRESH_S  = 2;

fig = figure('Name', 'Baseline monitoring', 'Position', [100 100 1100 700]);

while ishandle(fig)
    if ~isfile(CSV_PATH)
        pause(REFRESH_S); continue;
    end
    S = readtable(CSV_PATH);
    if height(S) < N_CAL + 5
        pause(REFRESH_S); continue;
    end

    plot_channel(1, S.Temperature, 'Temperature (C)', ...
        N_CAL, WIN_SIZE, FIT_SIZE, PRED_SIZE, POLY_ORDER);
    plot_channel(2, S.Humidity,    'Humidity (%)', ...
        N_CAL, WIN_SIZE, FIT_SIZE, PRED_SIZE, POLY_ORDER);

    drawnow;
    pause(REFRESH_S);
end


function plot_channel(row, y, ylbl, N_cal, W, Nfit, Npred, order)
    mu  = mean(y(1:N_cal));
    sig = std(y(1:N_cal));
    UCL = mu + 3 * sig;
    LCL = mu - 3 * sig;

    n = length(y);
    t = (1:n)';

    % Path A: X-chart
    subplot(2, 2, (row - 1) * 2 + 1);
    plot(t, y, 'b.-'); hold on;
    yline(UCL, 'r--', 'UCL');
    yline(LCL, 'r--', 'LCL');
    yline(mu,  'g-',  'Mean');
    oob = (y > UCL) | (y < LCL);
    plot(t(oob), y(oob), 'ro', 'MarkerFaceColor', 'r');
    title([ylbl ' - X-chart']);
    xlabel('t'); ylabel(ylbl); grid on; hold off;

    % Path B: window polynomial fit + 10-step forecast
    subplot(2, 2, (row - 1) * 2 + 2);
    if n >= W
        win = y(n - W + 1 : n);
        tr  = win(1:Nfit);
        xr  = (1:Nfit)';
        p   = polyfit(xr, tr, order);
        xp  = (Nfit + 1 : Nfit + Npred)';
        yp  = polyval(p, xp);

        t_win  = (n - W + 1 : n)';
        t_fit  = t_win(1:Nfit);
        t_pred = (n + 1 : n + Npred)';

        plot(t_win, win, 'b.-'); hold on;
        plot(t_fit, polyval(p, xr), 'g-',  'LineWidth', 1.5);
        plot(t_pred, yp,            'r--', 'LineWidth', 1.5);
        yline(UCL, 'r:');
        yline(LCL, 'r:');
        if any(yp > UCL | yp < LCL)
            title([ylbl ' - polynomial forecast (ALERT)']);
        else
            title([ylbl ' - polynomial forecast']);
        end
        xlabel('t'); ylabel(ylbl); grid on; hold off;
    end
end
